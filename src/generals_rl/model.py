"""Policy/value network: BoardTransformer (CNN stem, transformer, CNN decoder).

Input: the observation dict of features.observe (batched). Output:
    logits (B, H*W*8 + 1)  flat action space (per cell: direction * 2 + split, then pass)
    value  (B,)            expected return in [-1, 1] from the viewer's perspective

1. Stem: residual 3x3 convolution blocks over the cells; their features are added to the planes, so every cell
   already knows its neighbourhood, also across patch borders.
2. Transformer: the board is cut into patch x patch patches, one token each, plus one token per player slot (public
   scoreboard) and a CLS token. Attention is global, and every layer adds a learned bias per head and per offset
   (rows, cols) between two board tokens, as in the Swin transformer, so the same relation means the same thing
   anywhere on the board and on every board size; absolute position comes from the coordinate planes. Board sizes up
   to max_board and 2..8 players work with the same weights.
3. Decoder: the move logits come from a small CNN over the cells that joins each cell's token output with the stem
   features (a skip connection as in a U-Net). Pass and value come from the CLS token.

The value head outputs a distribution over value_bins equal bins on [-1, 1] and is trained with a cross-entropy
against a Gaussian of width sigma around the target (HL-Gauss, Farebrother et al. 2024, as in Straka et al. 2026);
the value is the distribution's mean.

The defaults are the size of Straka, Lisy & Schmid 2026 (dim 448, depth 7, 3x3 patches): 15.5M parameters.
"""
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

from .features import MAX_PLAYERS, NUM_DIRECTIONS, NUM_SPLITS

MOVE_LOGITS = NUM_DIRECTIONS * NUM_SPLITS  # 8 per cell


class BiasedAttention(nn.Module):
    """Multi-head self-attention with an additive bias (heads, T, T) on the attention logits."""
    dim: int
    heads: int

    @nn.compact
    def __call__(self, x, mask, bias):
        b, t, _ = x.shape
        hd = self.dim // self.heads
        q, k, v = jnp.split(nn.Dense(3 * self.dim)(x).reshape(b, t, 3 * self.heads, hd), 3, axis=2)
        logits = jnp.einsum("bqhd,bkhd->bhqk", q, k) / jnp.sqrt(hd) + bias[None]
        weights = jax.nn.softmax(jnp.where(mask, logits, -1e9), axis=-1)
        return nn.Dense(self.dim)(jnp.einsum("bhqk,bkhd->bqhd", weights, v).reshape(b, t, self.dim))


def relative_bias(table: jnp.ndarray, gh: int, gw: int, extra: int) -> jnp.ndarray:
    """(heads, T, T) bias for T = extra + gh * gw tokens: table[h, dr, dc] between two board tokens, 0 elsewhere."""
    g = (table.shape[1] + 1) // 2
    r, c = jnp.divmod(jnp.arange(gh * gw), gw)
    board = table[:, r[:, None] - r[None] + g - 1, c[:, None] - c[None] + g - 1]
    return jnp.pad(board, ((0, 0), (extra, 0), (extra, 0)))


class Block(nn.Module):
    dim: int
    heads: int
    ff_factor: int
    grid: tuple                                         # (gh, gw, g, extra tokens) for the relative bias

    @nn.compact
    def __call__(self, x, mask):
        h = nn.LayerNorm()(x)
        gh, gw, g, extra = self.grid
        table = self.param("rel_bias", nn.initializers.zeros, (self.heads, 2 * g - 1, 2 * g - 1))
        x = x + BiasedAttention(self.dim, self.heads)(h, mask, relative_bias(table, gh, gw, extra))
        h = nn.LayerNorm()(x)
        h = nn.Dense(self.dim * self.ff_factor)(h)
        return x + nn.Dense(self.dim)(nn.gelu(h))


class BoardTransformer(nn.Module):
    dim: int = 448
    depth: int = 7
    heads: int = 8
    ff_factor: int = 3                  # transformer MLP width as a multiple of dim
    patch: int = 3
    max_board: int = 36
    stem: int = 2                       # residual blocks (two 3x3 convolutions each) over the cells
    stem_dim: int = 64
    value_bins: int = 128

    @classmethod
    def from_config(cls, cfg, max_board: int = 36):
        """The model of a run's config (a dataclass or the dict from config.json)."""
        get = (lambda k: cfg[k]) if isinstance(cfg, dict) else (lambda k: getattr(cfg, k))
        return cls(dim=get("dim"), depth=get("depth"), heads=get("heads"), ff_factor=get("ff_factor"),
                   patch=get("patch"), stem=get("stem"), stem_dim=get("stem_dim"), value_bins=get("value_bins"),
                   max_board=max_board)

    @nn.compact
    def __call__(self, obs, value_logits: bool = False):
        """(move logits, value); with value_logits also the value head's bin logits."""
        planes, players, pmask = obs["planes"], obs["players"], obs["player_mask"]
        b, h, w, _ = planes.shape
        p = self.patch
        gh, gw = h // p, w // p
        g = self.max_board // p

        conv = lambda k: nn.Conv(self.stem_dim, (k, k), padding="SAME")
        cells = conv(1)(planes)                         # per-cell features, (B, H, W, stem_dim)
        for _ in range(self.stem):                      # pre-norm residual blocks, as in the transformer
            cells = cells + conv(3)(nn.gelu(conv(3)(nn.gelu(nn.LayerNorm()(cells)))))
        planes = jnp.concatenate([planes, cells], -1)
        c = planes.shape[-1]

        # Board tokens: patch -> linear embedding.
        x = planes.reshape(b, gh, p, gw, p, c).transpose(0, 1, 3, 2, 4, 5).reshape(b, gh * gw, p * p * c)
        board = nn.Dense(self.dim)(x)

        # Player tokens: scoreboard row + slot embedding (slot 0 = me).
        slot = self.param("slot", nn.initializers.normal(0.02), (MAX_PLAYERS, self.dim))
        ptok = nn.Dense(self.dim)(players) + slot[None]
        cls = jnp.broadcast_to(self.param("cls", nn.initializers.normal(0.02), (1, 1, self.dim)), (b, 1, self.dim))

        tokens = jnp.concatenate([cls, ptok, board], axis=1)
        valid = jnp.concatenate([jnp.ones((b, 1), bool), pmask, jnp.ones((b, gh * gw), bool)], axis=1)
        attn_mask = nn.make_attention_mask(valid, valid)
        for _ in range(self.depth):
            tokens = Block(self.dim, self.heads, self.ff_factor, (gh, gw, g, 1 + MAX_PLAYERS))(tokens, attn_mask)
        tokens = nn.LayerNorm()(tokens)

        # Decoder: token outputs back onto their cells, joined with the stem features.
        cls_out, board_out = tokens[:, 0], tokens[:, 1 + MAX_PLAYERS:]
        up = nn.Dense(p * p * self.stem_dim)(board_out)
        up = up.reshape(b, gh, gw, p, p, self.stem_dim).transpose(0, 1, 3, 2, 4, 5).reshape(b, h, w, self.stem_dim)
        z = nn.gelu(conv(3)(nn.gelu(jnp.concatenate([up, cells], -1))))
        moves = nn.Conv(MOVE_LOGITS, (1, 1))(z).reshape(b, -1)                     # (B, H*W*8)
        pass_logit = nn.Dense(1)(cls_out)
        head = nn.Dense(self.value_bins)                 # created before the hidden layer (parameter names)
        vlogits = head(nn.gelu(nn.Dense(self.dim)(cls_out)))
        value = (jax.nn.softmax(vlogits) * bin_centers(self.value_bins)).sum(-1)
        out = (jnp.concatenate([moves, pass_logit], -1), value)
        return out + (vlogits,) if value_logits else out


def bin_centers(bins: int, lo: float = -1.0, hi: float = 1.0) -> jnp.ndarray:
    edges = jnp.linspace(lo, hi, bins + 1)
    return (edges[:-1] + edges[1:]) / 2


def hl_gauss_target(target: jnp.ndarray, bins: int, sigma: float, lo: float = -1.0, hi: float = 1.0) -> jnp.ndarray:
    """(..., bins) probabilities of a Gaussian N(target, sigma^2) integrated over each bin, renormalized to the range."""
    edges = jnp.linspace(lo, hi, bins + 1)
    cdf = norm.cdf((edges - jnp.clip(target, lo, hi)[..., None]) / sigma)
    mass = cdf[..., 1:] - cdf[..., :-1]
    return mass / mass.sum(-1, keepdims=True)


def value_loss(vlogits, target, sigma: float):
    """HL-Gauss cross-entropy of the value head's bin logits against the targets."""
    probs = hl_gauss_target(target, vlogits.shape[-1], sigma)
    return -jnp.mean((probs * jax.nn.log_softmax(vlogits)).sum(-1))


def masked_logits(logits, mask):
    return jnp.where(mask, logits, -1e9)
