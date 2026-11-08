"""Behavior cloning on FFA replays (data/processed/<set>.npz from scripts/prepare_replays.py): the clone of a player
(by default Human.exe, the strongest FFA bot) that self-play starts from.

Each "chunk": pick a few games, replay them through the simulator on the device (the game state of every
tick), sample (game, tick, player) positions of the imitated players, turn those into observations with
features.observe and take several gradient steps on them. The games themselves are never filtered; the other
players stay in the game as opponents, only their moves are not used as targets.

A position is used when
    - the player is one of `imitate` and has at least min_player_stars and min_player_moves moves in the game,
      and did not go AFK before early_quit_ticks;
    - it is not in the last afk_cut_ticks before the player's first AFK event;
    - the recorded move is executable and in the action mask; of the ticks without a move, pass_keep are kept.
The value target is the player's final placement mapped to [-1, 1].
"""
import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import optax
from flax import serialization

from .data import replays
from .evaluate import eval_vs_bot, make_eval_env
from .features import (DIRECTIONS, HISTORY_TICKS, PASS, SCORE_HISTORY, TRAIL_DECAY, _move_cells, _remember as remember,
                       encode_action, history_init, observe, scoreboard)
from .model import BoardTransformer, masked_logits, value_loss

PAD, NP = replays.PAD, replays.NP


@dataclass
class Config:
    data: str = "data/processed/humanexe_ffa.npz"
    out: str = "runs/clone"
    pad: int = PAD                     # board size, the same as self-play's
    # model (model.py; the self-play run that starts from the clone needs the same)
    dim: int = 448
    depth: int = 7
    heads: int = 8
    patch: int = 3
    ff_factor: int = 3
    stem: int = 2
    stem_dim: int = 64
    value_bins: int = 128
    hl_sigma: float = 0.04
    # optimization: about 0.4 s per step at batch 128 on a 16 GB GPU (256 does not fit)
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 128
    total_steps: int = 8_000
    games_per_chunk: int = 16
    steps_per_chunk: int = 16
    value_coef: float = 0.5
    # who / what to imitate
    imitate: str = "Human.exe"         # comma-separated usernames, or a file with one per line
    min_player_stars: int = 0
    early_quit_ticks: int = 100
    afk_cut_ticks: int = 50
    min_player_moves: int = 20
    pass_keep: float = 1.0             # Human.exe waits on purpose in the opening; fewer passes make the clone rush
    # bookkeeping
    val_frac: float = 0.05
    log_every: int = 100
    eval_every: int = 2_000
    eval_games: int = 32
    eval_size: int = 27                # median 8-player FFA map
    eval_players: int = 8
    eval_bots: str = "expander"        # harvester is much slower (BFS per tick per bot)
    save_every: int = 2_000
    time_limit_hours: float = 24.0
    seed: int = 0


# ---------------------------------------------------------------- data

def imitated_names(spec: str) -> set:
    """Usernames from a file (one per line) or a comma-separated list."""
    path = Path(spec)
    lines = path.read_text().splitlines() if path.is_file() else spec.split(",")
    return {x.strip() for x in lines if x.strip()}


class Dataset:
    def __init__(self, path: str, cfg: Config):
        self.d = np.load(path)
        self.meta = json.loads(str(self.d["meta"]))
        self.n_games = len(self.meta)
        n_players, n_ticks = self.d["n_players"], self.d["n_ticks"]
        stars, first_afk, n_moves = self.d["stars"], self.d["first_afk"], self.d["n_moves"]
        slot = np.arange(NP)[None]

        eligible = (slot < n_players[:, None]) & (stars >= cfg.min_player_stars) & (n_moves >= cfg.min_player_moves)
        eligible &= ~((first_afk >= 0) & (first_afk < cfg.early_quit_ticks))
        if cfg.imitate:
            names = imitated_names(cfg.imitate)
            listed = np.zeros_like(eligible)
            for g, m in enumerate(self.meta):
                for p, u in enumerate(m["usernames"][:NP]):
                    listed[g, p] = u in names
            eligible &= listed
        self.eligible = eligible
        self.until = np.where(first_afk >= 0, np.maximum(first_afk - cfg.afk_cut_ticks, 0), n_ticks[:, None])
        self.map_hw = np.array([[m["map"][1], m["map"][0]] for m in self.meta], np.int32)   # rows, cols of the map
        self.pad = cfg.pad
        self.eligible &= (self.map_hw.max(1) <= cfg.pad)[:, None]                          # maps that fit the board

        print(f"{self.n_games} games, {int(eligible.sum())} imitated player-games "
              f"({eligible.any(1).sum()} games with at least one)")

    def chunk(self, games):
        """Replay the given games on the device. Returns states (G, T, ...) and, host side, legal (G, T, NP),
        eliminated (G, T, NP), dense actions (G, T, NP, 5), placements (G, NP), n_ticks (G,)."""
        loaded = [replays.load_game(self.d, g) for g in games]
        tp = replays.bucket(max(x[4] for x in loaded))
        grids = np.stack([x[0][: self.pad, : self.pad] for x in loaded])    # the cut part is padding (mountains)
        actions = np.stack([replays.pad_ticks(x[1], tp, replays.PASS) for x in loaded])
        afk = np.stack([replays.pad_ticks(x[2], tp) for x in loaded])
        teams = np.stack([x[3] for x in loaded])
        states, legal, _, final_elim = simulate_batch(grids, actions, afk, teams)
        n_ticks = np.array([x[4] for x in loaded])
        elim = np.asarray(states.eliminated)
        final_elim = np.asarray(final_elim)
        place = np.ones((len(games), NP), np.int32)
        for i, g in enumerate(games):
            n = int(self.d["n_players"][g])
            place[i, :n] = replays.placements(elim[i, : n_ticks[i]], final_elim[i], n)
        score = scoreboards(states)                                       # (G, T, NP, 2), for the history features
        return (states, score), np.asarray(legal), elim, jnp.asarray(actions), place, n_ticks


simulate_batch = jax.jit(jax.vmap(replays.simulate))
scoreboards = jax.jit(jax.vmap(jax.vmap(scoreboard)))


def sample_positions(rng, ds: Dataset, games, legal, elim, actions, n_ticks, cfg: Config, k: int):
    """Host side: k (game index in chunk, tick, player) positions of imitated players that can be used (see the top)."""
    t_idx = np.arange(legal.shape[1])[None, :, None]
    ok = legal & ~elim & ds.eligible[games][:, None, :]
    ok &= t_idx < np.minimum(ds.until[games], n_ticks[:, None])[:, None, :]
    is_pass = np.asarray(actions[..., 0]) == 1
    ok &= ~is_pass | (rng.random(is_pass.shape) < cfg.pass_keep)
    g, t, p = np.nonzero(ok)
    if len(g) == 0:
        return None
    sel = rng.choice(len(g), size=k, replace=len(g) < k)
    return g[sel], t[sel], p[sel]


def is_new_path(now, prev, t):
    """A move that does not start on the cell the player's move of the previous tick led to (actions (..., 5))."""
    target = prev[..., 1:3] + DIRECTIONS[prev[..., 3]]
    continues = (t > 0) & (prev[..., 0] == 0) & (now[..., 1:3] == target).all(-1)
    return (now[..., 0] == 0) & ~continues


def replay_memory(states, actions, g, t):
    """features' memory of each sample: what every player had seen in game g before tick t and where its own moves
    went, built by replaying the ticks 0 .. t-1 like history_push does. Runs only up to the largest t of the batch."""
    first = jax.tree.map(lambda x: x[g, 0], states)
    memory = jax.vmap(history_init)(first)["memory"]
    h, w = first.armies.shape[-2:]

    def body(s, memory):
        seen = jax.vmap(remember)(memory, jax.tree.map(lambda x: x[g, s], states))
        moved = jax.vmap(lambda a: _move_cells(a, h, w))(actions[g, s])
        seen["trail"] = jnp.maximum(memory["trail"] * TRAIL_DECAY, moved)
        keep = s < t
        return jax.tree.map(lambda new, old: jnp.where(keep.reshape((-1,) + (1,) * (new.ndim - 1)), new, old), seen, memory)
    return jax.lax.fori_loop(0, jnp.max(t), body, memory)


@jax.jit
def make_batch(sim, actions, place, n_players, g, t, p):
    states, score = sim
    st = jax.tree.map(lambda x: x[g, t], states)
    # history of each sample: states/actions at t-1 ... t-HISTORY_TICKS, scoreboard at t-1 ... t-SCORE_HISTORY;
    # before the first tick the game start stands in (no moves, unchanged scoreboard), like features.history_init
    hi = t[:, None] - 1 - jnp.arange(HISTORY_TICKS)[None]
    hist = dict(
        states=jax.tree.map(lambda x: x[g[:, None], jnp.maximum(hi, 0)], states),
        actions=jnp.where((hi >= 0)[..., None, None], actions[g[:, None], jnp.maximum(hi, 0)], jnp.array(PASS, jnp.int32)),
        score=score[g[:, None], jnp.maximum(t[:, None] - 1 - jnp.arange(SCORE_HISTORY)[None], 0)],
        memory=replay_memory(states, actions, g, t),
    )
    obs = jax.vmap(observe)(st, p, hist)
    board = st.armies.shape[-1]
    target = encode_action(actions[g, t, p], board, board)
    weight = jnp.take_along_axis(obs["action_mask"], target[:, None], 1)[:, 0].astype(jnp.float32)  # in the mask
    rank01 = (place[g, p] - 1) / jnp.maximum(n_players[g] - 1, 1)          # 0 = first place, 1 = last
    new_path = is_new_path(actions[g, t, p], actions[g, jnp.maximum(t - 1, 0), p], t)
    return dict(obs=obs, target=target, weight=weight, new_path=new_path, value_target=1.0 - 2.0 * rank01)


# ---------------------------------------------------------------- training

def make_fns(model, tx, cfg: Config):
    def loss_fn(params, b):
        logits, _, vlogits = model.apply(params, b["obs"], value_logits=True)
        logp = jax.nn.log_softmax(masked_logits(logits, b["obs"]["action_mask"]))
        nll = -jnp.take_along_axis(logp, b["target"][:, None], 1)[:, 0]
        w = b["weight"]
        ce = (nll * w).sum() / jnp.maximum(w.sum(), 1e-6)
        vloss = value_loss(vlogits, b["value_target"], cfg.hl_sigma)
        hit = jnp.argmax(logp, -1) == b["target"]
        acc = (hit * (w > 0)).sum() / jnp.maximum((w > 0).sum(), 1)
        new = b["new_path"] & (w > 0)
        acc_new = (hit * new).sum() / jnp.maximum(new.sum(), 1)
        ce_new = (nll * new).sum() / jnp.maximum(new.sum(), 1)
        # acc_new / ce_new: moves that start a new path (not continuing the last move), the hard part to imitate
        return ce + cfg.value_coef * vloss, dict(ce=ce, value_loss=vloss, acc=acc, acc_new=acc_new, ce_new=ce_new)

    @jax.jit
    def train_step(params, opt_state, b):
        (_, m), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, b)
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, m

    return jax.jit(loss_fn), train_step


def train(cfg: Config):
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=1))
    log_file = open(out / "metrics.jsonl", "a")
    t_start = time.time()

    def log(step, **m):
        m = {k: float(v) for k, v in m.items()}
        log_file.write(json.dumps({"step": step, "time_h": (time.time() - t_start) / 3600, **m}) + "\n")
        log_file.flush()
        print(f"step {step:>7}  " + "  ".join(f"{k} {v:.4f}" for k, v in m.items()), flush=True)

    print("devices:", jax.devices())
    ds = Dataset(cfg.data, cfg)
    rng = np.random.default_rng(cfg.seed)
    usable = np.nonzero(ds.eligible.any(1))[0]
    perm = rng.permutation(usable)
    n_val = max(1, int(len(perm) * cfg.val_frac))
    val_games, train_games = perm[:n_val], perm[n_val:]
    print(f"{len(train_games)} train games, {len(val_games)} validation games")

    env = make_eval_env(cfg.eval_size, cfg.eval_players, pad_to=cfg.pad)
    pool, _ = env.reset(jrandom.PRNGKey(123))
    model = BoardTransformer.from_config(cfg, cfg.pad)
    key = jrandom.PRNGKey(cfg.seed)
    dummy = jax.tree.map(lambda x: x[None], observe(env.init_state(key), 0))
    params = model.init(key, dummy)
    print("parameters:", sum(x.size for x in jax.tree.leaves(params)))
    sched = optax.warmup_cosine_decay_schedule(0.0, cfg.lr, min(1000, cfg.total_steps // 10), cfg.total_steps)
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=cfg.weight_decay))
    opt_state = tx.init(params)
    loss_fn, train_step = make_fns(model, tx, cfg)
    n_players = jnp.asarray(ds.d["n_players"])

    def batch_from(games, states, legal, elim, actions, place, n_ticks, k, rng_):
        idx = sample_positions(rng_, ds, games, legal, elim, actions, n_ticks, cfg, k)
        if idx is None:
            return None
        g, t, p = idx
        return make_batch(states, actions, place, n_players[games], g, t, p)

    vg = val_games[: cfg.games_per_chunk]
    val_batch = batch_from(vg, *ds.chunk(vg), cfg.batch_size * 4, np.random.default_rng(1))

    bots = [b for b in cfg.eval_bots.split(",") if b]
    eval_fns = {b: jax.jit(lambda prm, pool, k, b=b: eval_vs_bot(model.apply, prm, b, env, pool, k, cfg.eval_games))
                for b in bots}

    def save(tag):
        (out / f"params_{tag}.msgpack").write_bytes(serialization.to_bytes(params))

    step, metrics = 0, []
    while step < cfg.total_steps:
        games = rng.choice(train_games, min(cfg.games_per_chunk, len(train_games)), replace=False)
        states, legal, elim, actions, place, n_ticks = ds.chunk(games)
        idx = sample_positions(rng, ds, games, legal, elim, actions, n_ticks, cfg, cfg.batch_size * cfg.steps_per_chunk)
        if idx is None:
            continue
        for s in range(min(cfg.steps_per_chunk, cfg.total_steps - step)):
            sl = slice(s * cfg.batch_size, (s + 1) * cfg.batch_size)
            g, t, p = (x[sl] for x in idx)
            b = make_batch(states, actions, place, n_players[games], g, t, p)
            params, opt_state, m = train_step(params, opt_state, b)
            metrics.append(m)
            step += 1
            if step % cfg.log_every == 0:
                avg = jax.tree.map(lambda *x: np.mean(x), *metrics)
                metrics = []
                vm = loss_fn(params, val_batch)[1] if val_batch is not None else {}
                log(step, **{f"train_{k}": v for k, v in avg.items()}, **{f"val_{k}": v for k, v in vm.items()})
            if step % cfg.eval_every == 0:
                for bot, fn in eval_fns.items():
                    win, place_ = fn(params, pool, jrandom.PRNGKey(step))
                    log(step, **{f"win_vs_{bot}": win, f"place_vs_{bot}": place_})
            if step % cfg.save_every == 0:
                save(step)
                save("latest")
            if time.time() - t_start > cfg.time_limit_hours * 3600:
                print("time limit reached")
                cfg.total_steps = step
                break
    save("latest")
    print("done, checkpoints in", out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name, default in asdict(Config()).items():
        p.add_argument("--" + name.replace("_", "-"), type=type(default), default=default)
    train(Config(**vars(p.parse_args())))


if __name__ == "__main__":
    main()
