"""Play one free-for-all game live, tick by tick: a trained policy in seat 0, opponents in the other seats.

Used by scripts/watch.py (live viewer). Each `advance(n)` call runs n jit-compiled ticks and returns compact frames
the browser can draw. The board is padded to 36 x 36 like in training; frames are cropped to the real map.
"""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from flax import serialization

from generals import GeneralsEnv
from generals.core import game
from generals.core.game import get_visibility

from .data import replays
from .evaluate import BOTS, bot_act
from .features import PLANE_NAMES, PLAYER_FEATURE_NAMES, decode_action, history_init, history_push, observe
from .mapgen import REAL_SIDES, realistic_map
from .model import BoardTransformer, masked_logits

PAD = 36
REAL_CITIES = (18, 28)            # for generated maps of a fixed size; real 8-player maps have 17-32 cities


def load_policy(run_dir: str, ckpt: str = "latest"):
    """(apply_fn, params, config dict) of a training run directory (config.json + params_<ckpt>.msgpack)."""
    run = Path(run_dir)
    cfg = json.loads((run / "config.json").read_text())
    model = BoardTransformer.from_config(cfg, PAD)
    env = GeneralsEnv(grid_dims=(20, 20), pad_to=PAD, num_players=8, min_generals_distance=5, pool_size=2)
    dummy = jax.tree.map(lambda x: x[None], observe(env.init_state(jrandom.PRNGKey(0)), 0))
    template = model.init(jrandom.PRNGKey(0), dummy)
    params = serialization.from_bytes(template, (run / f"params_{ckpt}.msgpack").read_bytes())
    return model.apply, jax.tree.map(jnp.asarray, params), cfg       # numpy leaves break the relative position bias


def realistic_state(seed: int, players: int = 8):
    """A map like a real FFA map (mapgen.py): size, spawns and cities drawn like real ones, at the top left."""
    rng = np.random.default_rng(seed)
    sides, p = np.array(list(REAL_SIDES)), np.array(list(REAL_SIDES.values()), float)
    h, w = (int(x) for x in rng.choice(sides, 2, p=p / p.sum()))
    env = GeneralsEnv(grid_dims=(h, w), pad_to=PAD, num_players=players, min_generals_distance=10,
                      num_castles_range=REAL_CITIES, pool_size=1)
    for attempt in range(20):
        state = env.init_state(jrandom.PRNGKey(seed * 100 + attempt))
        m = realistic_map(jax.tree.map(np.asarray, state)._asdict(), players, rng)
        if m is not None:
            return type(state)(**jax.tree.map(jnp.asarray, m)), (h, w)
    raise RuntimeError("no room for real spawns on 20 maps in a row")


class LiveGame:
    """One game: the policy in seat 0, `bots` in the next seats.

    board: "real" for a map like real FFA maps (mapgen.py, 8 players), "replay" for the map of a real game in `data`
           (its mountains, cities and spawns, with as many seats as that game had), or a side length for a generated
           square map with len(bots) + 1 players.
    """

    def __init__(self, run_dir: str, bots: list[str], board: int | str = "real", seed: int = 0, greedy: bool = False,
                 ckpt: str = "latest", data: str = "data/processed/humanexe_ffa.npz",
                 past: tuple[str, str] | None = None):
        """bots: per seat after the policy a scripted bot, "self" (a copy of the policy) or "past" (the checkpoint
        `past` = (run directory, checkpoint)), or "humanexe": then every opponent is Human.exe and the game runs in its
        own simulator (humanexe.py), one turn per frame."""
        apply_fn, params, cfg = load_policy(run_dir, ckpt)
        past_params = params
        if "past" in bots:
            if past is None:
                raise ValueError("opponents of type past need an opponent checkpoint")
            _, past_params, _ = load_policy(*past)
        key = jrandom.PRNGKey(seed)
        if board == "replay":
            d = np.load(data)
            g = seed % len(d["grids"])
            self.state, (self.h, self.w), n, users = replays.start_state(d, g)
            bots = bots[: n - 1]
            self.map_info = f"real map {json.loads(str(d['meta']))[g]['id']} ({self.w}×{self.h}, originally {', '.join(users)})"
        elif board == "real":
            self.state, (self.h, self.w) = realistic_state(seed)
            bots = bots[:7]
            self.map_info = f"realistic map {self.w}×{self.h}"
        else:
            size = int(board)
            env = GeneralsEnv(grid_dims=(size, size), pad_to=PAD, num_players=len(bots) + 1, min_generals_distance=10,
                              num_castles_range=REAL_CITIES, pool_size=2)
            self.state = env.init_state(key)
            self.h = self.w = size
            self.map_info = f"generated map {size}×{size}"
        self.key = jrandom.fold_in(key, 1)
        self.humanexe = None
        if "humanexe" in bots:                               # Human.exe takes every opponent seat
            from .humanexe import HumanExe
            bots = ["humanexe"] * len(bots)
            self.humanexe = HumanExe(log_path=str(Path(run_dir).resolve().parents[1] / "runs" / "humanexe_host.log"))
            self.state = self.humanexe.start_game(self.state, [0])
        self.n = len(bots) + 1
        label = {"self": "Copy", "past": f"Older ({past[1] if past else ''})", "humanexe": "Human.exe"}
        self.names = ["Policy"] + [f"{label.get(b, b.capitalize())} {i + 1}" for i, b in enumerate(bots)]
        seats = self.state.ownership.shape[0]                    # 8 for replay maps (padded), n otherwise
        # Seats played by a network, grouped by weights so each group needs one forward pass per tick.
        groups = [(params, [0] + [i + 1 for i, b in enumerate(bots) if b == "self"]),
                  (past_params, [i + 1 for i, b in enumerate(bots) if b == "past"])]
        groups = [(prm, jnp.array(ss)) for prm, ss in groups if ss]
        scripted = [(i + 1, bot_act(b)) for i, b in enumerate(bots) if b not in ("self", "past", "humanexe")]
        h, w = self.h, self.w

        def tick(state, hist, key):
            batched = jax.tree.map(lambda x: x[None], state)
            actions = jnp.tile(jnp.array([1, 0, 0, 0, 0], jnp.int32), (seats, 1))     # empty seats pass
            for g, (prm, ss) in enumerate(groups):
                obs = jax.vmap(lambda me: observe(state, me, hist))(ss)
                logits, _ = apply_fn(prm, obs)
                logits = masked_logits(logits, obs["action_mask"])
                idx = jnp.argmax(logits, -1) if greedy else jrandom.categorical(jrandom.fold_in(key, 100 + g), logits)
                board = state.armies.shape[-1]
                actions = actions.at[ss].set(decode_action(idx, board, board))
            for i, act in scripted:
                actions = actions.at[i].set(act(batched, i, jrandom.fold_in(key, i))[0])
            new, _ = game.step(state, actions, general_trade=True)
            return new, history_push(hist, state, actions), actions

        def frame(state, hist):
            own = state.ownership[:, :h, :w]
            # What the policy thinks in this position: its value estimate and, per cell, the probability of the next move.
            obs = jax.tree.map(lambda x: x[None], observe(state, 0, hist))
            logits, value = apply_fn(params, obs)
            probs = jax.nn.softmax(masked_logits(logits, obs["action_mask"]))[0]
            board = state.armies.shape[0]
            moves = probs[:-1].reshape(board, board, 4, 2).sum(-1)[:h, :w]    # (h, w, directions)
            owner = jnp.where(own.any(0), own.argmax(0), -1)
            land = own.sum((1, 2))
            army = (state.armies[None, :h, :w] * own).sum((1, 2))
            seen = get_visibility(state.ownership[0])[:h, :w]
            return dict(owner=owner, armies=state.armies[:h, :w], generals=state.generals[:h, :w],
                        castles=state.castles[:h, :w], seen=seen, land=land, army=army,
                        eliminated=state.eliminated, time=state.time, winner=state.winner,
                        value=value[0], move_probs=moves, pass_prob=probs[-1],
                        planes=obs["planes"][0][:h, :w], players=obs["players"][0], player_mask=obs["player_mask"][0])

        def policy_move(state, hist, key):                   # seat 0 only, for games against Human.exe
            obs = jax.tree.map(lambda x: x[None], observe(state, 0, hist))
            logits, _ = apply_fn(params, obs)
            logits = masked_logits(logits, obs["action_mask"])[0]
            idx = jnp.argmax(logits) if greedy else jrandom.categorical(key, logits)
            board = state.armies.shape[-1]
            return decode_action(idx, board, board)

        self._policy_move = jax.jit(policy_move)
        self.observations: list[dict] = []                       # what the policy saw, one entry per frame of the page
        self.hist = history_init(self.state)
        self._tick = jax.jit(tick)
        self._frame = jax.jit(frame)
        self.mountains = np.asarray(self.state.mountains[:h, :w]).astype(int).ravel().tolist()

    def _encode(self, f, actions=None) -> dict:
        f = {k: np.asarray(v) for k, v in f.items()}
        out = {
            "t": int(f["time"]), "winner": int(f["winner"]),
            "owner": f["owner"].astype(int).ravel().tolist(), "armies": f["armies"].astype(int).ravel().tolist(),
            "generals": np.flatnonzero(f["generals"]).tolist(), "castles": np.flatnonzero(f["castles"]).tolist(),
            "seen": np.flatnonzero(f["seen"]).tolist(), "land": f["land"].astype(int).tolist(),
            "army": f["army"].astype(int).tolist(), "eliminated": f["eliminated"].astype(bool).tolist(),
            "value": round(float(f["value"]), 3), "pass": round(float(f["pass_prob"]), 4),
        }
        probs = f["move_probs"].reshape(self.h * self.w, 4)
        out["policy"] = [[int(i), *np.round(probs[i], 4).tolist()] for i in np.flatnonzero(probs.sum(-1) >= 0.002)]
        if actions is not None:          # moves of this tick as [seat, from cell, to cell]
            a = np.asarray(actions)
            dirs = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
            w = self.w
            out["moves"] = [[i, int(r * w + c), int((r + dirs[d][0]) * w + c + dirs[d][1])]
                            for i, (p, r, c, d, _) in enumerate(a) if p == 0 and r < self.h and c < w]
        return out

    def _keep_observation(self, raw) -> None:
        self.observations.append({k: np.asarray(raw[k]) for k in ("planes", "players", "player_mask")})

    def observation(self, i: int) -> dict:
        """Everything the policy saw at frame i, for the page."""
        o = self.observations[max(0, min(i, len(self.observations) - 1))]
        return {"planes": {n: np.round(o["planes"][..., k], 3).ravel().tolist() for k, n in enumerate(PLANE_NAMES)},
                "players": np.round(o["players"][: self.n], 3).tolist(), "alive": o["player_mask"][: self.n].astype(bool).tolist(),
                "player_features": list(PLAYER_FEATURE_NAMES)}

    def meta(self) -> dict:
        raw = self._frame(self.state, self.hist)
        self.observations = []
        self._keep_observation(raw)
        f = self._encode(raw)
        for k in ("land", "army", "eliminated"):
            f[k] = f[k][: self.n]
        return {"h": self.h, "w": self.w, "names": self.names, "map": self.map_info, "mountains": self.mountains, "frame": f}

    def advance(self, n: int = 1) -> list[dict]:
        frames = []
        for _ in range(n):
            if int(self.state.winner) >= 0:
                break
            self.key, k = jrandom.split(self.key)
            if self.humanexe is not None:
                if self.humanexe.msg["done"]:
                    break
                new, actions = self.humanexe.step({0: self._policy_move(self.state, self.hist, k)})
                winner = self.humanexe.msg["winner"]
                if self.humanexe.msg["done"]:
                    new = new._replace(winner=jnp.asarray(winner if winner is not None else 0, jnp.int32))
                self.hist = history_push(self.hist, self.state, actions)
                self.state = new
            else:
                self.state, self.hist, actions = self._tick(self.state, self.hist, k)
            raw = self._frame(self.state, self.hist)
            self._keep_observation(raw)
            f = self._encode(raw, actions)
            for k in ("land", "army", "eliminated"):
                f[k] = f[k][: self.n]                            # drop empty padded seats
            f["moves"] = [m for m in f["moves"] if m[0] < self.n]
            frames.append(f)
        return frames

    def close(self):
        """Ends a game against Human.exe and its process (nothing to do otherwise)."""
        if self.humanexe is not None:
            self.humanexe.stop_game()
            self.humanexe.__exit__(None, None, None)
            self.humanexe = None


def bot_lineup(preset: str, n_bots: int) -> list[str]:
    """'expander' -> all Expanders; 'mix' -> expander, hunter, harvester, random, ... in turn; 'self' -> copies of the
    policy; 'past' -> an older checkpoint; 'selfpast' -> copies and older checkpoints in turn; 'field' -> a mixed table
    of copies, older checkpoints and every bot; or a comma list of these names."""
    presets = {"mix": ["expander", "hunter", "harvester", "random"], "selfpast": ["self", "past"],
               "field": ["self", "expander", "past", "harvester", "self", "hunter", "past", "random"]}
    order = presets.get(preset, preset.split(","))
    unknown = [b for b in order if b not in BOTS and b not in ("self", "past", "humanexe")]
    if unknown:
        raise ValueError(f"unknown bots {unknown}; known: {list(BOTS)}")
    return [order[i % len(order)] for i in range(n_bots)]
