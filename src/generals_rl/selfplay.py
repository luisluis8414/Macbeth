"""Self-play PPO for 8-player free-for-all, started from an imitation policy (kickstarting).

The recipe of Straka, Lisy & Schmid 2026 (superhuman 1v1, "Average Joe"), adapted to FFA:
    - PPO with GAE, gamma = 1, one pass over the top quarter of the samples by |advantage| per iteration, an EMA of
      the weights for evaluation and checkpoints, power-law schedules for the learning rate (over samples) and the
      entropy bonus, an HL-Gauss value head;
    - seats: one seat per game is the learning policy; every other seat is drawn per game: the current policy (it
      learns too), a frozen older snapshot from a pool, or an AFK player. A snapshot joins the pool when the EMA wins
      gate_win of its own gate games against the newest one (the even share is 1 / players);
    - a privileged critic: a separate, smaller network that sees the whole board without fog and which kind of
      player sits in each seat; its values give the advantages (training only);
    - reward: win_reward for a real win (the last seat alive), loss_reward for every other end (out, or alive at
      truncation), plus a convex bonus per general taken (the k-th of a game pays k * capture_total / (1 + ... + 7));
    - shaping: potential-based (land, army and cities against the other players), read only on the ticks where every
      cell grows (every 50), so it pays for what a player holds then, not for moving early; it adds up to zero over
      a game;
    - kickstarting: the forward KL to the imitation policy (bc.py), fading to zero over anchor_fade iterations;
    - maps: the simulator's terrain with real spawns, city counts and sizes (mapgen.py), at random places on the
      36 x 36 board.

An iteration: num_steps ticks in num_envs parallel games. Board planes are stored as 16-bit integers during the
rollout (every value is a multiple of 1/200 up to the limits of the game), which is what makes it fit on one GPU.
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

from generals import GeneralsEnv

from .evaluate import bot_act, head_to_head, play_match, policy_act
from .features import MAX_PLAYERS, PASS, decode_action, history_init, history_push, observe
from .mapgen import REAL_SIDES, realistic_pool
from .model import BoardTransformer, masked_logits, value_loss

PLANE_SCALE = 200.0       # stored planes: int16 / 200
MAX_BOARD = 36            # the board every map is padded to, shared with bc.py and the viewer
KINDS = ("self", "past", "afk")   # who sits in a seat; the critic sees it as a one-hot per player slot


@dataclass
class Config:
    out: str = "runs/selfplay"
    init: str = ""                       # params_*.msgpack to start from, usually the bc.py clone
    anchor: str = ""                     # params_*.msgpack of the imitation policy the policy is pulled towards
    anchor_coef: float = 1.0             # weight of the forward KL to the anchor (the cross-entropy to its moves)
    anchor_fade: int = 600               # the anchor's weight falls linearly to 0 over this many iterations
    # seats
    players: int = 8
    past_share: float = 0.5              # each non-learning seat: a frozen snapshot with this probability,
    afk_share: float = 0.0               # an AFK player (never moves, like one who left) with this one, else self
    past_keep: int = 50                  # snapshots kept in the pool
    past_newest: float = 0.5             # share of iterations that play the newest snapshot; the rest a random older one
    past_hold: int = 8                   # draw a new snapshot every this many iterations
    gate_every: int = 100                # gate games of the EMA against the pool every this many iterations
    gate_games: int = 256                # this many against the newest snapshot (and as many against older ones)
    gate_chunk: int = 128
    gate_window: int = 1000              # a window of this many gate games against the newest is judged at once
    gate_win: float = 0.15               # promote the EMA once it wins this share of them (even share 1 / players)
    gate_old_min: float = 0.14           # ... and at least this share against the older snapshots
    # games
    pad: int = MAX_BOARD
    truncation: int = 1500
    pool_size: int = 2048                # maps per pool, a new pool every pool_refresh_every iterations
    pool_refresh_every: int = 200
    # model
    dim: int = 448
    depth: int = 7
    heads: int = 8
    ff_factor: int = 3
    patch: int = 3
    stem: int = 2
    stem_dim: int = 64
    value_bins: int = 128
    hl_sigma: float = 0.04               # width of the HL-Gauss target (Straka et al. 2026)
    critic_dim: int = 256                # the privileged critic: the same architecture, smaller
    critic_depth: int = 4
    critic_lr: float = 1e-4
    critic_frac: float = 0.5             # share of the active samples the critic trains on per iteration
    critic_minibatch: int = 1024
    # rollout and PPO
    num_envs: int = 64
    num_steps: int = 96
    total_iters: int = 39_000
    minibatch_size: int = 1024
    micro_batch: int = 256               # compute each minibatch in chunks of this size (same step, less memory)
    adv_top_frac: float = 0.25
    gae_lambda: float = 0.97
    clip_eps: float = 0.2
    max_grad_norm: float = 0.267
    vf_coef: float = 0.5
    ema_decay: float = 0.999
    # reward
    win_reward: float = 0.5
    loss_reward: float = -0.15
    capture_total: float = 0.45
    shaping: float = 0.5                 # weight of the potential-based shaping; 0 = off
    shaping_every: int = 50              # read the potential every this many ticks (50: the land bonus)
    shaping_max_ratio: float = 100.0     # a land / army / city ratio of this size counts as a full advantage
    shaping_weights: str = "0.3,0.3,0.4"  # land, army, cities
    # schedules: lr = numerator * (samples / sample_scale + 1)^-exponent, clipped to [lr_min, lr_max]
    lr_max: float = 3e-5
    lr_min: float = 5e-6
    lr_warmup: int = 50                  # iterations of linear warmup at the start of a run
    lr_numerator: float = 0.5
    lr_exponent: float = 1.1
    lr_sample_scale: float = 524_288.0
    ent_start: float = 0.01              # entropy bonus start * (1 + iteration)^-power, at least ent_min
    ent_min: float = 0.001
    ent_power: float = 0.2
    # bookkeeping
    log_every: int = 1
    eval_every: int = 0                  # > 0: games against eval_bots every this many iterations (slow: Harvester)
    eval_games: int = 32
    eval_bots: str = "expander,harvester"
    save_every: int = 25
    time_limit_hours: float = 168.0
    seed: int = 0


def anchor_at(cfg: Config, t: int) -> float:
    """Weight of the anchor at iteration t: anchor_coef falling linearly to 0 over anchor_fade iterations."""
    if not cfg.anchor:
        return 0.0
    return cfg.anchor_coef * max(0.0, 1.0 - t / cfg.anchor_fade) if cfg.anchor_fade > 0 else cfg.anchor_coef


def lr_at(cfg: Config, t: int, samples: int) -> float:
    """Learning rate at iteration t after `samples` samples went through the optimizer (Straka et al. 2026: their
    s / 512 optimizer steps of 1024 samples), so it does not depend on the batch size."""
    lr = float(np.clip(cfg.lr_numerator * (samples / cfg.lr_sample_scale + 1) ** -cfg.lr_exponent, cfg.lr_min,
                       cfg.lr_max))
    return lr * min(1.0, t / cfg.lr_warmup) if cfg.lr_warmup > 0 else lr


def ent_at(cfg: Config, t: int) -> float:
    return max(cfg.ent_min, cfg.ent_start * (1 + t) ** -cfg.ent_power)


def winner_reward(win, finished, kills, captures, n, win_reward, loss_reward, capture_total):
    """Reward of one tick per seat: win_reward for a real win, loss_reward for every other end, plus the convex capture
    bonus. win, finished: (N,) the seat wins / leaves the game on this tick; kills: (N,) generals taken on this tick;
    captures: (N,) taken before it in this game."""
    step = capture_total / (n * (n - 1) / 2)
    bonus = step * (kills * captures + kills * (kills + 1) / 2)            # captures + 1, ..., captures + kills
    return jnp.where(finished, jnp.where(win, win_reward, loss_reward), 0.0) + bonus


def fresh_outcome(cfg):
    """Per game and seat, reset when a game ends: generals taken so far, and the potential at the last bonus tick
    (periodic_shaping; 0 at the start, where every seat is equal)."""
    zeros = lambda: jnp.zeros((cfg.num_envs, cfg.players))
    return dict(captures=zeros(), phi=zeros())


def shaping_weights(cfg):
    w = tuple(float(x) for x in cfg.shaping_weights.split(","))
    assert len(w) == 3, "--shaping-weights takes land,army,cities"
    return w


def potential(state, max_ratio, weights=(0.3, 0.3, 0.4)):
    """Material advantage per seat in [-1, 1]: by default 0.3 land + 0.3 army + 0.4 cities (Straka & Schmid 2025),
    each the log ratio to the mean of the other players still in the game. Seats that are out have 0."""
    land = state.ownership.sum((1, 2)).astype(jnp.float32)
    army = (state.armies[None] * state.ownership).sum((1, 2)).astype(jnp.float32)
    cities = (state.castles[None] & state.ownership).sum((1, 2)).astype(jnp.float32)
    alive = (~state.eliminated).astype(jnp.float32)

    def part(x, floor):
        others = ((x * alive).sum() - x * alive) / jnp.maximum(alive.sum() - alive, 1.0)
        return jnp.clip(jnp.log(jnp.maximum(x + floor, 1.0) / jnp.maximum(others + floor, 1.0)) / jnp.log(max_ratio),
                        -1.0, 1.0)

    wl, wa, wc = weights
    return (wl * part(land, 0.0) + wa * part(army, 0.0) + wc * part(cities, 1.0)) * alive


def periodic_shaping(raw, phi_mark, phi_after, done, bonus, weight):
    """Shaping that reads the potential only on bonus ticks: raw + weight * (phi(now) - phi(last bonus tick)) on a bonus
    tick, raw - weight * phi(last bonus tick) when the seat is done (out, or the game ended), raw otherwise. The terms
    add up to phi(end) - phi(start) = 0 over a game, but only the standing at the bonus ticks counts, not when the land
    was taken. Divided by 1 + weight, so returns stay in [-1, 1] where the value head can reach them.
    Returns (reward, phi_mark for the next tick)."""
    after = jnp.where(done, 0.0, phi_after)
    read = done | bonus
    reward = raw + weight * jnp.where(read, after - phi_mark, 0.0)
    return reward / (1.0 + weight), jnp.where(read, after, phi_mark)


def gae(rewards, values, dones, last_values, lam, gamma=1.0):
    """Advantages and returns (T, B). dones[t] ends the sequence of that seat after tick t."""
    def body(next_adv, x):
        r, v, next_v, d = x
        delta = r + gamma * (1.0 - d) * next_v - v
        adv = delta + gamma * lam * (1.0 - d) * next_adv
        return adv, adv

    next_values = jnp.concatenate([values[1:], last_values[None]])
    _, adv = jax.lax.scan(body, jnp.zeros_like(last_values), (rewards, values, next_values, dones.astype(jnp.float32)),
                          reverse=True)
    return adv, adv + values


def pack_planes(planes):
    return jnp.clip(jnp.round(planes * PLANE_SCALE), -32768, 32767).astype(jnp.int16)


def unpack_planes(planes):
    return planes.astype(jnp.float32) / PLANE_SCALE


def make_env(cfg: Config, truncation: int, pool_size: int) -> GeneralsEnv:
    """The simulator on every real map size. Its own spawns are replaced by mapgen.realistic_pool; the simulator makes
    the same number of maps per (height, width) and drops the remainder, so the pool size is rounded down to that."""
    lo, hi = min(REAL_SIDES), max(REAL_SIDES)
    combos = (hi - lo + 1) ** 2
    return GeneralsEnv(min_grid_size=lo, max_grid_size=hi, pool_size=pool_size // combos * combos, pad_to=cfg.pad,
                       truncation=truncation, num_players=cfg.players, min_generals_distance=10,
                       num_castles_range=(18, 28))


def place_maps(pool, key):
    """Move every map of the pool to a random place inside its padding (the simulator puts it top left), so all board
    positions get trained, not only those of the largest maps. Wrapping only moves padding mountains around."""
    pad = pool.mountains.shape[-1]

    def one(st, k):
        open_ = ~st.mountains
        h = pad - jnp.argmax(open_.any(1)[::-1])            # rows and columns the map uses, from the top left
        w = pad - jnp.argmax(open_.any(0)[::-1])
        kr, kc = jrandom.split(k)
        dr, dc = jrandom.randint(kr, (), 0, pad - h + 1), jrandom.randint(kc, (), 0, pad - w + 1)
        roll = lambda x: jnp.roll(x, (dr, dc), axis=(-2, -1)) if x.ndim >= 2 and x.shape[-2:] == (pad, pad) else x
        st = jax.tree.map(roll, st)
        return st._replace(general_positions=st.general_positions + jnp.stack([dr, dc]).astype(st.general_positions.dtype))
    return jax.vmap(one)(pool, jrandom.split(key, pool.mountains.shape[0]))


def new_pool(env: GeneralsEnv, cfg: Config, key):
    """A pool of real-looking maps (mapgen.py) at random places on the board."""
    k1, k2, k3, k_place = jrandom.split(key, 4)
    seed = int(jrandom.randint(k3, (), 0, 2**31 - 1))
    pool = realistic_pool([env.reset(k1)[0], env.reset(k2)[0]], env.pool_size, cfg.players, np.random.default_rng(seed))
    return place_maps(pool, k_place)


def captured_generals(ownership, general_positions, out_now):
    """(N,) generals each seat took on this tick: the cell of a general that just fell belongs to whoever took it.
    ownership: (N, H, W) after the tick; general_positions: (N, 2) where each general started; out_now: (N,)."""
    holder = ownership[:, general_positions[:, 0], general_positions[:, 1]]       # (owner, fallen seat)
    return (holder & out_now[None, :]).sum(1).astype(jnp.float32)


def draw_kinds(key, pool_idx, envs, n_seat, probs, learner):
    """Who plays each (env, seat), as an index into KINDS. A game keeps its pool_idx and a new game gets a new one, so
    the draw is random between games and fixed within a game, without extra state. The learner seat is always the
    current policy, so every game gives training samples."""
    logp = jnp.log(jnp.asarray(probs))

    def one(idx, env):
        return jrandom.categorical(jrandom.fold_in(jrandom.fold_in(key, idx), env), logp, shape=(n_seat,))
    return jax.vmap(one)(pool_idx, envs).at[envs, learner].set(0)


def critic_model(cfg: Config):
    """The privileged critic: the actor's architecture at critic_dim x critic_depth."""
    return BoardTransformer.from_config(dict(asdict(cfg), dim=cfg.critic_dim, depth=cfg.critic_depth), MAX_BOARD)


def make_rollout(apply_fn, critic_apply, env: GeneralsEnv, cfg: Config):
    """rollout(params, past_params, critic_params, st, hist, outcome, pool, key): num_steps ticks of num_envs games.
    Every seat is a sample row; rows of the learning policy are active, the rest masked out."""
    n_env, n_seat = cfg.num_envs, cfg.players
    seats, envs = jnp.arange(n_seat), jnp.arange(n_env)
    learner = envs % n_seat                                 # the seat that is always the current policy
    probs = np.array([1.0 - cfg.past_share - cfg.afk_share, cfg.past_share, cfg.afk_share])
    draw_key = jrandom.PRNGKey(cfg.seed)
    phi = jax.vmap(lambda s: potential(s, cfg.shaping_max_ratio, shaping_weights(cfg)))

    flat = lambda x: x.reshape((n_env * n_seat,) + x.shape[2:])           # (num_envs, seats, ...) -> rows
    net_in = lambda obs: {**obs, "planes": unpack_planes(obs["planes"])}
    kinds_of = lambda st: draw_kinds(draw_key, st.pool_idx, envs, n_seat, probs, learner)

    def observe_all(st, hist, full=False):
        obs = jax.vmap(lambda s, h: jax.vmap(lambda me: observe(s, me, h, full))(seats))(st, hist)
        obs = jax.tree.map(flat, obs)
        obs["planes"] = pack_planes(obs["planes"])
        return obs

    def critic_obs(st, hist, kind):
        """The critic's input: the full view of every row, plus per player slot which kind of player sits there."""
        obs = observe_all(st, hist, full=True)
        me = jnp.broadcast_to(seats, (n_env, n_seat)).reshape(-1, 1)
        slot = jnp.arange(MAX_PLAYERS)[None]
        k = jnp.take_along_axis(jnp.repeat(kind, n_seat, axis=0), (me + slot) % n_seat, axis=1)  # (rows, MAX_PLAYERS)
        onehot = jax.nn.one_hot(k, len(KINDS)) * (slot < n_seat)[..., None]
        obs["players"] = jnp.concatenate([obs["players"], onehot], -1)
        return obs

    @jax.jit
    def rollout(params, past_params, critic_params, st, hist, outcome, pool, key):
        def body(carry, _):
            st, hist, outcome, key = carry
            key, k_act, k_past = jrandom.split(key, 3)
            obs = observe_all(st, hist)
            logits = masked_logits(apply_fn(params, net_in(obs))[0], obs["action_mask"])
            action = jrandom.categorical(k_act, logits)
            logp = jnp.take_along_axis(jax.nn.log_softmax(logits), action[:, None], 1)[:, 0]
            moves = decode_action(action, cfg.pad, cfg.pad).reshape(n_env, n_seat, 5)
            kind = kinds_of(st)
            past_moves = moves                              # never drawn without a past share, so not computed
            if cfg.past_share > 0:
                past_logits = masked_logits(apply_fn(past_params, net_in(obs))[0], obs["action_mask"])
                past_moves = decode_action(jrandom.categorical(k_past, past_logits), cfg.pad, cfg.pad
                                           ).reshape(n_env, n_seat, 5)
            afk = jnp.broadcast_to(jnp.array(PASS, jnp.int32), (n_env, n_seat, 5))
            joint = jnp.stack([moves, past_moves, afk])[kind, envs[:, None], seats]
            learn = flat(kind == 0)

            pushed = jax.vmap(history_push)(hist, st, joint)
            ts, st2 = jax.vmap(lambda s, a: env.step(s, a, pool))(st, joint)
            before, after = st.eliminated, ts.last_state.eliminated
            ended = ts.terminated | ts.truncated
            alive_at_end = ended[:, None] & ~after
            win = alive_at_end & ((~after).sum(1) == 1)[:, None]             # the last seat alive
            finished = (after & ~before) | alive_at_end
            done = (after & ~before) | ended[:, None]
            kills = jax.vmap(captured_generals)(ts.last_state.ownership, st.general_positions, after & ~before)
            reward = jax.vmap(lambda w, f, k, c: winner_reward(w, f, k, c, n_seat, cfg.win_reward, cfg.loss_reward,
                                                               cfg.capture_total))(win, finished, kills, outcome["captures"])
            phi_mark = outcome["phi"]
            if cfg.shaping > 0:
                bonus = (ts.last_state.time % cfg.shaping_every == 0)[:, None]
                reward, phi_mark = periodic_shaping(reward, phi_mark, phi(ts.last_state), done, bonus, cfg.shaping)
            reward = reward * ~before

            fresh = jax.vmap(history_init)(st2)
            hist2 = jax.tree.map(lambda a, b: jnp.where(ended.reshape((-1,) + (1,) * (a.ndim - 1)), a, b), fresh, pushed)
            active = flat(~before) & learn
            cobs = critic_obs(st, hist, kind)
            _, value = critic_apply(critic_params, net_in(cobs))
            sample = dict(obs=obs, cobs=cobs, action=action, logp=logp, value=value * active, reward=flat(reward),
                          done=flat(done), active=active)
            stats = dict(ended=ended.sum(), length=(ts.last_state.time * ended).sum(),
                         passes=((action == logits.shape[-1] - 1) & active).sum(), wins=(flat(win) & learn).sum(),
                         kills=kills.sum(), learner_kills=(flat(kills) * learn).sum(), alive_at_end=alive_at_end.sum(),
                         learner_seats_ended=(flat(jnp.broadcast_to(ended[:, None], (n_env, n_seat))) & learn).sum(),
                         seats_by_kind=jnp.stack([(kind == k).sum() for k in range(len(KINDS))]))
            outcome = jax.tree.map(lambda x: jnp.where(ended[:, None], 0.0, x),
                                   dict(captures=outcome["captures"] + kills, phi=phi_mark))
            return (st2, hist2, outcome, key), (sample, stats)

        (st, hist, outcome, key), (traj, stats) = jax.lax.scan(body, (st, hist, outcome, key), None, length=cfg.num_steps)
        _, last_value = critic_apply(critic_params, net_in(critic_obs(st, hist, kinds_of(st))))
        last_value = last_value * flat(~st.eliminated)
        adv, ret = gae(traj["reward"], traj["value"], traj["done"], last_value, cfg.gae_lambda)
        return st, hist, outcome, traj, adv, ret, jax.tree.map(lambda x: x.sum(0), stats), key

    return rollout


def accumulate(grad_fn, params, b, micro: int):
    """grad_fn(params, b) on the whole minibatch, or (micro > 0) on chunks of `micro` samples with gradients and
    metrics averaged, which gives the same step for mean losses with a fraction of the memory."""
    n = jax.tree.leaves(b)[0].shape[0]
    if not micro or micro >= n:
        return grad_fn(params, b)
    chunks = jax.tree.map(lambda x: x.reshape((n // micro, micro) + x.shape[1:]), b)

    def body(acc, chunk):
        return jax.tree.map(jnp.add, acc, grad_fn(params, chunk)), None
    zero = jax.tree.map(jnp.zeros_like, jax.eval_shape(grad_fn, params, jax.tree.map(lambda x: x[0], chunks)))
    total, _ = jax.lax.scan(body, zero, chunks)
    return jax.tree.map(lambda x: x / (n // micro), total)


def make_update(apply_fn, tx, cfg: Config, anchor=None):
    """One PPO step on a minibatch. anchor: weights of the imitation policy the policy is pulled towards with
    anchor_coef * KL(anchor || policy), the cross-entropy to its moves (kickstarting, Schmitt et al. 2018)."""
    def loss_fn(params, b, ent_coef, anchor, anchor_coef):
        obs = {**b["obs"], "planes": unpack_planes(b["obs"]["planes"])}
        logits, _, vlogits = apply_fn(params, obs, value_logits=True)
        lp_all = jax.nn.log_softmax(masked_logits(logits, obs["action_mask"]))
        logp = jnp.take_along_axis(lp_all, b["action"][:, None], 1)[:, 0]
        ratio = jnp.exp(logp - b["logp"])
        pg = -jnp.minimum(ratio * b["adv"], jnp.clip(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * b["adv"]).mean()
        entropy = -jnp.where(obs["action_mask"], jnp.exp(lp_all) * lp_all, 0.0).sum(-1).mean()
        vloss = value_loss(vlogits, b["ret"], cfg.hl_sigma)          # the policy's own value head, an auxiliary loss
        loss = pg + cfg.vf_coef * vloss - ent_coef * entropy
        m = dict(pg=pg, vloss=vloss, entropy=entropy, kl=(b["logp"] - logp).mean(),
                 clipfrac=(jnp.abs(ratio - 1) > cfg.clip_eps).mean())
        if anchor is not None:
            anchor_logits, _ = apply_fn(anchor, obs)
            lp_anchor = jax.nn.log_softmax(masked_logits(jax.lax.stop_gradient(anchor_logits), obs["action_mask"]))
            kl_anchor = jnp.where(obs["action_mask"], jnp.exp(lp_anchor) * (lp_anchor - lp_all), 0.0).sum(-1).mean()
            loss = loss + anchor_coef * kl_anchor
            m["kl_anchor"] = kl_anchor
        return loss, m

    @jax.jit
    def step_with(params, opt_state, b, ent_coef, anchor, anchor_coef):
        grad_fn = jax.value_and_grad(lambda p, x: loss_fn(p, x, ent_coef, anchor, anchor_coef), has_aux=True)
        (_, m), grads = accumulate(grad_fn, params, b, cfg.micro_batch)
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, m

    def step(params, opt_state, b, ent_coef, anchor_coef=0.0):
        return step_with(params, opt_state, b, ent_coef, anchor, jnp.float32(anchor_coef))   # the anchor is an argument
    return step


def make_critic_update(critic_apply, tx, sigma: float, micro: int = 0):
    """One step of the privileged critic on a minibatch: HL-Gauss cross-entropy to the returns."""
    def loss_fn(params, b):
        _, _, vlogits = critic_apply(params, {**b["cobs"], "planes": unpack_planes(b["cobs"]["planes"])},
                                     value_logits=True)
        return value_loss(vlogits, b["ret"], sigma)

    @jax.jit
    def step(params, opt_state, b):
        loss, grads = accumulate(jax.value_and_grad(loss_fn), params, b, micro)
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss
    return step


def select_top(adv, active, frac):
    """Indices (host side) of the top `frac` of the active samples by |advantage|, plus the normalized advantages."""
    adv, active = np.asarray(adv), np.asarray(active)
    live = adv[active]
    norm = (adv - live.mean()) / (live.std() + 1e-8)
    live_idx = np.flatnonzero(active)
    k = max(1, int(len(live_idx) * frac))
    keep = live_idx[np.argpartition(-np.abs(norm[live_idx]), k - 1)[:k]]
    return keep, norm


def explained_variance(value, ret, active):
    """1 - Var(ret - value) / Var(ret) over the active samples: 1 = the values predict the returns, 0 = no better
    than their mean, below 0 = worse. The returns are bootstrapped from the critic, so this flatters it."""
    n = jnp.maximum(active.sum(), 1)
    var = lambda x: (jnp.where(active, x, 0.0) ** 2).sum() / n - (jnp.where(active, x, 0.0).sum() / n) ** 2
    return 1.0 - var(ret - value) / jnp.maximum(var(ret), 1e-8)


def start_games(pool, cfg: Config, key):
    """num_envs games from the pool; different pool_idx values so finished games continue with different maps."""
    idx = jrandom.choice(key, pool.armies.shape[0], (cfg.num_envs,), replace=False)
    st = jax.tree.map(lambda x: x[idx], pool)
    st = st._replace(pool_idx=idx.astype(st.pool_idx.dtype))
    return st, jax.vmap(history_init)(st)


def train(cfg: Config):
    if cfg.pad % cfg.patch or max(REAL_SIDES) > cfg.pad:
        raise SystemExit(f"--pad {cfg.pad} must be divisible by --patch {cfg.patch} and hold {max(REAL_SIDES)} x "
                         f"{max(REAL_SIDES)} maps")
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=1))
    log_file = open(out / "metrics.jsonl", "a")
    t_start = time.time()

    def log(it, **m):
        m = {k: float(v) for k, v in m.items()}
        log_file.write(json.dumps({"iter": it, "time_h": (time.time() - t_start) / 3600, **m}) + "\n")
        log_file.flush()
        print(f"iter {it:>6}  " + "  ".join(f"{k} {v:.4g}" for k, v in m.items()), flush=True)

    print("devices:", jax.devices())
    key = jrandom.PRNGKey(cfg.seed)
    env = make_env(cfg, cfg.truncation, cfg.pool_size)
    model, critic = BoardTransformer.from_config(cfg, MAX_BOARD), critic_model(cfg)
    key, k_init = jrandom.split(key)
    dummy = jax.tree.map(lambda x: x[None], observe(env.init_state(k_init), 0))
    params = model.init(k_init, dummy)
    print("parameters:", sum(x.size for x in jax.tree.leaves(params)))
    if cfg.init:
        params = serialization.from_bytes(params, Path(cfg.init).read_bytes())
        print("weights from", cfg.init)
    tx = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.inject_hyperparams(optax.adam)(learning_rate=cfg.lr_max))
    opt_state = tx.init(params)
    ema = params
    cdummy = dict(dummy, players=jnp.concatenate([dummy["players"], jnp.zeros(dummy["players"].shape[:2] + (len(KINDS),))], -1))
    critic_params = critic.init(jrandom.PRNGKey(cfg.seed + 1), cdummy)
    ctx = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(cfg.critic_lr))
    critic_opt = ctx.init(critic_params)
    print("critic parameters:", sum(x.size for x in jax.tree.leaves(critic_params)))
    it0, samples_seen = 0, 0

    resume = out / "resume.msgpack"
    if resume.exists():                                    # "stage" is kept in the file for older runs, unused
        saved = serialization.from_state_dict(dict(params=params, ema=ema, opt_state=opt_state, it=0, stage=0,
                                                   samples=0, critic=critic_params, critic_opt=critic_opt),
                                              serialization.msgpack_restore(resume.read_bytes()))
        params, ema, opt_state, critic_params, critic_opt = (saved[k] for k in ("params", "ema", "opt_state", "critic",
                                                                                "critic_opt"))
        it0, samples_seen = int(saved["it"]), int(saved["samples"])
        print(f"resuming at iteration {it0}")

    # Frozen snapshots for the "past" seats, saved as checkpoints and listed in past.json so they survive a restart;
    # before the first one, the weights the run started from.
    past_file = out / "past.json"
    past_iters = json.loads(past_file.read_text()) if past_file.exists() else []
    # snapshots wait in host memory and go to the GPU only when drawn: many of them fragment GPU memory
    past = [jax.device_get(serialization.from_bytes(ema, (out / f"params_{i}.msgpack").read_bytes()))
            for i in past_iters[-cfg.past_keep:]] or [jax.device_get(ema)]
    gate_wins, gate_games, old_wins, old_games = 0.0, 0, 0.0, 0
    pick = 0

    def save(it):
        (out / f"params_{it}.msgpack").write_bytes(serialization.to_bytes(ema))
        (out / "params_latest.msgpack").write_bytes(serialization.to_bytes(ema))
        tmp = out / "resume.tmp"
        tmp.write_bytes(serialization.to_bytes(dict(params=params, ema=ema, opt_state=opt_state, it=it, stage=0,
                                                    samples=samples_seen, critic=critic_params, critic_opt=critic_opt)))
        tmp.replace(resume)

    rollout = make_rollout(model.apply, critic.apply, env, cfg)
    critic_update = make_critic_update(critic.apply, ctx, cfg.hl_sigma, cfg.micro_batch)
    gate_fns = [jax.jit(lambda prm, opp, pl, k, seat=seat: head_to_head(model.apply, prm, opp, seat, env, pl, k,
                                                                         cfg.gate_chunk))
                for seat in range(cfg.players)]

    def eval_fn(bot):                                      # the EMA in seat 0, the bot everywhere else, on pool maps
        def run(prm, pl, k):
            k_map, k_play = jrandom.split(k)
            idx = jrandom.randint(k_map, (cfg.eval_games,), 0, pl.armies.shape[0])
            starts = jax.tree.map(lambda x: x[idx], pl)._replace(pool_idx=idx.astype(pl.pool_idx.dtype))
            acts = [policy_act(model.apply, prm)] + [bot_act(bot)] * (cfg.players - 1)
            wins, place = play_match(acts, env, pl, k_play, cfg.eval_games, starts)
            return wins[0], place[0]
        return jax.jit(run)
    evals = {bot: eval_fn(bot) for bot in cfg.eval_bots.split(",") if bot} if cfg.eval_every > 0 else {}

    anchor = None
    if cfg.anchor:
        anchor = serialization.from_bytes(params, Path(cfg.anchor).read_bytes())
        print(f"anchor {cfg.anchor}, weight {cfg.anchor_coef} fading over {cfg.anchor_fade} iterations")
    update = make_update(model.apply, tx, cfg, anchor)
    key, k_pool, k_start = jrandom.split(key, 3)
    pool = new_pool(env, cfg, k_pool)
    st, hist = start_games(pool, cfg, k_start)
    outcome = fresh_outcome(cfg)
    rng = np.random.default_rng(cfg.seed + it0)
    frames = 0

    for it in range(it0 + 1, cfg.total_iters + 1):
        t0 = time.time()
        if (it - it0 - 1) % cfg.past_hold == 0 or pick >= len(past):
            if len(past) > 1:
                pick = len(past) - 1 if rng.random() < cfg.past_newest else int(rng.integers(len(past) - 1))
            else:
                pick = 0
        st, hist, outcome, traj, adv, ret, stats, key = rollout(params, past[pick], critic_params, st, hist, outcome,
                                                                pool, key)
        jax.block_until_ready(adv)
        t1 = time.time()

        flat2 = lambda x: x.reshape((-1,) + x.shape[2:])
        samples = jax.tree.map(flat2, dict(obs=traj["obs"], action=traj["action"], logp=traj["logp"]))
        ret_f, adv_f, active = flat2(ret), flat2(adv), flat2(traj["active"])
        expl_var = explained_variance(flat2(traj["value"]), ret_f, active)
        keep, norm = select_top(adv_f, active, cfg.adv_top_frac)
        rng.shuffle(keep)
        samples = dict(samples, ret=ret_f, adv=jnp.asarray(norm))
        ent_coef = ent_at(cfg, it)
        opt_state[1].hyperparams["learning_rate"] = jnp.asarray(lr_at(cfg, it, samples_seen), jnp.float32)
        ms = []
        for s in range(0, len(keep) - cfg.minibatch_size + 1, cfg.minibatch_size):
            idx = jnp.asarray(keep[s: s + cfg.minibatch_size])
            b = jax.tree.map(lambda x, i=idx: x[i], samples)
            params, opt_state, m = update(params, opt_state, b, ent_coef, anchor_at(cfg, it))
            ms.append(m)
            samples_seen += cfg.minibatch_size
        critic_losses = []                                 # the critic learns from a random share of the active samples
        live = np.flatnonzero(np.asarray(active))
        pick_c = rng.choice(live, size=int(len(live) * cfg.critic_frac), replace=False)
        cobs = jax.tree.map(flat2, traj["cobs"])
        for s in range(0, len(pick_c) - cfg.critic_minibatch + 1, cfg.critic_minibatch):
            idx = jnp.asarray(pick_c[s: s + cfg.critic_minibatch])
            b = dict(cobs=jax.tree.map(lambda x, i=idx: x[i], cobs), ret=ret_f[idx])
            critic_params, critic_opt, closs = critic_update(critic_params, critic_opt, b)
            critic_losses.append(float(closs))
        del cobs, samples, traj
        decay = min(cfg.ema_decay, (1 + it) / (10 + it))
        ema = jax.tree.map(lambda e, p: decay * e + (1 - decay) * p, ema, params)
        t2 = time.time()

        frames += cfg.num_envs * cfg.num_steps
        if it % cfg.log_every == 0:
            mean = jax.tree.map(lambda *x: float(np.mean(x)), *ms) if ms else {}
            n_active = int(np.asarray(active).sum())
            ended = int(stats["ended"])
            seats = np.asarray(stats["seats_by_kind"], float)
            log(it, learner_win=int(stats["wins"]) / max(int(stats["learner_seats_ended"]), 1),
                learner_kill_share=float(stats["learner_kills"]) / max(int(stats["kills"]), 1),
                **{f"seats_{k}": s / seats.sum() for k, s in zip(KINDS, seats)}, past_versions=len(past_iters),
                rollout_s=t1 - t0, update_s=t2 - t1, fps=cfg.num_envs * cfg.num_steps / (t2 - t0), frames_total=frames,
                active=n_active / active.size, opt_steps=len(ms), lr=lr_at(cfg, it, samples_seen),
                anchor_coef=anchor_at(cfg, it), samples_seen=samples_seen, ent_coef=ent_coef, games_ended=ended,
                game_len=int(stats["length"]) / max(ended, 1), kills=int(stats["kills"]),
                alive_at_end=int(stats["alive_at_end"]) / max(ended, 1),
                pass_rate=int(stats["passes"]) / max(n_active, 1), explained_var=float(expl_var), **mean,
                **({"critic_loss": float(np.mean(critic_losses))} if critic_losses else {}))
        if evals and it % cfg.eval_every == 0:
            t_e = time.time()
            for bot, fn in evals.items():
                win, place = fn(ema, pool, jrandom.PRNGKey(it))
                log(it, **{f"win_vs_{bot}": win, f"place_vs_{bot}": place, "eval_s": time.time() - t_e})
        if it % cfg.pool_refresh_every == 0:
            key, k_pool = jrandom.split(key)
            pool = new_pool(env, cfg, k_pool)

        # Gate: games of the EMA in one seat against the newest snapshot in all others (and against a random older
        # one), the seat rotating. A window of gate_window games is judged at once; the EMA joins the pool if it won
        # gate_win of them and gate_old_min against the older snapshots (like AlphaGo Zero's 55 % gate).
        if cfg.gate_games and it % cfg.gate_every == 0:
            t_g = time.time()
            for c in range(cfg.gate_games // cfg.gate_chunk):
                key, k_new, k_old = jrandom.split(key, 3)
                seat = (it // cfg.gate_every * (cfg.gate_games // cfg.gate_chunk) + c) % cfg.players
                gate_wins += float(gate_fns[seat](ema, past[-1], pool, k_new)[2])
                gate_games += cfg.gate_chunk
                if len(past) > 1:
                    old_wins += float(gate_fns[seat](ema, past[int(rng.integers(len(past) - 1))], pool, k_old)[2])
                    old_games += cfg.gate_chunk
            log(it, gate_vs_newest=gate_wins / max(gate_games, 1), gate_games=gate_games,
                gate_vs_old=old_wins / max(old_games, 1), gate_old_games=old_games, gate_s=time.time() - t_g)
        promote = False
        if gate_games >= cfg.gate_window:
            old_ok = old_games == 0 or old_wins / old_games >= cfg.gate_old_min
            promote = gate_wins / gate_games >= cfg.gate_win and old_ok
            print(f"gate (EMA): won {gate_wins / gate_games:.1%} of {gate_games} vs the newest snapshot, "
                  + (f"{old_wins / old_games:.1%} of {old_games} vs older" if old_games else "no older snapshots")
                  + (", joins the pool" if promote else ""), flush=True)
            gate_wins, gate_games, old_wins, old_games = 0.0, 0, 0.0, 0
        if promote:
            past = (past + [jax.device_get(ema)])[-cfg.past_keep:]
            past_iters.append(it)
            save(it)                                       # also on disk, so a restart finds the same snapshots
            past_file.write_text(json.dumps(past_iters))
        if it % cfg.save_every == 0:
            save(it)
        if time.time() - t_start > cfg.time_limit_hours * 3600:
            print("time limit reached")
            save(it)
            break
    print("done, checkpoints in", out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name, default in asdict(Config()).items():
        p.add_argument("--" + name.replace("_", "-"), type=type(default), default=default)
    train(Config(**vars(p.parse_args())))


if __name__ == "__main__":
    main()
