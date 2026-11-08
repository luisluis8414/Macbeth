"""Evaluate agents in many parallel free-for-all games.

An "act function" act(states, seat, key, hist) maps a batch of GameStates (N, ...), the seat it plays, a key and
the batched game history (features.history_*) to actions (N, 5). Scripted bots see the simulator's own observation
and ignore the history; our policy sees features.observe with the history.
"""
import jax
import jax.numpy as jnp
import jax.random as jrandom

from generals import GeneralsEnv
from generals.agents import ExpanderAgent, HunterAgent, RandomAgent
from generals.agents.harvester_agent import HarvesterAgent  # not re-exported by generals.agents
from generals.core import game

from .features import decode_action, history_init, history_push, observe
from .model import masked_logits

PAD = 36

# Scripted opponents from generals-bots, roughly weakest first.
BOTS = {
    "random": RandomAgent,
    "expander": ExpanderAgent,
    "hunter": HunterAgent,
    "harvester": HarvesterAgent,
}


def make_eval_env(size: int = 27, num_players: int = 8, truncation: int = 1500, pad_to: int = PAD) -> GeneralsEnv:
    """size x size boards padded with mountains to pad_to x pad_to (36, like the replay data). Defaults match real
    8-player FFA games: median map 27 x 27 with ~24 cities."""
    return GeneralsEnv(grid_dims=(size, size), pad_to=pad_to, truncation=truncation, num_players=num_players,
                       min_generals_distance=5, num_castles_range=(18, 28), pool_size=512)


def bot_act(name: str):
    agent = BOTS[name]()

    def act(states, seat, key, hist=None):
        obs = jax.vmap(lambda s: game.get_observation(s, seat))(states)
        return jax.vmap(agent.act)(obs, jrandom.split(key, states.armies.shape[0]))
    return act


def policy_act(apply_fn, params, greedy: bool = False):
    def act(states, seat, key, hist):
        obs = jax.vmap(lambda s, h: observe(s, seat, h))(states, hist)
        logits, _ = apply_fn(params, obs)
        logits = masked_logits(logits, obs["action_mask"])
        idx = jnp.argmax(logits, -1) if greedy else jrandom.categorical(key, logits)
        h, w = states.armies.shape[-2:]
        return decode_action(idx, w, h)
    return act


def play_games(acts, env: GeneralsEnv, pool, key, n_games: int = 64, starts=None):
    """acts[i] plays seat i (len(acts) == env.num_players); the first game of each environment counts.

    starts: optional batched GameStates to begin from, for example maps drawn from the pool. Otherwise each game
    starts from env.init_state, which is always max_grid_size x max_grid_size.

    Returns per game: the winner (N,) (-1 if none) and the placement of each seat (N, N_seats), where 1 = winner and
    a player who drops out later places better; players still alive at truncation share the best remaining place.
    Stops once every first game has ended."""
    n = len(acts)
    states = jax.vmap(env.init_state)(jrandom.split(key, n_games)) if starts is None else starts
    hist = jax.vmap(history_init)(states)
    n_games = states.armies.shape[0]

    def body(carry):
        states, hist, done, winner, out_at, key, t = carry
        key, *ks = jrandom.split(key, n + 1)
        actions = jnp.stack([act(states, i, k, hist) for i, (act, k) in enumerate(zip(acts, ks))], 1)
        hist = jax.vmap(history_push)(hist, states, actions)       # stale after a game ends; only the first counts
        ts, states = jax.vmap(lambda s, a: env.step(s, a, pool))(states, actions)
        live = ~done
        newly_out = ts.last_state.eliminated & (out_at < 0) & live[:, None]   # last_state: before auto-reset
        out_at = jnp.where(newly_out, t, out_at)
        ended = (ts.terminated | ts.truncated) & live
        winner = jnp.where(ended, ts.info.winner, winner)
        return states, hist, done | ended, winner, out_at, key, t + 1

    init = (states, hist, jnp.zeros(n_games, bool), jnp.full(n_games, -1), jnp.full((n_games, n), -1), key, 0)
    _, _, _, winner, out_at, _, _ = jax.lax.while_loop(lambda c: (c[6] < env.truncation) & ~c[2].all(), body, init)
    out_at = jnp.where(out_at < 0, env.truncation, out_at)
    place = 1 + (out_at[:, None, :] > out_at[:, :, None]).sum(-1)          # (games, seats)
    return winner, place


def play_match(acts, env: GeneralsEnv, pool, key, n_games: int = 64, starts=None):
    """play_games, summed up per seat: win fraction (N_seats,) and mean placement (N_seats,)."""
    winner, place = play_games(acts, env, pool, key, n_games, starts)
    wins = jnp.stack([jnp.mean(winner == i) for i in range(len(acts))])
    return wins, place.mean(0)


def head_to_head(apply_fn, params, opp_params, seat: int, env: GeneralsEnv, pool, key, n_games: int = 64):
    """`params` in `seat`, `opp_params` in every other seat, on maps drawn from `pool`. Returns (score, pairs): one
    pair per other seat and game, scored 1 if `params` placed better, 0.5 if equal. Also the games `params` won."""
    k_map, k_play = jrandom.split(key)
    idx = jrandom.randint(k_map, (n_games,), 0, pool.armies.shape[0])
    starts = jax.tree.map(lambda x: x[idx], pool)._replace(pool_idx=idx.astype(pool.pool_idx.dtype))
    acts = [policy_act(apply_fn, opp_params)] * env.num_players
    acts[seat] = policy_act(apply_fn, params)
    winner, place = play_games(acts, env, pool, k_play, n_games, starts)
    mine, others = place[:, seat:seat + 1], jnp.delete(place, seat, axis=1)
    return ((mine < others) + 0.5 * (mine == others)).sum(), others.size, (winner == seat).sum()


def eval_vs_bot(apply_fn, params, bot: str, env: GeneralsEnv, pool, key, n_games: int = 64):
    """Policy in seat 0, `bot` in every other seat: (win fraction, mean placement) of the policy."""
    acts = [policy_act(apply_fn, params)] + [bot_act(bot)] * (env.num_players - 1)
    wins, place = play_match(acts, env, pool, key, n_games)
    return wins[0], place[0]
