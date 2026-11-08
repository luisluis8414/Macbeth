"""Sanity check of the setup: one 8-player FFA game between scripted bots, then a throughput benchmark.

    uv run scripts/smoke_test.py [--envs 1024] [--steps 200] [--players 8] [--size 20]
"""
import argparse
import time

import jax
import jax.numpy as jnp
import jax.random as jrandom

from generals import GeneralsEnv, get_observation
from generals.agents import ExpanderAgent
from generals.core import game

from generals_rl.features import decode_action, encode_action, observe


def single_game(env, key):
    pool, state = env.reset(key)
    agent = ExpanderAgent()
    n = env.num_players
    for t in range(env.truncation):
        key, *ks = jrandom.split(key, n + 1)
        actions = jnp.stack([agent.act(get_observation(state, i), ks[i]) for i in range(n)])
        before = state
        ts, state = env.step(state, actions, pool)
        if bool(ts.terminated | ts.truncated):
            break
    obs = observe(before, 0)
    h, w = before.armies.shape
    if int(actions[0, 0]) == 0:
        idx = encode_action(actions[0], w, h)
        assert bool(obs["action_mask"][idx]), "bot chose a move outside our action mask"
        assert jnp.array_equal(decode_action(idx, w, h), actions[0])
    print(f"single game: {n} Expanders, {t + 1} ticks, winner={int(ts.info.winner)}, "
          f"planes {obs['planes'].shape}, {int(obs['action_mask'].sum())}/{obs['action_mask'].size} legal actions")


def benchmark(env, key, n_envs, n_steps):
    pool, _ = env.reset(key)
    states = jax.vmap(env.init_state)(jrandom.split(key, n_envs))
    agent = ExpanderAgent()
    n = env.num_players

    @jax.jit
    def rollout(states, key):
        def body(carry, _):
            states, key = carry
            key, sub = jrandom.split(key)
            obs = jax.vmap(game.get_observations)(states)                          # (E, n, ...)
            keys = jrandom.split(sub, n_envs * n).reshape(n_envs, n, -1)
            acts = jax.vmap(jax.vmap(agent.act))(obs, keys)                          # (E, n, 5)
            _, states = jax.vmap(lambda s, a: env.step(s, a, pool))(states, acts)
            return (states, key), None
        (states, key), _ = jax.lax.scan(body, (states, key), None, length=n_steps)
        return states

    jax.block_until_ready(rollout(states, key))  # compile
    t0 = time.perf_counter()
    jax.block_until_ready(rollout(states, key))
    dt = time.perf_counter() - t0
    print(f"benchmark: {n_envs} envs x {n_steps} ticks, {n} Expanders incl. observations: "
          f"{n_envs * n_steps / dt:,.0f} game ticks/s on {jax.devices()[0].platform}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--envs", type=int, default=1024)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--players", type=int, default=8)
    p.add_argument("--size", type=int, default=27)
    args = p.parse_args()
    env = GeneralsEnv(grid_dims=(args.size, args.size), pad_to=36, truncation=1500, num_players=args.players,
                      min_generals_distance=5, pool_size=1024)
    key = jrandom.PRNGKey(0)
    print("devices:", jax.devices())
    single_game(env, key)
    benchmark(env, key, args.envs, args.steps)
