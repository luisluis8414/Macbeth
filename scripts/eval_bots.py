"""Checkpoints against the scripted bots: the policy in seat 0, the bot in every other seat, on the same maps.

    uv run scripts/eval_bots.py --policy runs/clone:latest runs/selfplay:latest \
        --bots expander harvester --players 8 --size 27 --games 64

A fair share is 1 / players wins and placement (players + 1) / 2.
"""
import argparse
import time

import jax
import jax.random as jrandom

from generals_rl.evaluate import eval_vs_bot, make_eval_env
from generals_rl.play import load_policy


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", nargs="+", required=True, help="run:checkpoint, e.g. runs/selfplay:latest")
    p.add_argument("--bots", nargs="+", default=["expander", "harvester"])
    p.add_argument("--players", type=int, default=8)
    p.add_argument("--size", type=int, default=27)
    p.add_argument("--games", type=int, default=64)
    p.add_argument("--truncation", type=int, default=1500)
    args = p.parse_args()

    n = args.players
    print(f"{n}-player FFA on {args.size}x{args.size}, {args.games} games each "
          f"(fair share {1 / n:.1%} wins, placement {(n + 1) / 2:.1f})", flush=True)
    for spec in args.policy:
        run, ckpt = spec.rsplit(":", 1)
        apply_fn, params, cfg = load_policy(run, ckpt)
        env = make_eval_env(args.size, n, args.truncation, pad_to=cfg["pad"])
        pool, _ = env.reset(jrandom.PRNGKey(0))
        for bot in args.bots:
            t0 = time.perf_counter()
            win, place = jax.jit(lambda prm, k: eval_vs_bot(apply_fn, prm, bot, env, pool, k, args.games))(
                params, jrandom.PRNGKey(1))
            print(f"{spec:<32} vs {n - 1}x {bot:<10} wins {float(win):6.1%}  place {float(place):5.2f}"
                  f"   ({time.perf_counter() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
