"""Play a trained policy against Human.exe (EklipZ's generals-bot, set up with scripts/setup_humanexe.sh) on realistic
FFA maps (mapgen.py).

    uv run scripts/eval_humanexe.py --run runs/selfplay --players 4 --games 8      # 1 policy seat vs 3 Human.exe
    uv run scripts/eval_humanexe.py --run runs/selfplay --players 2 --games 1 --view

Seats 0 .. policy-seats-1 play the policy, the rest Human.exe; the seats are rotated over the games. Runs on the CPU;
Human.exe takes about 0.1 s per move and seat, so a 4-player game takes 3-10 minutes. Every game is appended to --log
as JSON.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import json
import time

import numpy as np

from macbeth.humanexe import HumanExe, policy_actor
from macbeth.play import load_policy, realistic_state


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--ckpt", default="latest")
    p.add_argument("--players", type=int, default=4)
    p.add_argument("--policy-seats", type=int, default=1)
    p.add_argument("--games", type=int, default=8)
    p.add_argument("--max-turns", type=int, default=2000)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--view", action="store_true", help="show the game in Human.exe's own viewer (a window), in real time")
    p.add_argument("--turn-time", type=float, default=0.25, help="with --view: seconds per turn (the real game: 0.5)")
    p.add_argument("--log", default="runs/humanexe_eval.jsonl")
    args = p.parse_args()

    apply_fn, params, run_cfg = load_policy(args.run, args.ckpt)
    pad = run_cfg["pad"]
    act = policy_actor(apply_fn, params, pad, args.greedy, args.temperature, args.seed)
    games = [realistic_state(args.seed * 1000 + g, args.players)[0] for g in range(args.games)]
    print(f"{len(games)} maps, {args.players} players, pad {pad}, policy {args.run} {args.ckpt}", flush=True)
    wins = places = 0
    with HumanExe(log_path="runs/humanexe_host.log") as bot:
        for g, start in enumerate(games):
            seats = [(g + k) % args.players for k in range(args.policy_seats)]      # rotate who sits where
            t0 = time.time()
            viewer = next(s for s in range(args.players) if s not in seats) if args.view else -1
            res = bot.play(start, {s: act for s in seats}, args.max_turns, viewer, args.turn_time)
            out = [t if t is not None else 10**9 for t in res["out_at"]]
            place = {s: 1 + sum(o > out[s] for o in out) for s in range(args.players)}  # later out = better place
            won = res["winner"] in seats
            wins += won
            places += np.mean([place[s] for s in seats])
            row = dict(game=g, run=args.run, ckpt=args.ckpt, players=args.players, policy_seats=seats, winner=res["winner"],
                       out_at=res["out_at"], turns=res["turns"], policy_won=bool(won), seconds=round(time.time() - t0))
            with open(args.log, "a") as f:
                f.write(json.dumps(row) + "\n")
            verdict = "won" if won else "cut off" if res["winner"] is None else "lost"
            print(f"game {g + 1}: {verdict} (winner seat {res['winner']}, policy seats {seats}), "
                  f"{res['turns']} turns, {row['seconds']} s", flush=True)
    n = len(games)
    print(f"policy won {wins} of {n} ({wins / max(n, 1):.0%}), mean place {places / max(n, 1):.2f} of {args.players}")


if __name__ == "__main__":
    main()
