"""Convert downloaded FFA replays (.gior) into one compact training file and report what is in it.

    uv run scripts/prepare_replays.py data/raw/humanexe_ffa                # -> data/processed/humanexe_ffa.npz
    uv run scripts/prepare_replays.py data/raw/humanexe_ffa --validate 200 # + replay 200 games through the simulator

Whole games are dropped only for technical reasons: not FFA, special map features or modifiers, broken data, map
larger than 36x36, fewer than --min-ticks ticks. Who gets imitated (player, stars, AFK, ...)
is decided later in bc.py from the per-player metadata stored here, so thresholds can change without
re-running this script.

Output arrays (G games, NP = 8 player slots):
    grids (G, 36, 36) int16, teams (G, 8), n_players (G,), n_ticks (G,)
    moves (M, 6) int16 [tick, player, row, col, dir, split] + move_offsets (G+1,)
    afks (A, 3) int16 [tick, player, count] + afk_offsets (G+1,)
    stars (G, 8), first_afk (G, 8) (-1 = never), n_moves (G, 8)
    meta: JSON list (id, usernames, map size)
"""
import argparse
import collections
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from macbeth.data import replay_prep, replays

NP = replays.NP


def pct(xs, q):
    return int(np.percentile(xs, q)) if len(xs) else 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("gior_dir", help="directory with .gior files, e.g. data/raw/humanexe_ffa")
    p.add_argument("--out", default=None, help="default: data/processed/<dir name>.npz")
    p.add_argument("--min-ticks", type=int, default=50)
    p.add_argument("--max-ticks", type=int, default=3000)
    p.add_argument("--validate", type=int, default=0, help="replay this many games through the simulator")
    args = p.parse_args()
    out = Path(args.out or f"data/processed/{Path(args.gior_dir).name}.npz")

    excluded = collections.Counter()
    rec = collections.defaultdict(list)
    for name, row in tqdm(replays.iter_gior(args.gior_dir), desc="replays", unit=" games"):
        if not replays.is_ffa(row):
            excluded["not FFA (1v1 or teams)"] += 1
            continue
        prep = replay_prep.prepare(row)
        if prep["exclude"] is not None:
            excluded[prep["exclude"]] += 1
            continue
        t = prep["meta"]["n_ticks"]
        if t < args.min_ticks or t > args.max_ticks:
            excluded[f"length outside {args.min_ticks}..{args.max_ticks} ticks"] += 1
            continue
        n = prep["meta"]["n_players"]
        r = replays.to_record(prep)
        stars = np.zeros(NP, np.int16)
        given = row["stars"][:n]                       # can be empty (unrated games): missing stars count as 0
        stars[:len(given)] = given
        first_afk = np.full(NP, -1, np.int32)
        for a in sorted(row["afks"], key=lambda a: a["turn"], reverse=True):
            first_afk[a["index"]] = a["turn"]
        n_moves = np.bincount(r["moves"][:, 1].astype(np.int64), minlength=NP)[:NP]
        rec["grids"].append(r["grid"])
        rec["teams"].append(r["teams"])
        rec["moves"].append(r["moves"])
        rec["afks"].append(r["afks"])
        rec["stars"].append(stars)
        rec["first_afk"].append(first_afk)
        rec["n_moves"].append(n_moves)
        rec["n_players"].append(n)
        rec["n_ticks"].append(t)
        rec["meta"].append(dict(id=row["id"], usernames=row["usernames"], map=[row["mapWidth"], row["mapHeight"]]))

    games = len(rec["grids"])
    print(f"\nkept {games} games, excluded {sum(excluded.values())}: {dict(excluded.most_common())}")
    if not games:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        grids=np.stack(rec["grids"]), teams=np.stack(rec["teams"]),
        n_players=np.array(rec["n_players"]), n_ticks=np.array(rec["n_ticks"]),
        moves=np.concatenate(rec["moves"]), move_offsets=np.cumsum([0] + [len(m) for m in rec["moves"]]),
        afks=np.concatenate(rec["afks"]), afk_offsets=np.cumsum([0] + [len(a) for a in rec["afks"]]),
        stars=np.stack(rec["stars"]), first_afk=np.stack(rec["first_afk"]), n_moves=np.stack(rec["n_moves"]),
        meta=json.dumps(rec["meta"]),
    )
    print(f"wrote {out}")

    # ---- report
    n_players = np.array(rec["n_players"])
    ticks = np.array(rec["n_ticks"])
    stars = np.stack(rec["stars"])
    afk = np.stack(rec["first_afk"])
    real = np.arange(NP)[None] < n_players[:, None]
    print("\n== contents")
    print("players per game:", dict(sorted(collections.Counter(n_players.tolist()).items())))
    print(f"ticks: median {pct(ticks, 50)}, p10 {pct(ticks, 10)}, p90 {pct(ticks, 90)}")
    print(f"moves: {sum(len(m) for m in rec['moves']):,}")
    s = stars[real]
    print(f"player stars: median {pct(s, 50)}, p90 {pct(s, 90)}; >= 50 stars: {(s >= 50).mean():.0%} of player slots")
    print(f"players with an AFK event: {(afk[real] >= 0).mean():.0%}, games with any AFK: {(afk >= 0).any(1).mean():.0%}")

    if args.validate:
        d = np.load(out)
        fracs, synced = [], 0
        for g in range(min(args.validate, games)):
            grid, actions, afk_g, teams, t = replays.load_game(d, g)
            tp = replays.bucket(t)
            _, legal, winner, _ = replays.simulate(grid, replays.pad_ticks(actions, tp, replays.PASS),
                                                   replays.pad_ticks(afk_g, tp), teams)
            legal, winner = np.asarray(legal)[:t], np.asarray(winner)[:t]
            moves = actions[:, :, 0] == 0
            fracs.append(legal[moves].mean() if moves.any() else 1.0)
            synced += int(winner[-1] >= 0 and (winner[:-1] < 0).all())
        print(f"\n== validation ({len(fracs)} games through the simulator)")
        print(f"legal recorded moves: {np.mean(fracs):.2%}; games decided exactly on their last recorded tick: "
              f"{synced}/{len(fracs)} (the rest ended by surrender/timeout or desynced)")


if __name__ == "__main__":
    main()
