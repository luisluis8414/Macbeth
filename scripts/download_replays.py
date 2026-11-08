"""Download the recent FFA replays of given players (politely, resumable, deduplicated across players).

    uv run scripts/download_replays.py Human.exe --out data/raw/humanexe_ffa
    uv run scripts/download_replays.py --players-file players.txt --out data/raw/<set>

The site keeps replay files for only about a week (older ones return 404 although they are still listed), so by
default only games from the last --max-age-days are fetched. Run it daily to keep collecting.

Output: <out>/<replay id>.gior (shared by all players, a game with several of them is stored once) and
<out>/index.jsonl (one index record per downloaded game).
"""
import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://generals.io"
UA = "generals-rl research project (replay download, 1 req per few seconds)"


def fetch(url: str, delay: float, binary: bool = False):
    """GET with a pause before every request; retries network errors and server overload, raises on 4xx."""
    for attempt in range(6):
        time.sleep(delay)
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=30) as r:
                data = r.read()
            return data if binary else json.loads(data)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(60 * (attempt + 1))
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"  network error ({e}), retrying", flush=True)
            time.sleep(20 * (attempt + 1))
    raise RuntimeError(f"giving up on {url}")


def player_games(name: str, cutoff_ms: float, delay: float, game_type: str):
    games, offset = [], 0
    while True:
        q = urllib.parse.urlencode({"u": name, "offset": offset, "count": 200})
        page = fetch(f"{BASE}/api/replaysForUsername?{q}", delay)
        if not page:
            break
        games += [r for r in page if r.get("type") == game_type and r["started"] >= cutoff_ms]
        offset += len(page)
        if len(page) < 200 or min(r["started"] for r in page) < cutoff_ms:
            break
    return games


def main():
    p = argparse.ArgumentParser()
    p.add_argument("players", nargs="*")
    p.add_argument("--players-file")
    p.add_argument("--type", default="classic", help="classic = FFA")
    p.add_argument("--max-age-days", type=float, default=8)
    p.add_argument("--delay", type=float, default=2.0)
    p.add_argument("--out", default="data/raw/humanexe_ffa")
    args = p.parse_args()
    if args.delay < 1.0:
        p.error("--delay below 1 s is not polite to a small game's server")
    players = list(args.players)
    if args.players_file:
        players += [line.strip() for line in open(args.players_file) if line.strip()]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    have = {f.stem for f in out.glob("*.gior")}
    index = open(out / "index.jsonl", "a")
    cutoff = (time.time() - args.max_age_days * 86400) * 1000
    total_new = 0
    for name in players:
        games = player_games(name, cutoff, args.delay, args.type)
        todo = [g for g in games if g["id"] not in have]
        print(f"{name}: {len(games)} {args.type} games in the last {args.max_age_days:g} days, {len(todo)} new", flush=True)
        ok = missing = 0
        for g in todo:
            try:
                data = fetch(f"{BASE}/api/replay/{urllib.parse.quote(g['id'])}", args.delay, binary=True)
            except urllib.error.HTTPError as e:
                missing += 1
                if e.code != 404:
                    print(f"  {g['id']}: HTTP {e.code}", flush=True)
                continue
            (out / f"{g['id']}.gior").write_bytes(data)       # written only after a successful download
            index.write(json.dumps(g) + "\n")
            index.flush()
            have.add(g["id"])
            ok += 1
        total_new += ok
        print(f"  {ok} downloaded, {missing} no longer available", flush=True)
    print(f"done: {total_new} new replays, {len(have)} in {out}")


if __name__ == "__main__":
    main()
