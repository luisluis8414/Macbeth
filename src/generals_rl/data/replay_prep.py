# Vendored from generals-bots (MIT, Matej Straka), paper/validation/ffa/board_diff_ffa.py @ 954927f:
# `prepare` (N-player replay -> padded grid, per-tick actions, AFK events, teams) and `afk_event`, verbatim;
# the constants of paper/validation/replay_prep.py are inlined as the namespace `rp`.
from types import SimpleNamespace

import jax
import numpy as np

from generals.core import game

rp = SimpleNamespace(
    MOUNTAIN=-2,
    PASS=(1, 0, 0, 0, 0),
    DELTA_TO_DIR={(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3},   # UP DOWN LEFT RIGHT
)

PAD = 36          # FFA maps go up to 33 x 33
NP = 8            # every game is padded to this many players (absent players start eliminated: no general on the grid)
PASS = np.array(rp.PASS, dtype=np.int32)


def prepare(row: dict, n_pad: int = NP, pad: int = PAD) -> dict:
    """N-player variant of replay_prep.prepare: grid values 1..N are the generals, cities > N;
    actions (T, n_pad, 5); afk_events (T, n_pad) = number of site AFK events for that player at that tick;
    teams (n_pad,) team ids (the replay's `teams` when present, else arange; padded players get their own teams)."""
    w, h = int(row["mapWidth"]), int(row["mapHeight"])
    generals = list(row["generals"] or []); N = len(row["usernames"] or [])
    meta = {"id": row["id"], "version": int(row["version"]), "map_w": w, "map_h": h, "n_players": N}
    if N < 2 or N > n_pad: return {"exclude": f"{N} players", "meta": meta}
    if len(generals) != N or any(g is None or g < 0 or g >= w * h for g in generals): return {"exclude": "generals", "meta": meta}
    if w > pad or h > pad: return {"exclude": f"map larger than {pad}", "meta": meta}
    cities, city_armies, mountains = list(row["cities"] or []), list(row["cityArmies"] or []), list(row["mountains"] or [])
    if len(cities) != len(city_armies) or any(a <= n_pad for a in city_armies): return {"exclude": "city armies", "meta": meta}
    for k in ("modifiers", "lookouts", "observatories", "tunnels", "swamps", "deserts", "strongholds", "neutrals"):
        if row.get(k) or (row.get("extras") or {}).get(k): return {"exclude": f"not the base game: {k}", "meta": meta}
    special = set(cities) | set(mountains) | set(generals)
    if len(special) != len(cities) + len(mountains) + len(generals): return {"exclude": "overlapping special tiles", "meta": meta}
    teams_raw = row.get("teams")
    if teams_raw:
        if len(teams_raw) != N: return {"exclude": "teams length", "meta": meta}
        ids = {t: i for i, t in enumerate(dict.fromkeys(teams_raw))}
        teams = np.array([ids[t] for t in teams_raw] + list(range(len(ids), len(ids) + n_pad - N)), dtype=np.int32)
    else:
        teams = np.arange(n_pad, dtype=np.int32)
    grid = np.full((pad, pad), rp.MOUNTAIN, dtype=np.int32); board = np.zeros(h * w, dtype=np.int32)
    board[mountains] = rp.MOUNTAIN
    for t, a in zip(cities, city_armies): board[t] = int(a)
    for p, g in enumerate(generals): board[g] = p + 1
    grid[:h, :w] = board.reshape(h, w)
    moves = row["moves"] or []
    if not moves: return {"exclude": "no moves", "meta": meta}
    turns = [m[4] for m in moves]
    if turns != sorted(turns): return {"exclude": "moves not sorted", "meta": meta}
    T = max(int(turns[-1]), max([a["turn"] for a in row["afks"]] + [0])) + 1
    actions = np.tile(PASS, (T, n_pad, 1)); seen = set(); n_dup = 0
    for p, s, e, is50, t in moves:
        p, s, e, t = int(p), int(s), int(e), int(t)
        if not (0 <= p < N) or t < 0 or not (0 <= s < w * h and 0 <= e < w * h): return {"exclude": "bad move", "meta": meta}
        r0, c0 = divmod(s, w); r1, c1 = divmod(e, w); d = rp.DELTA_TO_DIR.get((r1 - r0, c1 - c0))
        if d is None: return {"exclude": "non-adjacent move", "meta": meta}
        if (p, t) in seen: n_dup += 1; continue
        seen.add((p, t)); actions[t, p] = (0, r0, c0, d, 1 if is50 else 0)
    afk = np.zeros((T, n_pad), dtype=np.int32)
    for a in row["afks"]: afk[int(a["turn"]), int(a["index"])] += 1
    meta.update(n_moves=len(moves) - n_dup, n_dropped_duplicate=n_dup, n_ticks=T, n_afk_events=len(row["afks"]),
                first_afk_turn=min([a["turn"] for a in row["afks"]], default=-1), teams=teams[:N].tolist(), team_game=bool(teams_raw))
    return {"exclude": None, "grid": grid, "actions": actions, "afk": afk, "teams": teams, "meta": meta}


def afk_event(s, p):
    """One site AFK event for player p (processReplayAfks): killPlayer while alive, tryNeutralizePlayer once dead."""
    return jax.lax.cond(s.eliminated[p], lambda s: game.neutralize(s, p), lambda s: game.surrender(s, p), s)
