"""Play a trained policy on generals.io, through generals-bots' client.

The policy reads a full simulator GameState and applies the fog itself (features.observe). The server only sends
what we can see, so `state_from_server` builds a GameState that gives the same observation:

- visible cells are copied as they are;
- each opponent's land and army in the fog, known from the public scoreboard, are put on fog cells, where the
  observation never looks at them except for the scoreboard totals;
- obstacles in the fog count as mountains unless the server has told us they are cities.

What differs from training: the opponents' moves are not sent by the server, so the planes of enemy moves stay
empty, and the action mask treats an unknown obstacle in the fog as a mountain.
"""
import http.server
import json
import threading
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from generals.core.game import GameState

from .features import PASS

MOUNTAIN, FOG, FOG_OBSTACLE = -2, -3, -4


def seat_order(gs) -> list[int]:
    """Server player index of each seat: seat 0 is us, then the others in server order."""
    me = gs.player_index
    return [me] + [i for i in range(len(gs.usernames)) if i != me]


def state_from_server(gs, pad: int) -> GameState:
    """GameState (us = player 0, the others = players 1.. in server order) from generals-bots' GeneralsIOstate, padded
    with mountains. Works for 1v1 and FFA up to 8 players."""
    w, h = gs.map[0], gs.map[1]
    if max(h, w) > pad:
        raise ValueError(f"map {h}x{w} is larger than the padding {pad}")
    armies = np.asarray(gs.map[2: 2 + h * w]).reshape(h, w)
    terrain = np.asarray(gs.map[2 + h * w: 2 + 2 * h * w]).reshape(h, w)
    seats = seat_order(gs)
    n = len(seats)

    cities = np.zeros((h, w), bool)
    for c in gs.castles:
        cities[c // w, c % w] = True
    generals = np.zeros((h, w), bool)
    general_positions = np.zeros((n, 2), np.int32)
    for player, g in enumerate(gs.generals):
        if g != -1:
            generals[g // w, g % w] = True
            if player in seats:
                general_positions[seats.index(player)] = (g // w, g % w)
    mountains = (terrain == MOUNTAIN) | ((terrain == FOG_OBSTACLE) & ~cities)
    cities &= ~generals

    owned = np.stack([terrain == player for player in seats])                 # (n, h, w)
    army = np.where(terrain >= -1, armies, 0)
    # Each opponent's hidden land and army, from the scoreboard, on fog cells of its own (any: the observation hides
    # them and only the scoreboard totals see them).
    score = {s["i"]: s for s in gs.scores}
    dead = np.array([bool(score.get(player, {}).get("dead", False)) for player in seats])
    fog = list(np.flatnonzero((terrain == FOG).ravel()))
    for k, player in enumerate(seats[1:], start=1):
        if player not in score or dead[k]:
            continue
        hidden_land = max(0, score[player]["tiles"] - int(owned[k].sum()))
        hidden_army = max(0, score[player]["total"] - int((army * owned[k]).sum()))
        cells, fog = fog[:hidden_land], fog[hidden_land:]
        if cells:
            owned[k].ravel()[cells] = True
            army.ravel()[cells] = 1
            army.ravel()[cells[0]] += max(0, hidden_army - len(cells))

    def padded(x, fill):
        out = np.full((pad, pad), fill, x.dtype)
        out[:h, :w] = x
        return out

    mountains_p = padded(mountains, True)
    return GameState(
        armies=jnp.asarray(padded(army.astype(np.int32), 0)),
        ownership=jnp.asarray(np.stack([padded(o, False) for o in owned])),
        ownership_neutral=jnp.asarray(padded(terrain == -1, False)),
        generals=jnp.asarray(padded(generals, False)),
        castles=jnp.asarray(padded(cities, False)),
        mountains=jnp.asarray(mountains_p),
        passable=jnp.asarray(~mountains_p),
        general_positions=jnp.asarray(general_positions),
        teams=jnp.arange(n, dtype=jnp.int32),
        eliminated=jnp.asarray(dead),
        time=jnp.asarray(gs.turn - 1, jnp.int32),
        winner=jnp.asarray(-1, jnp.int32),
        pool_idx=jnp.asarray(0, jnp.int32),
    )


class LiveView:
    """The bot's current game for web/live.html: what the server sent, the move and what the policy thought of it.
    Serves the page and /api/frame on localhost:`port` from a background thread (port 0: no server)."""

    def __init__(self, port: int):
        self.frame = {"status": "starting", "seq": 0}
        if port:
            page = Path(__file__).resolve().parents[2] / "web" / "live.html"
            view = self

            class Handler(http.server.BaseHTTPRequestHandler):
                def do_GET(self):  # noqa: N802
                    if self.path.startswith("/api/frame"):
                        body, kind = json.dumps(view.frame).encode(), "application/json"
                    else:
                        body, kind = page.read_bytes(), "text/html; charset=utf-8"
                    self.send_response(200)
                    self.send_header("Content-Type", kind)
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)

                def log_message(self, *args):
                    pass

            server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
            threading.Thread(target=server.serve_forever, daemon=True).start()

    def status(self, text: str, replay: str | None = None) -> None:
        self.frame = {**self.frame, "status": text, "replay": replay, "seq": self.frame["seq"] + 1}

    def update(self, gs, move, value: float, pass_prob: float, per_cell: np.ndarray) -> None:
        w, h = gs.map[0], gs.map[1]
        probs = per_cell[:h, :w].ravel()
        score = {s["i"]: s for s in gs.scores}
        self.frame = {
            **self.frame, "seq": self.frame["seq"] + 1, "w": w, "h": h, "turn": gs.turn, "me": gs.player_index,
            "names": gs.usernames, "scores": [[score[i]["tiles"], score[i]["total"]] if i in score else [0, 0]
                                              for i in range(len(gs.usernames))],
            "armies": list(gs.map[2: 2 + h * w]), "terrain": list(gs.map[2 + h * w: 2 + 2 * h * w]),
            "cities": list(gs.castles), "generals": list(gs.generals),
            "move": None if move is None else [move[0], move[1]], "value": round(value, 3), "pass": round(pass_prob, 3),
            "probs": [[int(i), round(float(probs[i]), 3)] for i in np.flatnonzero(probs >= 0.005)],
        }


def all_actions(ours, n: int):
    """Our action and a pass for each of the n - 1 others, whose moves the server does not send."""
    return jnp.stack([ours] + [jnp.array(PASS, jnp.int32)] * (n - 1))
