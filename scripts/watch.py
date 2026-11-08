"""Watch a trained policy play a live 8-player game in the browser: against scripted bots, copies of itself, an older
checkpoint or Human.exe, on a realistic map, a map from the replay data or a generated square map. The page shows
what the policy sees (every observation plane) and what it thinks (move probabilities, value).

    uv run scripts/watch.py            # then open http://localhost:8051; lists every run in runs/

The game runs locally on the CPU, one tick at a time (JAX, jit-compiled; the very first tick compiles for
~30-90 s). The page asks for the next ticks at the chosen speed, so you can pause, speed up or rewind.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")             # leave the GPU to training

import argparse
import http.server
import json
import threading
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from flax.serialization import msgpack_restore

from macbeth.features import PLANE_NAMES
from macbeth.play import LiveGame, bot_lineup

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "web" / "watch.html"
lock = threading.Lock()
current: dict = {}
fits: dict = {}


def fits_current_code(run: Path, conf: dict) -> bool:
    """False for runs of older network layouts (before the CNN decoder, relative positions and the binned value) or
    observations (not 82 planes)."""
    if conf.get("value_bins", 128) <= 0 or conf.get("cell_head", 1) != 1 or conf.get("rel_pos", 1) != 1:
        return False
    params = next(iter(sorted(run.glob("params_*.msgpack"))), None)
    if params is None:
        return False
    if run not in fits:
        kernel = msgpack_restore(params.read_bytes())["params"].get("Conv_0", {}).get("kernel")
        fits[run] = kernel is not None and kernel.shape[-2] == len(PLANE_NAMES)
    return fits[run]


def checkpoints() -> list[dict]:
    out = []
    for cfg in sorted(ROOT.glob("runs/*/config.json")):
        run = cfg.parent
        if not fits_current_code(run, json.loads(cfg.read_text())):
            continue
        for p in sorted(run.glob("params_*.msgpack")):
            out.append({"run": str(run.relative_to(ROOT)), "ckpt": p.stem.removeprefix("params_"), "mtime": p.stat().st_mtime})
    return out


class Handler(http.server.BaseHTTPRequestHandler):
    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path == "/":
                body = PAGE.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(body)
            elif url.path == "/api/checkpoints":
                self._json(checkpoints())
            elif url.path == "/api/new":
                players = int(q.get("players", 8))
                preset = q.get("bots", "expander")
                if preset == "mix" and players == 2:                           # one opponent: the seed picks which
                    preset = bot_lineup("mix", 4)[int(q.get("seed", 0)) % 4]
                bots = bot_lineup(preset, players - 1)
                board = q.get("size", "real")
                if board in ("real", "replay"):
                    bots = bot_lineup(q.get("bots", "expander"), 7)            # the map decides how many seats are used
                past = None
                if q.get("past"):                                              # "run|checkpoint" for the past seats
                    past_run, past_ckpt = q["past"].split("|")
                    past = (str(ROOT / past_run), past_ckpt)
                with lock:
                    if current.get("game") is not None:
                        current["game"].close()                    # ends a running game against Human.exe
                    g = LiveGame(str(ROOT / q["run"]), bots, board=board, seed=int(q.get("seed", 0)),
                                 greedy=q.get("greedy") == "1", ckpt=q.get("ckpt", "latest"),
                                 data=str(ROOT / "data/processed/humanexe_ffa.npz"), past=past)
                    current["game"] = g
                    self._json(g.meta())
            elif url.path == "/api/obs":
                with lock:
                    g = current.get("game")
                    if g is None:
                        return self._json({"error": "no game started"}, 400)
                    self._json(g.observation(int(q.get("i", 0))))
            elif url.path == "/api/step":
                with lock:
                    g = current.get("game")
                    if g is None:
                        return self._json({"error": "no game started"}, 400)
                    self._json({"frames": g.advance(max(1, min(int(q.get("n", 1)), 50)))})
            else:
                self._json({"error": "not found"}, 404)
        except Exception as e:  # noqa: BLE001  (report errors to the page)
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def log_message(self, *args):
        pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8051)
    args = p.parse_args()
    print(f"live viewer on http://localhost:{args.port}  (Ctrl+C to stop)")
    if not checkpoints():
        print("no checkpoints in runs/ yet (README: train one, or fetch one from Modal with modal volume get)")
    http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
