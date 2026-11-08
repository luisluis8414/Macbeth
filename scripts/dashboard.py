"""Local live dashboard for training runs on Modal.

    uv run scripts/dashboard.py            # then open http://localhost:8050

The training container mirrors runs/<name>/metrics.jsonl into the Modal Dict "generals-rl-metrics" every 15 s
(scripts/modal_app.py). This server reads that Dict and serves web/dashboard.html, which refreshes itself.
It only reads; it cannot start, stop or change runs.
"""
import argparse
import http.server
import json
import shlex
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import modal

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "web" / "dashboard.html"
live = modal.Dict.from_name("generals-rl-metrics")


def _total_steps(args: str) -> int | None:
    argv = shlex.split(args or "")
    return int(argv[argv.index("--total-steps") + 1]) if "--total-steps" in argv else None


def _summary(name: str, run: dict) -> dict:
    rows = run.get("rows", [])
    end = run.get("ended") or time.time()
    return {
        "name": name, "status": run.get("status"), "gpu": run.get("gpu"), "usd_per_hour": run.get("usd_per_hour"),
        "started": run.get("started"), "ended": run.get("ended"), "updated": run.get("updated"),
        "elapsed_h": (end - run["started"]) / 3600 if run.get("started") else None,
        "max_hours": run.get("max_hours"), "total_steps": _total_steps(run.get("args", "")),
        "step": max((r["step"] for r in rows), default=0), "n_rows": len(rows), "args": run.get("args"),
    }


class Handler(http.server.BaseHTTPRequestHandler):
    def _send(self, body: bytes, ctype: str, code: int = 200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        url = urlparse(self.path)
        try:
            if url.path == "/":
                self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/api/runs":
                runs = [_summary(k, live.get(k)) for k in live.keys()]
                runs.sort(key=lambda r: r["started"] or 0, reverse=True)
                self._send(json.dumps(runs).encode(), "application/json")
            elif url.path == "/api/run":
                name = parse_qs(url.query).get("name", [""])[0]
                run = live.get(name)
                if run is None:
                    self._send(b'{"error": "unknown run"}', "application/json", 404)
                else:
                    self._send(json.dumps({**_summary(name, run), "rows": run.get("rows", [])}).encode(), "application/json")
            else:
                self._send(b"not found", "text/plain", 404)
        except Exception as e:  # noqa: BLE001  (show Modal/network errors in the page instead of crashing)
            self._send(json.dumps({"error": str(e)}).encode(), "application/json", 502)

    def log_message(self, *args):
        pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8050)
    args = p.parse_args()
    print(f"dashboard on http://localhost:{args.port}  (Ctrl+C to stop)")
    http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
