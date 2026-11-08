"""Run the pipeline on Modal (cloud GPUs). Everything persistent lives on the Modal volume "generals-rl":

    /vol/data/raw/<set>/          uploaded replays (modal volume put)
    /vol/data/processed/          prepared training files
    /vol/runs/<name>/             checkpoints + metrics.jsonl
    /vol/jax_cc/                  compiled XLA programs, so a repeated or resumed run skips most of the compiling

Commands (run from the repo root on any machine; the work happens in the cloud):

    uv run modal run scripts/modal_app.py --cmd check                       # GPU, JAX, simulator speed
    uv run modal run --detach scripts/modal_app.py --cmd prepare --args data/raw/humanexe_ffa   # replays -> .npz
    uv run modal run --detach scripts/modal_app.py --cmd train --gpu L40S --hours 2 \\
        --args "--out runs/clone"                                           # behavior cloning
    uv run modal run --detach scripts/modal_app.py --cmd selfplay --gpu H100 --hours 6 \\
        --args "--out runs/selfplay ..."                                    # self-play (configs/selfplay_modal.sh)
--args is passed through to scripts/train_bc.py / train_selfplay.py / prepare_replays.py (paths relative to /vol).
--cmd selfplay hands the run to Modal and returns at once (always with --detach): the machine that started it can
be switched off. --cmd train waits for the run, so a shutdown of the starting machine cancels it.
A self-play run continues from its runs/<name>/resume.msgpack, so starting the same command again resumes it.
Use --detach for anything longer than a few minutes: without it, Modal stops the run when the local connection
drops (e.g. flaky Wi-Fi).

Cost guard: --hours is a hard limit. Modal kills the container after it (function timeout) and the training stops
itself shortly before that (--time-limit-hours), so a run can never cost more than hours x GPU price. The
estimate is printed before the run starts.
"""
import json
import os
import shlex
import subprocess
import threading
import time
from pathlib import Path

import modal

APP_NAME = "generals-rl"
PRICE_PER_HOUR = {"T4": 0.59, "L4": 0.80, "A10": 1.10, "L40S": 1.95, "A100": 2.10, "A100-40GB": 2.10,
                  "A100-80GB": 2.50, "H100": 3.95, "H200": 4.54}   # USD, modal.com/pricing, 2026-11
VOL = "/vol"
SIM = "generals-bots @ https://github.com/strakam/generals-bots/archive/954927f508aa47899019d66317cc313af3e9cc19.tar.gz"

image = (
    modal.Image.debian_slim(python_version="3.12")
    # pinned: the XLA compilation cache on the volume (/vol/jax_cc) only holds for the same JAX version
    .uv_pip_install("jax[cuda12]==0.12.0.1", "flax==0.12.12", "optax==0.2.8", "numpy>=2.0", "scipy>=1.11", "tqdm>=4.66", SIM)
    .add_local_python_source("generals_rl")
    .add_local_dir("scripts", remote_path="/root/scripts")
)
app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(APP_NAME, create_if_missing=True)
# Live metrics of training runs for scripts/dashboard.py: {run name: {status, gpu, rows, ...}}
live = modal.Dict.from_name(f"{APP_NAME}-metrics", create_if_missing=True)


def _run(cmd: str, **env):
    """Run a command in /vol (so data/ and runs/ land on the volume) and stream its output."""
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd=VOL, env={**os.environ, "PYTHONUNBUFFERED": "1", **env})


def _commit_every(seconds: int, stop: threading.Event):
    """Persist checkpoints regularly, so a crash or timeout loses at most `seconds` of results."""
    while not stop.wait(seconds):
        volume.commit()


CHEAP_GPUS = ["L4", "A10", "T4", "L40S"]   # first one that is free; all cheap enough for checks and small runs


@app.function(gpu=CHEAP_GPUS, volumes={VOL: volume}, timeout=30 * 60)
def check():
    _run("nvidia-smi --query-gpu=name,memory.total --format=csv")
    _run("python -c \"import jax; print(jax.__version__, jax.devices())\"")
    _run("python /root/scripts/smoke_test.py --envs 1024 --steps 200")


@app.function(cpu=4, memory=8192, volumes={VOL: volume}, timeout=60 * 60)
def prepare(args: str = ""):
    _run(f"python /root/scripts/prepare_replays.py {args}", JAX_PLATFORMS="cpu")
    volume.commit()


def _gpu_name() -> str:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip().splitlines()[0]
    except Exception:  # noqa: BLE001
        return "unknown"


def _price(gpu_name: str) -> float:
    """USD/h for a name like 'NVIDIA A10' or 'NVIDIA L40S' (longest matching key wins)."""
    hits = [k for k in PRICE_PER_HOUR if k.split("-")[0] in gpu_name]
    return PRICE_PER_HOUR[max(hits, key=len)] if hits else float("nan")


def _stream_metrics(run: str, out_dir: str, info: dict, stop: threading.Event, every: int = 15):
    """Mirror runs/<name>/metrics.jsonl into the Modal Dict every `every` seconds (live dashboard)."""
    path = Path(VOL) / out_dir / "metrics.jsonl"
    rows, seen = [], 0
    while True:
        stopped = stop.wait(every)
        if path.exists():
            lines = path.read_text().splitlines()
            rows += [json.loads(line) for line in lines[seen:] if line.strip()]
            seen = len(lines)
        try:
            live[run] = {**info, "rows": rows, "updated": time.time()}
        except Exception as e:  # noqa: BLE001  (the dashboard must never break training)
            print(f"live metrics: {e}", flush=True)
        if stopped:
            return


@app.function(gpu=CHEAP_GPUS, volumes={VOL: volume}, timeout=2 * 60 * 60)
def train(args: str = "", max_hours: float = 2.0, script: str = "train_bc.py"):
    volume.reload()                      # see data uploaded since the container started
    argv = shlex.split(args)
    out_dir = argv[argv.index("--out") + 1] if "--out" in argv else "runs/bc"
    run = Path(out_dir).name
    gpu = _gpu_name()
    info = {"status": "running", "gpu": gpu, "usd_per_hour": _price(gpu), "started": time.time(),
            "max_hours": max_hours, "args": args}
    stop, stop_live = threading.Event(), threading.Event()
    threading.Thread(target=_commit_every, args=(600, stop), daemon=True).start()
    streamer = threading.Thread(target=_stream_metrics, args=(run, out_dir, info, stop_live), daemon=True)
    streamer.start()
    try:
        # compiled XLA programs persist on the volume: the map generator alone takes many minutes to compile, so a
        # resumed or repeated run would otherwise pay for that again
        _run(f"python /root/scripts/{script} {args}", XLA_PYTHON_CLIENT_MEM_FRACTION="0.90",
             JAX_COMPILATION_CACHE_DIR=f"{VOL}/jax_cc")
        info["status"] = "finished"
    except Exception:
        info["status"] = "failed"
        raise
    finally:
        info["ended"] = time.time()
        stop.set()
        stop_live.set()
        streamer.join(timeout=60)
        volume.commit()


@app.local_entrypoint()
def main(cmd: str = "check", gpu: str = "", args: str = "", hours: float = 2.0):
    t0 = time.time()
    if cmd == "check":
        fn = check
    elif cmd == "prepare":
        fn = prepare
    elif cmd in ("train", "selfplay"):
        fn = train
    else:
        raise SystemExit(f"unknown --cmd {cmd!r} (check | prepare | train | selfplay)")
    if cmd == "selfplay" and not gpu:
        raise SystemExit("--cmd selfplay needs --gpu (e.g. H100): the cheap fallback GPUs are too small for it")
    if hours > 24:
        raise SystemExit("--hours at most 24 (Modal's limit per function call); resume with the same command")
    gpus = gpu.split(",") if gpu else CHEAP_GPUS          # --gpu L40S  or a fallback list: --gpu L40S,A100
    if gpu and len(gpus) > 1:
        raise SystemExit("--gpu takes one GPU type (Modal's with_options has no fallback lists); "
                         "omit --gpu to use the CHEAP_GPUS fallback list")
    opts = {"gpu": gpus[0]} if gpu and cmd != "prepare" else {}
    if cmd in ("train", "selfplay"):
        opts["timeout"] = int(hours * 3600)
        if "--time-limit-hours" not in args:
            args += f" --time-limit-hours {max(hours - 0.15, 0.05):.2f}"   # leave time to save the last checkpoint
        price = max(PRICE_PER_HOUR.get(g.rstrip("!").split(":")[0], float("nan")) for g in gpus)
        print(f"train on {'/'.join(gpus)} for at most {hours:g} h -> at most ~${hours * price:.2f}")
    if opts:
        fn = fn.with_options(**opts)
    if cmd == "check":
        fn.remote()
    elif cmd == "train":
        fn.remote(args, hours)
    elif cmd == "selfplay":
        # spawn and return: with --detach the call keeps running in the cloud, and no local process is left that a
        # shutdown of this machine could take down with it (a SIGTERM to a waiting client cancels the call)
        call = fn.spawn(args, hours, "train_selfplay.py")
        print(f"spawned {call.object_id}; follow it with: uv run modal app logs <app id from 'modal app list'>")
        return
    else:
        fn.remote(args)
    print(f"finished in {(time.time() - t0) / 60:.1f} min; args were: {shlex.split(args)}")
