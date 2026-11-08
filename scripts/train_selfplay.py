"""Self-play PPO for 8-player FFA, started from the clone. All options: uv run scripts/train_selfplay.py --help

    uv run scripts/train_selfplay.py --out runs/selfplay --init runs/clone/params_latest.msgpack \\
        --anchor runs/clone/params_latest.msgpack

The defaults are the setup of the H100 run (configs/selfplay_modal.sh); on a 16 GB GPU use --num-envs 16
--micro-batch 64. A run continues from runs/<name>/resume.msgpack if that file exists.
"""
from macbeth.selfplay import main

if __name__ == "__main__":
    main()
