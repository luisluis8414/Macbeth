#!/bin/bash
# Self-play from the clone on a Modal H100 (selfplay.py defaults: 64 games x 96 ticks x 8 seats per iteration,
# minibatch 1024 in chunks of 256, about 10 s per iteration after 15-30 minutes of compiling the first time).
# Hours (default 6, at most 24 per call) cap the cost at hours x $3.95. The launcher hands the run to Modal and
# returns, so this machine can be switched off; started again, the run resumes from its last checkpoint.
#
#   configs/selfplay_modal.sh [hours]
#   uv run modal app logs <app id>                                   # follow it (uv run modal app list)
#   uv run modal volume get generals-rl /runs/selfplay runs/         # fetch the results
set -euo pipefail
cd "$(dirname "$0")/.."
HOURS=${1:-6}
uv run modal volume put --force generals-rl runs/clone/params_latest.msgpack /runs/clone/params_latest.msgpack
CLONE=runs/clone/params_latest.msgpack
uv run modal run --detach scripts/modal_app.py --cmd selfplay --gpu H100 --hours "$HOURS" \
  --args "--out runs/selfplay --init $CLONE --anchor $CLONE"
