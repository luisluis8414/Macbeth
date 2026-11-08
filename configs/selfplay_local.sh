#!/bin/bash
# Self-play from the clone on a local 16 GB GPU (selfplay.py): the setup of configs/selfplay_modal.sh with a batch
# that fits: 16 games per iteration instead of 64. Roughly 4-6 times slower per sample than an H100.
# Started again, it resumes from runs/selfplay/resume.msgpack.
set -euo pipefail
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80 JAX_COMPILATION_CACHE_DIR=$HOME/.cache/jax_cc
CLONE=runs/clone/params_latest.msgpack
uv run scripts/train_selfplay.py --out runs/selfplay --init $CLONE --anchor $CLONE \
  --num-envs 16 --micro-batch 64 --critic-minibatch 512 "$@" 2>&1 | tee -a runs/selfplay.log
