#!/bin/bash
# The clone of Human.exe (behavior cloning, bc.py): the start and teacher of self-play. The defaults of bc.py are
# this run: the full network (15.5M parameters), all of Human.exe's moves incl. its waiting in the opening,
# 8,000 steps of batch 128. About an hour on a 16 GB GPU (0.4 s per step); afterwards the accuracy by kind of move.
set -euo pipefail
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80 JAX_COMPILATION_CACHE_DIR=$HOME/.cache/jax_cc
uv run scripts/train_bc.py --data data/processed/humanexe_ffa.npz --out runs/clone "$@" 2>&1 | tee -a runs/clone.log
uv run scripts/bc_breakdown.py --run runs/clone --positions 3000
