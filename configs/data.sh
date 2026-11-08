#!/bin/bash
# Human.exe's free-for-all replays: download (generals.io keeps replay files for about a week, so run this
# regularly to collect more) and convert them into one training file, data/processed/humanexe_ffa.npz.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run scripts/download_replays.py Human.exe --out data/raw/humanexe_ffa
uv run scripts/prepare_replays.py data/raw/humanexe_ffa --validate 100
