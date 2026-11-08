#!/bin/bash
# Sets up Human.exe (EklipZ's generals-bot, MIT) in third_party/ for scripts/eval_humanexe.py: a pinned clone, its own
# Python 3.14 environment (the bot relies on 3.14's deferred annotations), its compiled extensions and its config file.
set -e
mkdir -p "$(dirname "$0")/../third_party"
cd "$(dirname "$0")/../third_party"
COMMIT=c27ec40
if [ ! -d generals-bot ]; then
  git clone https://github.com/EklipZgit/generals-bot generals-bot
fi
cd generals-bot
git checkout -q $COMMIT
uv venv .venv --python 3.14
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python setup.py build_ext --inplace > build.log 2>&1
cd ..
mkdir -p humanexe_logs
cat > run_config.txt <<CFG
repo_root=$PWD/generals-bot
log_folder=$PWD/humanexe_logs
grouped_folder=$PWD/humanexe_logs/grouped
bot_logs_root=$PWD/humanexe_logs/bots
using_pypy=False
current_gen_python_path=python3.14
CFG
echo "Human.exe is ready: .venv/bin/python scripts/eval_humanexe.py --run <run> --players 2 --games 10"
