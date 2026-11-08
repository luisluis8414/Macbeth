"""Behavior cloning: a clone of Human.exe from its FFA replays, the start of self-play. All options and the data
selection rules: uv run scripts/train_bc.py --help

    uv run scripts/train_bc.py --data data/processed/humanexe_ffa.npz --out runs/clone
"""
from macbeth.bc import main

if __name__ == "__main__":
    main()
