"""Top-1 / top-5 accuracy of a behaviour-cloning checkpoint on its own validation games, split by the kind of move:
continuing the previous move's path, starting a new path, or passing.

    uv run scripts/bc_breakdown.py --run runs/clone --positions 3000

The validation games are the ones the run held out (same seed and split as bc.train).
"""
import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np

from generals_rl.bc import Config, Dataset, make_batch, sample_positions
from generals_rl.model import masked_logits
from generals_rl.play import load_policy


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--ckpt", default="latest")
    p.add_argument("--positions", type=int, default=1500)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="", help="append the result as a JSON line to this file")
    args = p.parse_args()

    apply_fn, params, rc = load_policy(args.run, args.ckpt)
    cfg = Config(**{k: v for k, v in rc.items() if k in Config.__dataclass_fields__})
    ds = Dataset(cfg.data, cfg)
    perm = np.random.default_rng(cfg.seed).permutation(np.nonzero(ds.eligible.any(1))[0])
    val = perm[: max(1, int(len(perm) * cfg.val_frac))]
    forward = jax.jit(lambda o: masked_logits(apply_fn(params, o)[0], o["action_mask"]))
    n_players = jnp.asarray(ds.d["n_players"])

    rng = np.random.default_rng(args.seed)
    kinds, top1, top5, p_target = [], [], [], []
    per_chunk = args.positions // -(-len(val) // 8)
    for i in range(0, len(val), 8):
        games = val[i:i + 8]
        sim, legal, elim, actions, place, n_ticks = ds.chunk(games)
        idx = sample_positions(rng, ds, games, legal, elim, actions, n_ticks, cfg, per_chunk)
        if idx is None:
            continue
        b = make_batch(sim, actions, place, n_players[games], *idx)
        passed = np.asarray(actions[idx[0], idx[1], idx[2], 0]) == 1
        for s in range(0, len(idx[0]), 64):
            obs = jax.tree.map(lambda x: x[s:s + 64], b["obs"])
            logits = forward(obs)
            target = np.asarray(b["target"][s:s + 64])
            best = np.asarray(jnp.argsort(-logits, -1)[:, :5])
            prob = np.asarray(jax.nn.softmax(logits, -1))
            new = np.asarray(b["new_path"][s:s + 64])
            kinds += ["pass" if pa else "new" if nw else "continue" for pa, nw in zip(passed[s:s + 64], new)]
            top1 += list(best[:, 0] == target)
            top5 += list((best == target[:, None]).any(1))
            p_target += list(prob[np.arange(len(target)), target])

    kinds, top1, top5, p_target = map(np.asarray, (kinds, top1, top5, p_target))
    result = dict(run=args.run, ckpt=args.ckpt, positions=len(kinds))
    for kind in ("continue", "new", "pass", "all"):
        sel = np.ones(len(kinds), bool) if kind == "all" else kinds == kind
        if not sel.any():
            continue
        result[kind] = dict(share=float(sel.mean()), top1=float(top1[sel].mean()), top5=float(top5[sel].mean()),
                            p_target=float(p_target[sel].mean()))
        print(f"{kind:9s} share {sel.mean():.3f}  top1 {top1[sel].mean():.3f}  top5 {top5[sel].mean():.3f}  "
              f"p(target) {p_target[sel].mean():.3f}")
    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
