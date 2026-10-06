"""Real-data benchmark: for each (dataset, m, seed, n_train, body) cell, train every method
with every hyperparameter configuration and log validation and test metrics over training.

Splits (per seed, over items with at least m ratings): 20% test, 10% validation, the rest is
the training pool. Training sets of increasing size are nested prefixes of the shuffled pool.
Features are standardised with training-set statistics. Held-out annotator counts are passed
to the test set only, so they never influence training or model selection.

    python scripts/03_run.py --datasets cb scut --seeds 0 1 2 3 4
Output: $SCB_DATA/results/real/<dataset>_m<m>_s<seed>_n<n>_<body>.npz (existing cells are skipped)
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ordbench.paths import PROCESSED, RUNS  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ordbench.losses import METHODS  # noqa: E402
from ordbench.train import LRS, WDS, flatten, make_evalset, prior_metrics, run_cell  # noqa: E402

DATASETS = ("cb", "scut", "humicroedit", "mhs", "wikitox")


def load(name, m):
    z = np.load(PROCESSED / f"{name}.npz", allow_pickle=True)
    v = z[f"valid_m{m}"]
    return z["X"][v], z[f"mask_m{m}"][v], z[f"ho_m{m}"][v], int(z["K"])


def split(n, seed, test=0.2, val=0.1):
    perm = np.random.default_rng(seed).permutation(n)
    nt, nv = round(test * n), round(val * n)
    return perm[:nt], perm[nt:nt + nv], perm[nt + nv:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--m", nargs="+", type=int, default=[3])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--n_train", nargs="+", default=["128", "512", "2048", "8192", "full"])
    ap.add_argument("--bodies", nargs="+", default=["linear", "mlp"])
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--wds", nargs="+", type=float, default=None, help="override the L2 grid")
    ap.add_argument("--lrs", nargs="+", type=float, default=None, help="override the learning-rate grid")
    ap.add_argument("--out", default=str(RUNS / "real"))
    a = ap.parse_args()
    configs = [(lr, wd) for lr in (a.lrs or LRS) for wd in (a.wds or WDS)]
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True      # TF32 tensor cores for the GEMMs

    data = {}
    for seed in a.seeds:                       # seeds outermost: partial runs cover every dataset
        for name in a.datasets:
            for m in a.m:
                if (name, m) not in data:
                    data[name, m] = load(name, m)
                X, mask, ho, K = data[name, m]
                te, va, pool = split(len(X), seed)
                sizes = sorted({min(len(pool) if s == "full" else int(s), len(pool)) for s in a.n_train})
                for n in sizes:
                    todo = [b for b in a.bodies if not (out_dir / f"{name}_m{m}_s{seed}_n{n}_{b}.npz").exists()]
                    if not todo:
                        continue
                    tr = pool[:n]
                    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
                    Z = (X - mu) / sd
                    evalsets = {"val": make_evalset(Z[va], mask[va], device=dev),
                                "test": make_evalset(Z[te], mask[te], ho=ho[te], device=dev)}
                    for body in todo:
                        f = out_dir / f"{name}_m{m}_s{seed}_n{n}_{body}.npz"
                        t0 = time.time()
                        res, steps = run_cell(Z[tr], mask[tr], evalsets, K, a.methods, body=body, seed=seed,
                                              steps=a.steps, eval_every=a.eval_every, configs=configs, device=dev)
                        meta = dict(dataset=name, m=m, seed=seed, n_train=n, pool=len(pool), body=body,
                                    K=K, n_val=len(va), n_test=len(te), configs=configs, methods=a.methods,
                                    prior=prior_metrics(mask[tr], evalsets))
                        np.savez_compressed(f, steps=steps, meta=json.dumps(meta), **flatten(res))
                        print(f"{f.name}  {time.time() - t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
