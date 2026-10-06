"""Synthetic benchmark with a known population target (see ordbench/synthetic.py).

For each (family, seed, n_train, body) cell every method is trained exactly as on real data.
Validation uses only observed sets (uniform-over-set labels), as a practitioner would. The
test set is scored against the exact target t(x) = E[u_S | x] (`nll` - H(t) = KL(t || p_hat))
and against the rater distribution pi(x) (`ho_nll` - H(pi) = KL(pi || p_hat)).

    python scripts/04_synthetic.py --families po cr --seeds 0 1 2 3 4
Output: $SCB_DATA/results/synthetic/<family>_m<m>_s<seed>_n<n>_<body>.npz
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ordbench.paths import RUNS  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ordbench import synthetic as S  # noqa: E402
from ordbench.losses import METHODS  # noqa: E402
from ordbench.metrics import evaluate  # noqa: E402
from ordbench.train import CONFIGS, flatten, make_evalset, prior_metrics, run_cell  # noqa: E402


def entropy(p):
    return float(-(p * np.log(np.clip(p, 1e-12, 1))).sum(1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", default=list(S.FAMILIES))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--n_train", nargs="+", type=int, default=[256, 1024, 4096, 16384])
    ap.add_argument("--bodies", nargs="+", default=["linear", "mlp"])
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--m", type=int, default=3)
    ap.add_argument("--n_val", type=int, default=2000)
    ap.add_argument("--n_test", type=int, default=10000)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--out", default=str(RUNS / "synthetic"))
    a = ap.parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True      # TF32 tensor cores for the GEMMs

    for seed in a.seeds:                       # seeds outermost: partial runs cover every family
        for fam in a.families:
            # one problem (pi(.)) per seed; train / val / test are independent draws from it
            Xp, Mp, _ = S.make(fam, max(a.n_train), a.m, seed=100 * seed + 1, param_seed=seed)
            Xv, Mv, _ = S.make(fam, a.n_val, a.m, seed=100 * seed + 2, param_seed=seed)
            Xt, Mt, pit = S.make(fam, a.n_test, a.m, seed=100 * seed + 3, param_seed=seed)
            tt = S.set_target(pit, a.m)
            evalsets = {"val": make_evalset(Xv, Mv, device=dev),
                        "test": make_evalset(Xt, Mt, p=tt, ho=pit, device=dev)}
            ev = evalsets["test"]
            oracle = {}
            for name, q in (("t", tt), ("pi", pit)):
                lq = torch.log(torch.as_tensor(q, dtype=torch.float32, device=dev).clamp_min(1e-12))[None]
                oracle[name] = {k: float(v[0]) for k, v in evaluate(lq, ev).items()}
            for n in a.n_train:
                for body in a.bodies:
                    f = out_dir / f"{fam}_m{a.m}_s{seed}_n{n}_{body}.npz"
                    if f.exists():
                        continue
                    t0 = time.time()
                    res, steps = run_cell(Xp[:n], Mp[:n], evalsets, 5, a.methods, body=body, seed=seed,
                                          steps=a.steps, eval_every=a.eval_every, device=dev)
                    meta = dict(family=fam, m=a.m, seed=seed, n_train=n, body=body, K=5,
                                H_t=entropy(tt), H_pi=entropy(pit), oracle=oracle,
                                configs=CONFIGS, methods=a.methods, prior=prior_metrics(Mp[:n], evalsets))
                    np.savez_compressed(f, steps=steps, meta=json.dumps(meta), **flatten(res))
                    print(f"{f.name}  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
