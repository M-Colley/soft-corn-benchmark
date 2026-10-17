"""Synthetic benchmark with a known population target (see ordbench/synthetic.py).

For each (family, seed, n_train, body) cell every method is trained exactly as on real data.
Validation uses only observed sets (uniform-over-set labels), as a practitioner would. The
test set is scored against the exact target t(x) = E[u_S | x] (`nll` - H(t) = KL(t || p_hat))
and against the rater distribution pi(x) (`ho_nll` - H(pi) = KL(pi || p_hat)).

    python scripts/04_synthetic.py --families po cr --seeds 0 1 2 3 4
    python scripts/04_synthetic.py --K 8 --out $SCB_DATA/results/synth_K8           # more levels
    python scripts/04_synthetic.py --mechanism interval --out $SCB_DATA/results/synth_interval
Output: $SCB_DATA/results/synthetic/<family>_m<m>_s<seed>_n<n>_<body>.npz
Generator options (ordbench/synthetic.py): --K, --d, --signal, --nonlinear, --eps, --shift, --mechanism; keep each variant
in its own --out folder. The defaults reproduce the main benchmark.
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
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--d", type=int, default=10)
    ap.add_argument("--signal", type=float, default=1.0, help="scale of the predictors")
    ap.add_argument("--nonlinear", action="store_true", help="predictors with pairwise interactions")
    ap.add_argument("--eps", type=float, default=0.0, help="probability that a rating is uniform noise")
    ap.add_argument("--shift", type=float, nargs=3, default=None, metavar=("DOWN", "STAY", "UP"),
                    help="probabilities that a rating is shifted by -1 / 0 / +1 levels (rater heterogeneity)")
    ap.add_argument("--mechanism", choices=S.MECHANISMS, default="raters")
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

    kw = dict(d=a.d, K=a.K, signal=a.signal, nonlinear=a.nonlinear, eps=a.eps,
              shift=tuple(a.shift) if a.shift else None, mechanism=a.mechanism)
    rated = a.mechanism == "raters"
    m_eff = a.m if rated else 0               # raters per set (known only when sets come from ratings)
    for seed in a.seeds:                       # seeds outermost: partial runs cover every family
        for fam in a.families:
            # one problem (pi(.)) per seed; train / val / test are independent draws from it
            if rated:
                Xp, Mp, _, Rp = S.make(fam, max(a.n_train), a.m, seed=100 * seed + 1, param_seed=seed,
                                       return_ratings=True, **kw)
                Cp = np.eye(a.K)[Rp].sum(1)               # counts of the m ratings behind each training set
            else:
                Xp, Mp, _ = S.make(fam, max(a.n_train), a.m, seed=100 * seed + 1, param_seed=seed, **kw)
                Cp = None
            Xv, Mv, _ = S.make(fam, a.n_val, a.m, seed=100 * seed + 2, param_seed=seed, **kw)
            Xt, Mt, pit, tt = S.make(fam, a.n_test, a.m, seed=100 * seed + 3, param_seed=seed, return_target=True, **kw)
            evalsets = {"val": make_evalset(Xv, Mv, device=dev, m=m_eff),
                        "test": make_evalset(Xt, Mt, p=tt, ho=pit, device=dev, m=m_eff)}
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
                    res, steps = run_cell(Xp[:n], Mp[:n], evalsets, a.K, a.methods, body=body, seed=seed,
                                          steps=a.steps, eval_every=a.eval_every, device=dev,
                                          counts_tr=None if Cp is None else Cp[:n], m=m_eff)
                    meta = dict(family=fam, m=a.m, seed=seed, n_train=n, body=body, K=a.K, d=a.d, signal=a.signal,
                                nonlinear=a.nonlinear, eps=a.eps, shift=a.shift, mechanism=a.mechanism,
                                H_t=entropy(tt), H_pi=entropy(pit), oracle=oracle,
                                configs=CONFIGS, methods=a.methods, prior=prior_metrics(Mp[:n], evalsets))
                    np.savez_compressed(f, steps=steps, meta=json.dumps(meta), **flatten(res))
                    print(f"{f.name}  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
