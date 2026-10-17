"""The thesis's personalisation protocol on the ProVoice study: one cell per (test driver, seed, K, head).

Leave one driver out; the held-out driver and two validation drivers contribute only K calibration answers to training,
the other nine drivers all of theirs (ordbench/kcurve.py). Every method is trained with the whole hyper-parameter grid and
selected on the validation drivers' remaining answers, exactly as in the main benchmark. Besides the twelve core losses a
cell stores the population prior and a personal prior (the driver's calibration labels smoothed towards the population
marginal, smoothing chosen on the validation drivers).

    python scripts/15_kcurve.py --features task all --seeds 0 1 2
Output: $SCB_DATA/results/kcurve_<features>/<features>_d<driver>_s<seed>_k<K>_<body>.npz (existing cells are skipped)
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

from ordbench import kcurve  # noqa: E402
from ordbench.losses import METHODS  # noqa: E402
from ordbench.metrics import evaluate  # noqa: E402
from ordbench.train import CONFIGS, flatten, make_evalset, prior_metrics, run_cell, target_from_mask  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="+", default=["task", "all"], choices=["task", "all"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--ks", nargs="+", type=int, default=list(kcurve.KS))
    ap.add_argument("--bodies", nargs="+", default=["linear", "mlp"])
    ap.add_argument("--drivers", nargs="+", default=None, help="test drivers (default: all twelve)")
    ap.add_argument("--methods", nargs="+", default=METHODS, help="core twelve by default")
    ap.add_argument("--collapse", default="none", choices=["none", "lower", "upper", "random"],
                    help="train on one level of every set (a control without the set information); evaluation keeps the sets")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--out", default=str(RUNS / "kcurve"), help="prefix: results go to <out>_<features>")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True

    z = np.load(PROCESSED / "provoice.npz", allow_pickle=True)
    mask, drivers = z["mask_m0"], z["group"]
    windows = np.array([str(i).rsplit("p", 1)[0] for i in z["item_id"]])
    soft = target_from_mask(mask).p.numpy().astype(np.float64)
    K = int(z["K"])
    users = sorted(set(drivers))
    test_drivers = a.drivers or users
    design = {f: kcurve.feature_matrix(z["X"], z["feature_names"], drivers, f) for f in a.features}

    for seed in a.seeds:                              # seeds outermost: a partial run covers every driver
        for test_driver in test_drivers:
            for k in a.ks:
                train, val, test, cal, val_drivers, personal = kcurve.kshot_split(drivers, windows, test_driver, seed, k)
                pop = soft[train].mean(0)
                for feat in a.features:
                    X, code, _ = design[feat]
                    out_dir = Path(f"{a.out}_{feat}" + ("" if a.collapse == "none" else f"_collapse_{a.collapse}"))
                    out_dir.mkdir(parents=True, exist_ok=True)
                    todo = [b for b in a.bodies if not (out_dir / f"{feat}_d{test_driver}_s{seed}_k{k}_{b}.npz").exists()]
                    if not todo:
                        continue
                    mu, sd = X[train].mean(0), X[train].std(0) + 1e-6
                    C = kcurve.driver_code(code, drivers, personal, k)
                    Z = np.concatenate([(X - mu) / sd, C], 1)
                    evalsets = {"val": make_evalset(Z[val], mask[val], device=dev), "test": make_evalset(Z[test], mask[test], device=dev)}
                    mask_train = mask[train] if a.collapse == "none" else kcurve.collapse_sets(mask[train], a.collapse, seed)
                    prior = prior_metrics(mask_train, evalsets)
                    best = None
                    for alpha in kcurve.ALPHAS:       # personal prior: smoothing chosen on the validation drivers
                        lp = torch.as_tensor(kcurve.personal_prior(soft, drivers, val, cal, pop, alpha), dtype=torch.float32, device=dev)
                        score = float(evaluate(lp, evalsets["val"])["nll"][0])
                        if best is None or score < best[0]:
                            best = (score, alpha)
                    lp = torch.as_tensor(kcurve.personal_prior(soft, drivers, test, cal, pop, best[1]), dtype=torch.float32, device=dev)
                    pp = {m_: float(v[0]) for m_, v in evaluate(lp, evalsets["test"]).items()}
                    for body in todo:
                        f = out_dir / f"{feat}_d{test_driver}_s{seed}_k{k}_{body}.npz"
                        t0 = time.time()
                        res, steps = run_cell(Z[train], mask_train, evalsets, K, a.methods, body=body, seed=seed, steps=a.steps,
                                              eval_every=a.eval_every, configs=CONFIGS, device=dev, m=0)
                        meta = dict(dataset="provoice", features=feat, driver=test_driver, seed=seed, n_cal=k, body=body, K=K,
                                    n_train=len(train), n_val=len(val), n_test=len(test), val_drivers=val_drivers,
                                    configs=CONFIGS, methods=a.methods, prior=prior, collapse=a.collapse,
                                    pp_alpha=best[1], pp_nll=pp["nll"], pp_pll=pp["pll"], pp_set_mae_med=pp["set_mae_med"])
                        np.savez_compressed(f, steps=steps, meta=json.dumps(meta), **flatten(res))
                        print(f"{f.name}  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
