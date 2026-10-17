"""Distribution shift (FUTURE_WORK item 11): train on all but one group of items and test on the held-out group.

Groups (ordbench/groups.py): MHS platforms, Wikipedia Toxicity namespaces and years, SCUT demographic subsets. Everything
else follows scripts/03_run.py: frozen features, 12 methods, linear heads, the same grid and protocol. Checkpoints are
selected on an in-distribution validation split, as a practitioner would, and read on the out-of-distribution test
set (`test`) and, for reference, on an in-distribution test split (`idtest`).

    python scripts/12_shift.py --datasets mhs wikitox scut
Output: $SCB_DATA/results/shift/<dataset>-<scheme>-<group>_m3_s<seed>_n<n>_linear.npz (finished cells are skipped)
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ordbench.paths import PROCESSED, RUNS  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ordbench.groups import SCHEMES, item_groups, leave_one_group_out  # noqa: E402
from ordbench.losses import METHODS  # noqa: E402
from ordbench.train import CONFIGS, flatten, make_evalset, prior_metrics, run_cell  # noqa: E402

_spec = importlib.util.spec_from_file_location("run03", Path(__file__).with_name("03_run.py"))
run03 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run03)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(SCHEMES))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--n_train", nargs="+", default=["2048", "full"])
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--m", type=int, default=3)
    ap.add_argument("--out", default=str(RUNS / "shift"))
    a = ap.parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    m = a.m
    for seed in a.seeds:
        for name in a.datasets:
            X, mask, ho, K, cnt = run03.load(name, m)
            z = np.load(PROCESSED / f"{name}.npz", allow_pickle=True)
            item_id = z["item_id"][z[f"valid_m{m}"]]
            for scheme, held_outs in SCHEMES[name].items():
                groups = item_groups(name, scheme, item_id)
                for g in held_outs:
                    pool, va, idte, ood = leave_one_group_out(groups, g, seed)
                    sizes = sorted({min(len(pool) if s == "full" else int(s), len(pool)) for s in a.n_train})
                    for n in sizes:
                        f = out_dir / f"{name}-{scheme}-{g}_m{m}_s{seed}_n{n}_linear.npz"
                        if f.exists():
                            continue
                        t0 = time.time()
                        tr = pool[:n]
                        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
                        Z = (X - mu) / sd
                        evalsets = {"val": make_evalset(Z[va], mask[va], device=dev, m=m),
                                    "test": make_evalset(Z[ood], mask[ood], ho=ho[ood], device=dev, m=m),
                                    "idtest": make_evalset(Z[idte], mask[idte], ho=ho[idte], device=dev, m=m)}
                        res, steps = run_cell(Z[tr], mask[tr], evalsets, K, a.methods, body="linear", seed=seed,
                                              device=dev, steps=2000, eval_every=100, counts_tr=cnt[tr], m=m)
                        meta = dict(dataset=f"{name}/{scheme}/{g}", m=m, seed=seed, n_train=n, pool=len(pool), body="linear",
                                    K=K, n_val=len(va), n_test=len(ood), n_idtest=len(idte), configs=CONFIGS,
                                    methods=a.methods, prior=prior_metrics(mask[tr], {k: evalsets[k] for k in ("val", "test")}))
                        np.savez_compressed(f, steps=steps, meta=json.dumps(meta), **flatten(res))
                        print(f"{f.name}  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
