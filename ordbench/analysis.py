"""Offline model selection and tidy result tables.

Each run file holds, per method, validation and test metrics for every (evaluation step,
hyperparameter configuration). `select` picks the (step, configuration) that minimises a
validation criterion and reads the test metrics there, so early stopping and hyperparameter
choice use validation data only and every selection rule is applied to identical runs.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

# validation criteria: (primary, tie-breaker). set-MAE is discrete, so ties are broken by NLL.
CRITERIA = {"set_mae": ("set_mae_med", "nll"), "nll": ("nll", "set_mae_med"), "rps": ("rps", "nll")}


def _argmin(primary, secondary):
    a = np.nan_to_num(primary, nan=np.inf).ravel()
    b = np.nan_to_num(secondary, nan=np.inf).ravel()
    return np.lexsort((np.arange(a.size), b, a))[0]


def select(path, criteria=CRITERIA):
    """Rows: one per (method, criterion) with the run's metadata, the chosen step and config,
    and every test metric at that point."""
    with np.load(path, allow_pickle=False) as f:
        z = {k: f[k] for k in f.files}          # decompress every array once
    meta = json.loads(str(z["meta"]))
    steps = z["steps"]
    configs = meta.pop("configs")
    methods = meta.pop("methods")
    oracle = meta.pop("oracle", None)
    prior = meta.pop("prior", None)
    test_keys = sorted({k.split("/", 2)[2] for k in z if k.count("/") == 2 and "/test/" in k})
    rows = []
    for method in methods:
        for crit, (prim, sec) in criteria.items():
            i = _argmin(z[f"{method}/val/{prim}"], z[f"{method}/val/{sec}"])
            e, c = np.unravel_index(i, z[f"{method}/val/{prim}"].shape)
            row = dict(meta, method=method, select_by=crit, step=int(steps[e]),
                       lr=configs[c][0], wd=configs[c][1],
                       val_score=float(z[f"{method}/val/{prim}"][e, c]))
            for k in test_keys:
                row[k] = float(z[f"{method}/test/{k}"][e, c])
            if prior is not None:
                row["prior_nll"] = prior["test"]["nll"]
                row["prior_ho_nll"] = prior["test"].get("ho_nll", np.nan)
                row["prior_set_mae_med"] = prior["test"]["set_mae_med"]
            if oracle is not None:
                row["kl_t"] = row["nll"] - meta["H_t"]
                row["kl_pi"] = row["ho_nll"] - meta["H_pi"]
            rows.append(row)
    return rows


def collect(run_dir, pattern="*.npz"):
    rows = []
    for f in sorted(Path(run_dir).glob(pattern)):
        rows.extend(select(f))
    return pd.DataFrame(rows)


def corrected_ttest(diff, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled t-test for J paired differences from
    repeated random splits. Returns (mean, t, two-sided p)."""
    from scipy import stats
    diff = np.asarray(diff, float)
    J = len(diff)
    var = diff.var(ddof=1)
    if J < 2 or var == 0:
        return diff.mean(), np.nan, np.nan
    t = diff.mean() / np.sqrt((1 / J + n_test / n_train) * var)
    return diff.mean(), t, 2 * stats.t.sf(abs(t), J - 1)
