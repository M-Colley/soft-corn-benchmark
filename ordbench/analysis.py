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
# xnll (exact set likelihood, needs the number of raters per set) is target-agnostic: it can compare models trained on
# different soft targets, which NLL against the uniform-over-set label cannot
CRITERIA = {"set_mae": ("set_mae_med", "nll"), "nll": ("nll", "set_mae_med"), "rps": ("rps", "nll"),
            "xnll": ("xnll", "nll")}


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
    # further evaluation splits logged next to val / test (for example the in-distribution test set of a shift run)
    extra = sorted({k.split("/")[1] for k in z if k.count("/") == 2} - {"val", "test", "train"})
    rows = []
    for method in methods:
        for crit, (prim, sec) in criteria.items():
            if f"{method}/val/{prim}" not in z:          # e.g. xnll in runs made without the number of raters
                continue
            i = _argmin(z[f"{method}/val/{prim}"], z[f"{method}/val/{sec}"])
            e, c = np.unravel_index(i, z[f"{method}/val/{prim}"].shape)
            row = dict(meta, method=method, select_by=crit, step=int(steps[e]),
                       lr=configs[c][0], wd=configs[c][1],
                       val_score=float(z[f"{method}/val/{prim}"][e, c]))
            for k in test_keys:
                row[k] = float(z[f"{method}/test/{k}"][e, c])
            for split in extra:
                for k in sorted({k.split("/", 2)[2] for k in z if k.startswith(f"{method}/{split}/")}):
                    row[f"{split}_{k}"] = float(z[f"{method}/{split}/{k}"][e, c])
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


def pairwise_table(df, keys, metric, crit, baselines, proposed="soft_corn", rope=0.0, tcols=("n_train", "n_test")):
    """`proposed` minus each baseline on `metric` (lower is better) at the checkpoint selected by `crit`.
    Per problem (rows of `keys`): mean difference over seeds and the corrected resampled t-test (alpha 0.05).
    Across problems: win/tie/loss counts and the Bayesian signed-rank test (Benavoli et al., 2017) with a region of
    practical equivalence `rope`. baycomp assumes higher is better, so its probabilities are flipped for the
    lower-is-better metrics used here: p_better is the probability that `proposed` has the lower value."""
    import baycomp
    d = df[df.select_by == crit]
    rows = []
    for b in baselines:
        per = []
        for key, g in d.groupby(keys):
            a = g[g.method == proposed].set_index("seed")[metric]
            c = g[g.method == b].set_index("seed")[metric]
            j = a.index.intersection(c.index)
            if len(j) < 2:
                continue
            n_tr = g[tcols[0]].iloc[0]
            n_te = g[tcols[1]].iloc[0] if tcols[1] in g else n_tr
            mean, t, p = corrected_ttest((a[j] - c[j]).values, n_tr, n_te)
            per.append(dict(zip(keys, key if isinstance(key, tuple) else (key,)), diff=mean, p=p,
                            a=a[j].mean(), b=c[j].mean()))
        per = pd.DataFrame(per)
        if per.empty:
            continue
        win = ((per["diff"] < 0) & (per.p < 0.05)).sum()
        loss = ((per["diff"] > 0) & (per.p < 0.05)).sum()
        try:
            np.random.seed(0)  # baycomp samples from numpy's global generator: fix it so the tables reproduce exactly
            pl, pr, pg = baycomp.SignedRankTest.probs(per.a.values, per.b.values, rope=rope)
        except Exception:
            pl = pr = pg = np.nan
        rows.append(dict(baseline=b, problems=len(per), mean_diff=per["diff"].mean(),
                         se_diff=per["diff"].std(ddof=1) / np.sqrt(len(per)),
                         median_diff=per["diff"].median(), wins=win, ties=len(per) - win - loss, losses=loss,
                         p_better=pg, p_rope=pr, p_worse=pl))
    cols = ["baseline", "problems", "mean_diff", "se_diff", "median_diff", "wins", "ties", "losses",
            "p_better", "p_rope", "p_worse"]
    return pd.DataFrame(rows, columns=cols)


def random_effects(d, v, rope=0.0):
    """Random-effects meta-analysis of per-problem differences d_i with sampling variances v_i (for example the
    Nadeau-Bengio corrected variance of a problem's seed differences): d_i ~ N(mu, tau^2 + v_i), tau^2 by the
    DerSimonian-Laird estimator. Returns mu, its standard error, tau, and, from the normal posterior of mu
    under a flat prior, the probabilities that mu is below -rope, within the rope and above +rope.
    A transparent stand-in for the Stan-based hierarchical test of Corani et al. (2017), which needs PyStan."""
    from scipy import stats
    d, v = np.asarray(d, float), np.asarray(v, float)
    w = 1 / v
    mu_fe = (w * d).sum() / w.sum()
    q = (w * (d - mu_fe) ** 2).sum()
    c = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0.0, (q - (len(d) - 1)) / c) if c > 0 else 0.0
    w_re = 1 / (v + tau2)
    mu = (w_re * d).sum() / w_re.sum()
    se = float(np.sqrt(1 / w_re.sum()))
    below, above = stats.norm.cdf(-rope, mu, se), stats.norm.sf(rope, mu, se)
    return dict(mu=float(mu), se=se, tau=float(np.sqrt(tau2)), p_below=float(below), p_rope=float(1 - below - above),
                p_above=float(above), problems=len(d))
