"""Per-item analyses from the replayed predictions (scripts/08_replay.py).

1. Item-level paired bootstrap: soft-CORN minus each rival on the per-item NLL against the held-out raters,
   resampling test items within each seed (the corrected resampled t-test of 05_analyze.py uses only seed means).
2. Conformal prediction sets (split conformal, ordinal APS): the validation items calibrate a score threshold for
   the rating of a new rater; the set is an interval grown from the mode of the predicted PMF. Reported: coverage
   of the held-out raters' ratings, set size, and recall of the observed set.
3. Reliability and sharpness of P(y > k), and post-hoc temperature scaling per method (fitted on the validation
   sets), to separate discrimination from calibration.

    python scripts/09_setpred.py
Writes results/item_level_ci.csv, conformal.csv, temperature_scaling.csv, reliability.csv, setpred.md, figures/.
"""
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ordbench.paths import FIGURES, RUNS, TABLES  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from ordbench.losses import METHODS  # noqa: E402
from ordbench.metrics import evaluate  # noqa: E402
from ordbench.train import make_evalset  # noqa: E402

warnings.filterwarnings("ignore")
PROPOSED = "soft_corn"
LABEL = dict(zip(METHODS, ["CE (hard)", "CORN (hard)", "CORAL (hard)", "soft CE", "soft-CORAL", "soft-OR", "OLDL-CAD",
                           "OLDL-QFD", "OLDL-CJS", "PLL-softmax", "PLL-CORN", "soft-CORN"]))
RIVALS = ["soft_coral", "ce_soft", "soft_or", "oldl_qfd", "pll_corn", "corn_hard"]
SHOWN = ["soft_corn", "soft_coral", "ce_soft", "soft_or", "oldl_qfd", "pll_corn", "corn_hard"]
COLOR = dict(zip(SHOWN, ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#eda100", "#e87ba4", "#008300"]))
ALPHAS = (0.2, 0.1)
REPLAY = RUNS / "replay"


def load_cells(body="linear"):
    cells = []
    for f in sorted(REPLAY.glob(f"*_{body}.npz")):
        with np.load(f, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        meta = json.loads(str(d["meta"]))
        d["meta"] = meta
        cells.append(d)
    return cells


# --------------------------------------------------------------------------------------------
# 1. item-level paired bootstrap
# --------------------------------------------------------------------------------------------

def item_nll(cell, method):
    """Per-item NLL against the held-out raters (items that have any) and against the uniform-over-set label."""
    lp = np.log(np.clip(np.exp(cell[f"{method}/test"].astype(np.float64)), 1e-12, 1))
    ho = cell["test_ho"].astype(np.float64)
    has = ho.sum(1) > 0
    w = ho[has] / ho[has].sum(1, keepdims=True)
    mask = cell["test_mask"]
    p = mask / mask.sum(1, keepdims=True)
    return -(w * lp[has]).sum(1), -(p * lp).sum(1), has


def item_level(cells, B=500, seed=0):
    rng = np.random.default_rng(seed)
    probs = {}
    for c in cells:
        m = c["meta"]
        probs.setdefault((m["dataset"], m["n_train"]), []).append(c)
    rows = []
    for (ds, n), group in sorted(probs.items()):
        for target in ("ho_nll", "nll"):
            boots = {b: np.zeros(B) for b in RIVALS}
            means = {b: [] for b in RIVALS}
            for c in group:
                ho_i, nll_i, has = item_nll(c, PROPOSED)
                ref = ho_i if target == "ho_nll" else nll_i
                idx = rng.integers(0, len(ref), size=(B, len(ref)))
                for b in RIVALS:
                    other = item_nll(c, b)[0 if target == "ho_nll" else 1]
                    d = ref - other
                    means[b].append(d.mean())
                    boots[b] += d[idx].mean(1) / len(group)
            for b in RIVALS:
                lo, hi = np.percentile(boots[b], [2.5, 97.5])
                rows.append(dict(dataset=ds, n_train=n, metric=target, baseline=b, seeds=len(group),
                                 diff=float(np.mean(means[b])), ci_lo=float(lo), ci_hi=float(hi),
                                 verdict="win" if hi < 0 else "loss" if lo > 0 else "tie"))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# 2. conformal prediction sets
# --------------------------------------------------------------------------------------------

def aps_scores(pmf):
    """(N, K) score of every level: the cumulative mass of the interval grown greedily from the mode (adding
    the more probable neighbour) at the moment the level is added; the mode scores its own probability."""
    N, K = pmf.shape
    rows = np.arange(N)
    a = pmf.argmax(1)
    b = a.copy()
    cum = pmf[rows, a].copy()
    score = np.zeros((N, K))
    score[rows, a] = cum
    for _ in range(K - 1):
        left = np.where(a > 0, pmf[rows, np.maximum(a - 1, 0)], -1.0)
        right = np.where(b < K - 1, pmf[rows, np.minimum(b + 1, K - 1)], -1.0)
        active = (left >= 0) | (right >= 0)
        go_left = left >= right
        new = np.where(go_left, a - 1, b + 1)
        add = np.where(go_left, left, right)
        cum = np.where(active, cum + add, cum)
        score[rows[active], new[active]] = cum[active]
        a = np.where(active & go_left, a - 1, a)
        b = np.where(active & ~go_left, b + 1, b)
    return score


def conformal(cells, repeats=5, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for c in cells:
        m = c["meta"]
        K = c["test_mask"].shape[1]
        ho = c["test_ho"].astype(np.float64)
        has = ho.sum(1) > 0
        for method in m["methods"]:
            if f"{method}/test" not in c:
                continue
            sv = aps_scores(np.exp(c[f"{method}/val"].astype(np.float64)))
            st = aps_scores(np.exp(c[f"{method}/test"].astype(np.float64)))
            cnt = c["val_counts"].astype(np.float64)
            for alpha in ALPHAS:
                cov, size, rec = [], [], []
                for _ in range(repeats):
                    # one randomly chosen rating per validation item calibrates the threshold
                    u = rng.random(len(cnt))[:, None]
                    k = (np.cumsum(cnt / cnt.sum(1, keepdims=True), 1) < u).sum(1).clip(0, K - 1)
                    cal = sv[np.arange(len(sv)), k]
                    q = np.quantile(cal, min(1.0, np.ceil((len(cal) + 1) * (1 - alpha)) / len(cal)), method="higher")
                    inside = st <= q + 1e-12
                    cov.append(((ho[has] * inside[has]).sum(1) / ho[has].sum(1)).mean())
                    size.append(inside.sum(1).mean())
                    mk = c["test_mask"]
                    rec.append(((mk & inside).sum(1) / mk.sum(1)).mean())
                rows.append(dict(dataset=m["dataset"], n_train=m["n_train"], seed=m["seed"], method=method, alpha=alpha,
                                 coverage=np.mean(cov), size=np.mean(size), recall=np.mean(rec)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# 3. reliability, sharpness, temperature scaling
# --------------------------------------------------------------------------------------------

def reliability(cells, bins=10):
    acc = {}
    for c in cells:
        m = c["meta"]
        if m["n_train"] != m["pool"]:
            continue
        ho = c["test_ho"].astype(np.float64)
        has = ho.sum(1) > 0
        w = ho[has] / ho[has].sum(1, keepdims=True)
        share_above = 1 - np.cumsum(w, 1)[:, :-1]
        for method in m["methods"]:
            if f"{method}/test" not in c:
                continue
            pmf = np.exp(c[f"{method}/test"].astype(np.float64))[has]
            g = 1 - np.cumsum(pmf, 1)[:, :-1]
            b = np.minimum((g * bins).astype(int), bins - 1)
            d = acc.setdefault(method, dict(n=np.zeros(bins), g=np.zeros(bins), r=np.zeros(bins), ent=[], conf=[]))
            for j in range(bins):
                sel = b == j
                d["n"][j] += sel.sum()
                d["g"][j] += g[sel].sum()
                d["r"][j] += share_above[sel].sum()
            d["ent"].append(float(-(pmf * np.log(np.clip(pmf, 1e-12, 1))).sum(1).mean()))
            d["conf"].append(float(pmf.max(1).mean()))
    rows = []
    for method, d in acc.items():
        for j in range(bins):
            if d["n"][j] > 0:
                rows.append(dict(method=method, bin=j, n=d["n"][j], predicted=d["g"][j] / d["n"][j],
                                 observed=d["r"][j] / d["n"][j], entropy=np.mean(d["ent"]), max_prob=np.mean(d["conf"])))
    return pd.DataFrame(rows)


def temperature_scaling(cells, grid=np.exp(np.linspace(np.log(0.4), np.log(4.0), 41))):
    rows = []
    for c in cells:
        m = c["meta"]
        te = make_evalset(np.zeros((len(c["test_mask"]), 1)), c["test_mask"], ho=c["test_ho"])
        vmask = c["val_mask"]
        pv = vmask / vmask.sum(1, keepdims=True)
        for method in m["methods"]:
            if f"{method}/test" not in c:
                continue
            lv = torch.as_tensor(c[f"{method}/val"], dtype=torch.float64)
            lt = torch.as_tensor(c[f"{method}/test"], dtype=torch.float64)
            nll = [-(pv * torch.log_softmax(lv / T, -1).numpy()).sum(1).mean() for T in grid]
            T = float(grid[int(np.argmin(nll))])
            raw = {k: float(v[0]) for k, v in evaluate(lt[None].float(), te).items()}
            ts = {k: float(v[0]) for k, v in evaluate(torch.log_softmax(lt / T, -1)[None].float(), te).items()}
            rows.append(dict(dataset=m["dataset"], n_train=m["n_train"], seed=m["seed"], method=method, T=T,
                             **{f"raw_{k}": raw[k] for k in ("ho_nll", "nll", "ho_ece", "int80_iscore")},
                             **{f"ts_{k}": ts[k] for k in ("ho_nll", "nll", "ho_ece", "int80_iscore")}))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------

def md(df, fmt=".3f"):
    return df.to_markdown(floatfmt=fmt)


def baycomp_probs(a, b, rope):
    import baycomp
    try:
        np.random.seed(0)  # fixed so the tables reproduce exactly
        pl, pr, pg = baycomp.SignedRankTest.probs(np.asarray(a), np.asarray(b), rope=rope)
        return pg, pr, pl            # P(first lower = better), P(rope), P(first higher)
    except Exception:
        return np.nan, np.nan, np.nan


def main():
    global REPLAY
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", default=str(REPLAY))
    ap.add_argument("--body", default="linear")
    REPLAY = Path(ap.parse_args().replay)
    FIGURES.mkdir(exist_ok=True)
    cells = load_cells("linear")
    if not cells:
        print("no replayed cells in", REPLAY)
        return
    lines = ["# Per-item analyses of the replayed predictions (auto-generated by scripts/09_setpred.py)\n",
             f"{len(cells)} linear-head cells; deviation of replayed from logged test NLL: "
             f"{max(float(c['max_nll_deviation']) for c in cells):.1e}.\n"]

    # 1 -------------------------------------------------------------------------------------------
    il = item_level(cells)
    il.to_csv(TABLES / "item_level_ci.csv", index=False)
    summ = il.groupby(["metric", "baseline"]).agg(problems=("verdict", "size"), mean_diff=("diff", "mean"),
                                                  wins=("verdict", lambda s: (s == "win").sum()),
                                                  ties=("verdict", lambda s: (s == "tie").sum()),
                                                  losses=("verdict", lambda s: (s == "loss").sum())).reset_index()
    summ["baseline"] = summ.baseline.map(LABEL)
    lines += ["## Item-level paired bootstrap, soft-CORN minus rival (< 0 favours soft-CORN)\n",
              "Per problem (dataset, training size): 500 resamples of the test items within each seed, 95% percentile "
              "interval of the seed-averaged difference; win/tie/loss = interval below / around / above zero.\n",
              summ.to_markdown(index=False, floatfmt=".4f"), ""]

    # 2 -------------------------------------------------------------------------------------------
    cf = conformal(cells)
    cf.to_csv(TABLES / "conformal.csv", index=False)
    full = cf.merge(pd.DataFrame([dict(dataset=c["meta"]["dataset"], n_train=c["meta"]["n_train"], pool=c["meta"]["pool"])
                                  for c in cells]).drop_duplicates(), on=["dataset", "n_train"])
    for tag, d in (("all sizes", full), ("full pool", full[full.n_train == full.pool])):
        per = d.groupby(["alpha", "method"])[["coverage", "size", "recall"]].mean()
        t = per.reset_index()
        t["method"] = t.method.map(LABEL)
        lines += [f"## Conformal prediction sets, {tag} (mean over problems and seeds; target coverage 1 - alpha)\n",
                  "Split conformal with the ordinal APS score; coverage is the share of the held-out raters' ratings "
                  "inside the set, size the number of levels, recall the share of the observed set inside the set.\n",
                  t.to_markdown(index=False, floatfmt=".3f"), ""]
    rows = []
    for alpha in ALPHAS:
        d = full[full.alpha == alpha].groupby(["dataset", "n_train", "method"])["size"].mean().unstack()
        for b in RIVALS:
            pg, pr, pl = baycomp_probs(d[PROPOSED], d[b], 0.05)
            rows.append(dict(alpha=alpha, baseline=LABEL[b], problems=len(d), mean_diff_size=float((d[PROPOSED] - d[b]).mean()),
                             p_smaller=pg, p_rope=pr, p_larger=pl))
    lines += ["### Set size, soft-CORN minus rival (< 0: smaller sets at the same nominal coverage); Bayesian signed-rank, "
              "ROPE 0.05 levels\n", pd.DataFrame(rows).to_markdown(index=False, floatfmt=".3f"), ""]
    fig, axes = plt.subplots(1, len(ALPHAS), figsize=(3.4 * len(ALPHAS), 3.0), squeeze=False)
    for ax, alpha in zip(axes[0], ALPHAS):
        d = full[(full.alpha == alpha) & (full.n_train == full.pool)].groupby("method")[["coverage", "size"]].mean()
        for m in SHOWN:
            ax.scatter(d.loc[m, "size"], d.loc[m, "coverage"], color=COLOR[m], s=45 if m == PROPOSED else 28, label=LABEL[m])
        ax.axhline(1 - alpha, color="#52514e", lw=0.6, ls=":")
        ax.set_xlabel("mean set size (levels)", fontsize=8)
        ax.set_title(f"nominal coverage {1 - alpha:.1f}", fontsize=9)
        ax.grid(True, color="#e4e3df", lw=0.6)
    axes[0][0].set_ylabel("held-out rater coverage", fontsize=8)
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(FIGURES / "conformal_coverage_size.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 3 -------------------------------------------------------------------------------------------
    rel = reliability(cells)
    rel.to_csv(TABLES / "reliability.csv", index=False)
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    for m in SHOWN:
        t = rel[rel.method == m]
        ax.plot(t.predicted, t.observed, marker="o", ms=3, lw=2.0 if m == PROPOSED else 1.2, color=COLOR[m], label=LABEL[m])
    ax.plot([0, 1], [0, 1], color="#52514e", lw=0.7, ls=":")
    ax.set_xlabel("predicted P(y > k)", fontsize=8)
    ax.set_ylabel("held-out rater share above k", fontsize=8)
    ax.legend(fontsize=6.5, frameon=False)
    ax.grid(True, color="#e4e3df", lw=0.6)
    fig.tight_layout()
    fig.savefig(FIGURES / "reliability_real_linear.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    sharp = rel.groupby("method")[["entropy", "max_prob"]].first().loc[[m for m in METHODS if m in set(rel.method)]]
    sharp.index = sharp.index.map(LABEL)
    lines += ["## Sharpness (full pool, mean predictive entropy and mean maximum probability)\n", md(sharp), ""]

    ts = temperature_scaling(cells)
    ts.to_csv(TABLES / "temperature_scaling.csv", index=False)
    g = ts.groupby("method")[[c for c in ts.columns if c.startswith(("raw_", "ts_"))] + ["T"]].mean()
    g = g.loc[[m for m in METHODS if m in g.index]]
    g.index = g.index.map(LABEL)
    lines += ["## Post-hoc temperature scaling fitted on the validation sets (mean over all cells)\n",
              "T is the fitted temperature (> 1 softens). ho_nll, ho_ece and int80_iscore before (raw) and after (ts).\n",
              md(g, ".4f"), ""]
    rank = pd.DataFrame({"raw": g["raw_ho_nll"].rank(), "ts": g["ts_ho_nll"].rank()})
    lines += ["Rank of held-out NLL before and after scaling (1 = best):\n", md(rank, ".0f"), ""]
    (TABLES / "setpred.md").write_text("\n".join(lines), encoding="utf8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
