"""Analysis of the K-curve personalisation experiment (scripts/15_kcurve.py) on the ProVoice study.

For every feature set and head, test NLL and set-MAE of every loss against the number K of calibration answers of the
held-out driver. The unit is the driver: metrics are averaged over seeds within a driver, and soft-CORN is compared with
each rival paired over the 12 drivers (mean difference, bootstrap 95% interval over drivers, paired t-test, number of
drivers in which soft-CORN is better). Only (driver, seed) combinations present at every K are used.

    python scripts/16_kcurve.py
Reads $SCB_DATA/results/kcurve_<features>; writes results/kcurve.md, results/kcurve_*.csv and figures/kcurve_*.png.
"""
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
from scipy import stats  # noqa: E402

from ordbench.analysis import collect  # noqa: E402

warnings.filterwarnings("ignore")
PROPOSED = "soft_corn"
LABEL = {"soft_corn": "soft-CORN", "soft_coral": "soft-CORAL", "soft_or": "soft-OR", "ce_soft": "soft CE", "ce_hard": "CE (hard)",
         "corn_hard": "CORN (hard)", "coral_hard": "CORAL (hard)", "oldl_cad": "OLDL-CAD", "oldl_qfd": "OLDL-QFD",
         "oldl_cjs": "OLDL-CJS", "pll_softmax": "PLL-softmax", "pll_corn": "PLL-CORN", "prior": "population prior",
         "personal_prior": "personal prior"}
CORE = ["soft_corn", "soft_coral", "soft_or", "ce_soft", "oldl_qfd", "oldl_cad", "oldl_cjs", "pll_corn", "pll_softmax", "corn_hard",
        "coral_hard", "ce_hard"]
MATCH = {"nll": "nll", "pll": "nll", "set_mae_med": "set_mae"}
METRIC_TITLE = {"nll": "NLL against the uniform-over-set label", "set_mae_med": "set-MAE of the predicted median"}
SERIES = ["soft_corn", "soft_coral", "ce_soft", "corn_hard", "pll_corn", "prior", "personal_prior"]
COLORS = {"soft_corn": "#2a78d6", "soft_coral": "#eb6834", "ce_soft": "#1baf7a", "corn_hard": "#008300", "pll_corn": "#e87ba4",
          "prior": "#52514e", "personal_prior": "#8a8985"}
STYLE = {"prior": ":", "personal_prior": "--"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
FEATURE_NAME = {"task": "task code + driver code", "all": "all features + driver code"}
LINES = []


def md(df, fmt=".3f", index=True):
    return df.to_markdown(floatfmt=fmt, index=index)


def load():
    frames = []
    for feat in ("task", "all"):
        d = RUNS / f"kcurve_{feat}"
        if d.exists() and any(d.glob("*.npz")):
            frames.append(collect(d))
    df = pd.concat(frames, ignore_index=True)
    ks = sorted(df.n_cal.unique())
    cells = df.groupby(["features", "body", "driver", "seed"]).n_cal.nunique()
    complete = cells[cells == len(ks)].index
    key = pd.MultiIndex.from_frame(df[["features", "body", "driver", "seed"]])
    return df[key.isin(complete)].reset_index(drop=True), ks


def driver_table(df, feat, body, metric):
    """(driver, K) x method table of `metric`, averaged over seeds within a driver; prior rows from the cell metadata."""
    d = df[(df.features == feat) & (df.body == body)]
    sel = d[d.select_by == MATCH[metric]]
    t = sel.groupby(["driver", "n_cal", "method"])[metric].mean().unstack("method")
    cell = d[d.select_by == MATCH[metric]].drop_duplicates(["driver", "seed", "n_cal"])
    prior = {"nll": ("prior_nll", "pp_nll"), "set_mae_med": ("prior_set_mae_med", "pp_set_mae_med")}.get(metric)
    if prior:
        g = cell.groupby(["driver", "n_cal"])
        t["prior"] = g[prior[0]].mean()
        t["personal_prior"] = g[prior[1]].mean()
    return t


def bootstrap_ci(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    means = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return np.percentile(means, [2.5, 97.5])


def curve_section(df, ks):
    global LINES
    rows, diffs = [], []
    for feat in sorted(df.features.unique()):
        for body in ("linear", "mlp"):
            if df[(df.features == feat) & (df.body == body)].empty:
                continue
            n_dr = df[(df.features == feat) & (df.body == body)].driver.nunique()
            n_cells = df[(df.features == feat) & (df.body == body)].groupby("driver").seed.nunique().min()
            LINES.append(f"## {FEATURE_NAME[feat]}, {body} head ({n_dr} held-out drivers, at least {n_cells} seeds each)\n")
            for metric in ("nll", "set_mae_med"):
                t = driver_table(df, feat, body, metric)
                m = t.groupby("n_cal").mean()
                for method in m.columns:
                    for k in ks:
                        v = t[method].xs(k, level="n_cal")
                        lo, hi = bootstrap_ci(v.values)
                        rows.append(dict(features=feat, body=body, metric=metric, method=method, k=k, mean=v.mean(), lo=lo, hi=hi,
                                         drivers=len(v)))
                show = [mm for mm in CORE + ["prior", "personal_prior"] if mm in m.columns]
                tab = m[show].T
                tab.index = [LABEL[i] for i in tab.index]
                tab.columns = [f"K={k}" for k in tab.columns]
                LINES += [f"### {METRIC_TITLE[metric]}, mean over drivers\n", md(tab), ""]
                for k in ks:
                    for rival in CORE + ["prior", "personal_prior"]:
                        if rival == PROPOSED or rival not in t.columns:
                            continue
                        dd = (t[PROPOSED] - t[rival]).xs(k, level="n_cal").dropna()
                        if len(dd) < 3:
                            continue
                        lo, hi = bootstrap_ci(dd.values)
                        p = stats.ttest_1samp(dd.values, 0.0).pvalue if dd.std(ddof=1) > 0 else np.nan
                        diffs.append(dict(features=feat, body=body, metric=metric, rival=rival, k=k, mean=dd.mean(), lo=lo, hi=hi, p=p,
                                          better=int((dd < 0).sum()), drivers=len(dd)))
    pd.DataFrame(rows).to_csv(TABLES / "kcurve_curves.csv", index=False)
    d = pd.DataFrame(diffs)
    d.to_csv(TABLES / "kcurve_pairwise.csv", index=False)
    return pd.DataFrame(rows), d


def pairwise_section(diffs, ks):
    global LINES
    LINES.append("## soft-CORN minus rival, paired over drivers\n")
    LINES.append("Negative favours soft-CORN. Mean over the held-out drivers, bootstrap 95% interval over drivers in brackets, "
                 "paired t-test p, `n` the number of drivers in which soft-CORN is better.\n")
    key = ["soft_coral", "soft_or", "ce_soft", "oldl_qfd", "pll_corn", "corn_hard", "prior", "personal_prior"]
    for feat in sorted(diffs.features.unique()):
        for body in ("linear", "mlp"):
            for metric in ("nll", "set_mae_med"):
                sub = diffs[(diffs.features == feat) & (diffs.body == body) & (diffs.metric == metric) & diffs.rival.isin(key)]
                if sub.empty:
                    continue
                cell = sub.assign(txt=lambda x: [f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}] p={p:.3f} n={b}" for m, lo, hi, p, b in
                                                 zip(x["mean"], x.lo, x.hi, x.p, x.better)])
                tab = cell.pivot(index="rival", columns="k", values="txt").reindex(key)
                tab.index = [LABEL[i] for i in tab.index]
                tab.columns = [f"K={k}" for k in tab.columns]
                LINES += [f"### {FEATURE_NAME[feat]}, {body} head, {METRIC_TITLE[metric]}\n", md(tab, index=True), ""]


def gain_section(curves, ks):
    global LINES
    LINES.append("## What K calibration answers buy: change in mean NLL from K = 0\n")
    for feat in sorted(curves.features.unique()):
        for body in ("linear", "mlp"):
            c = curves[(curves.features == feat) & (curves.body == body) & (curves.metric == "nll")]
            if c.empty:
                continue
            piv = c.pivot(index="method", columns="k", values="mean")
            gain = piv.sub(piv[0], axis=0)
            show = [m for m in CORE + ["personal_prior"] if m in gain.index]
            g = gain.loc[show]
            g.index = [LABEL[i] for i in g.index]
            g.columns = [f"K={k}" for k in g.columns]
            LINES += [f"### {FEATURE_NAME[feat]}, {body} head\n", md(g), ""]


def control_section():
    """Does the set information itself help? The same losses trained on one level of every set (upper median, random
    member; the lower median is the hard-label loss of the main runs) against training on the sets."""
    global LINES
    base = RUNS / "kcurve_task"
    folders = {"upper": RUNS / "kcurve_task_collapse_upper", "random": RUNS / "kcurve_task_collapse_random"}
    if not all(f.exists() and any(f.glob("*.npz")) for f in folders.values()):
        return
    sets = collect(base)
    coll = {rule: collect(f) for rule, f in folders.items()}
    LINES.append("## Does the set information help? Training on the sets against training on one level of each set\n")
    LINES.append("Task code + driver code. Test NLL (against the uniform-over-set label) over all test answers, over single-level "
                 "answers only, and over set answers only, at the checkpoint selected on validation NLL; mean over drivers. "
                 "`lower` is the hard-label loss of the main runs (CORN (hard), CE (hard)); `upper` and `random` train the same "
                 "loss on the upper median or a random member of every set. The difference sets minus collapsed is paired over "
                 "drivers (bootstrap 95% interval, paired t-test p, n = drivers in which training on the sets is better).\n")
    rows = []
    for body in ("linear", "mlp"):
        for method, hard, name in (("soft_corn", "corn_hard", "CORN family"), ("ce_soft", "ce_hard", "softmax family")):
            for metric in ("nll", "nll_single", "nll_contig"):
                def per_driver(df, m):
                    d = df[(df.body == body) & (df.select_by == "nll") & (df.method == m)]
                    return d.groupby(["n_cal", "driver"])[metric].mean()
                series = {"sets": per_driver(sets, method), "lower": per_driver(sets, hard),
                          "upper": per_driver(coll["upper"], method), "random": per_driver(coll["random"], method)}
                for k in (5, 10, 20, 40):
                    row = dict(body=body, family=name, metric=metric, k=k)
                    for key, v in series.items():
                        row[key] = v.xs(k, level="n_cal").mean()
                    for key in ("lower", "upper", "random"):
                        dd = (series["sets"].xs(k, level="n_cal") - series[key].xs(k, level="n_cal")).dropna()
                        if len(dd) >= 3 and dd.std(ddof=1) > 0:
                            lo, hi = bootstrap_ci(dd.values)
                            row[f"{key}_diff"], row[f"{key}_lo"], row[f"{key}_hi"] = dd.mean(), lo, hi
                            row[f"{key}_p"], row[f"{key}_n"] = stats.ttest_1samp(dd.values, 0.0).pvalue, int((dd < 0).sum())
                    rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "kcurve_set_control.csv", index=False)
    for body in ("linear", "mlp"):
        for name in ("CORN family", "softmax family"):
            for metric, title in (("nll", "all test answers"), ("nll_single", "single-level test answers"),
                                  ("nll_contig", "set test answers")):
                t = out[(out.body == body) & (out.family == name) & (out.metric == metric)]
                if t.empty:
                    continue
                tab = t[["k", "sets", "lower", "upper", "random"]].set_index("k")
                for key in ("lower", "upper", "random"):
                    if f"{key}_diff" in t:
                        tab[f"sets - {key}"] = [f"{d:+.3f} (p={p:.3f}, n={n})" if pd.notna(d) else "" for d, p, n in
                                                zip(t[f"{key}_diff"], t[f"{key}_p"], t[f"{key}_n"].fillna(0).astype(int))]
                tab.index = [f"K={k}" for k in tab.index]
                LINES += [f"### {name}, {body} head, {title}\n", md(tab), ""]


def figure(curves, metric, path):
    feats = [f for f in ("task", "all") if f in set(curves.features)]
    bodies = [b for b in ("linear", "mlp") if b in set(curves.body)]
    if not feats:
        return
    fig, axes = plt.subplots(len(feats), len(bodies), figsize=(4.6 * len(bodies), 3.2 * len(feats)), sharex=True, squeeze=False)
    for i, feat in enumerate(feats):
        for j, body in enumerate(bodies):
            ax = axes[i][j]
            for method in SERIES:
                c = curves[(curves.features == feat) & (curves.body == body) & (curves.metric == metric) & (curves.method == method)]
                c = c.sort_values("k")
                if c.empty:
                    continue
                x = np.arange(len(c))
                ax.plot(x, c["mean"], color=COLORS[method], lw=1.8 if method in ("soft_corn", "soft_coral") else 1.3,
                        ls=STYLE.get(method, "-"), marker="o" if method not in STYLE else None, ms=3.5, label=LABEL[method])
                if method == "soft_corn":
                    ax.fill_between(x, c.lo, c.hi, color=COLORS[method], alpha=0.15, lw=0)
            ks = sorted(curves.k.unique())
            ax.set_xticks(range(len(ks)))
            ax.set_xticklabels(ks, fontsize=7, color=INK2)
            ax.grid(axis="y", color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.tick_params(axis="y", labelsize=7, colors=INK2)
            ax.set_title(f"{FEATURE_NAME[feat]}, {body} head", fontsize=8, color=INK, loc="left")
            if feat == "all" and metric == "nll":                     # a few splits overfit badly: show the useful range
                ax.set_ylim(1.1, 2.0)
                ax.text(0.99, 0.98, "values above 2.0 are off scale (see the table)", transform=ax.transAxes, ha="right", va="top",
                        fontsize=6.5, color=INK2)
            if i == len(feats) - 1:
                ax.set_xlabel("calibration answers K of the held-out driver", fontsize=7.5, color=INK2)
            if j == 0:
                ax.set_ylabel(METRIC_TITLE[metric], fontsize=7.5, color=INK2)
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=min(len(h), 7), fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    fig.savefig(FIGURES / path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    df, ks = load()
    LINES.append("# ProVoice K-curve personalisation (auto-generated by scripts/16_kcurve.py)\n")
    LINES.append("Leave one driver out; the held-out driver and two validation drivers contribute K calibration answers, the other nine "
                 "drivers all of theirs; a driver code is a feature (ordbench/kcurve.py). K = 0 is the population model.\n")
    curves, diffs = curve_section(df, ks)
    pairwise_section(diffs, ks)
    gain_section(curves, ks)
    control_section()
    figure(curves, "nll", "kcurve_nll.png")
    figure(curves, "set_mae_med", "kcurve_set_mae.png")
    (TABLES / "kcurve.md").write_text("\n".join(LINES), encoding="utf8")
    print(f"wrote {TABLES / 'kcurve.md'}")


if __name__ == "__main__":
    main()
