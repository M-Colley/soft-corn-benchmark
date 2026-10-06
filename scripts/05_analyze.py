"""Model selection, tables, statistical tests and figures for the real and synthetic runs.

    python scripts/05_analyze.py
Writes results/*.csv, results/summary.md and figures/*.png (the full selected-run tables go to
$SCB_DATA/results/*.parquet).
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ordbench.paths import FIGURES, RUNS, TABLES  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ordbench.analysis import collect, corrected_ttest  # noqa: E402
from ordbench.train import LRS, WDS  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
PROPOSED = "soft_corn"
LABEL = {"soft_corn": "soft-CORN", "soft_coral": "soft-CORAL", "soft_or": "soft-OR", "ce_soft": "soft CE",
         "ce_hard": "CE (hard)", "corn_hard": "CORN (hard)", "coral_hard": "CORAL (hard)",
         "oldl_cad": "OLDL-CAD", "oldl_qfd": "OLDL-QFD", "oldl_cjs": "OLDL-CJS",
         "pll_softmax": "PLL-softmax", "pll_corn": "PLL-CORN"}
# lower is better for all of these
METRICS = ["set_mae_med", "nll", "ho_nll", "rps", "ho_rps", "emd", "pll"]
ROPE = {"set_mae_med": 0.005, "nll": 0.005, "ho_nll": 0.005, "rps": 0.001, "ho_rps": 0.001,
        "emd": 0.005, "pll": 0.005, "kl_t": 0.002, "kl_pi": 0.002}
# each test metric is read at the step/config chosen by its validation counterpart
MATCH = {"set_mae_med": "set_mae", "set_acc_med": "set_mae", "nll": "nll", "ho_nll": "nll", "pll": "nll",
         "rps": "rps", "ho_rps": "rps", "emd": "rps"}
# figure series: fixed categorical order from the dataviz reference palette (light mode)
# (colour follows the method in every figure; OLDL is represented by its strongest variant, QFD)
SERIES = ["soft_corn", "soft_coral", "ce_soft", "oldl_qfd", "pll_corn", "corn_hard", "soft_or"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
COLOR_OF = dict(zip(SERIES, COLORS))
SOFT_SERIES = ["soft_corn", "soft_coral", "ce_soft", "oldl_qfd", "soft_or"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def fmt(m, s, digits=3):
    return f"{m:.{digits}f} ± {s:.{digits}f}"


def md_table(df, floatfmt=".3f"):
    return df.to_markdown(floatfmt=floatfmt)


# ---------------------------------------------------------------------------------------------
# Selection diagnostics
# ---------------------------------------------------------------------------------------------

def edge_report(df, steps_max):
    d = df[df.select_by == "nll"]
    out = d.groupby("body").apply(lambda g: pd.Series({
        "lr_min": (g.lr == min(LRS)).mean(), "lr_max": (g.lr == max(LRS)).mean(),
        "wd_min": (g.wd == min(WDS)).mean(), "wd_max": (g.wd == max(WDS)).mean(),
        "last_step": (g.step == steps_max).mean(), "n": len(g)}), include_groups=False)
    return out


# ---------------------------------------------------------------------------------------------
# Real data
# ---------------------------------------------------------------------------------------------

def problem_means(df, keys, metric):
    return df.groupby(keys + ["method"])[metric].mean().unstack("method")


def pairwise(df, keys, metric, crit, baselines, tcols=("n_train", "n_test")):
    """soft-CORN minus each baseline. Per problem: mean difference over seeds and the corrected
    resampled t-test (alpha 0.05). Across problems: win/tie/loss counts and the Bayesian
    signed-rank test (Benavoli et al., 2017) with a region of practical equivalence."""
    import baycomp
    d = df[df.select_by == crit]
    rows = []
    for b in baselines:
        per = []
        for key, g in d.groupby(keys):
            a = g[g.method == PROPOSED].set_index("seed")[metric]
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
            rope = ROPE.get(metric, ROPE.get(metric.rsplit("_", 1)[0], 0.0))   # e.g. nll_gapped -> nll
            pl, pr, pg = baycomp.SignedRankTest.probs(per.a.values, per.b.values, rope=rope)
        except Exception:
            pl = pr = pg = np.nan
        rows.append(dict(baseline=b, problems=len(per), mean_diff=per["diff"].mean(),
                         se_diff=per["diff"].std(ddof=1) / np.sqrt(len(per)),
                         median_diff=per["diff"].median(), wins=win, ties=len(per) - win - loss, losses=loss,
                         # baycomp (higher = better) returns P(a > b + rope), P(rope), P(b > a + rope);
                         # all metrics here are lower-is-better, so its "left" is soft-CORN worse
                         p_better=pg, p_rope=pr, p_worse=pl))
    cols = ["baseline", "problems", "mean_diff", "se_diff", "median_diff", "wins", "ties", "losses",
            "p_better", "p_rope", "p_worse"]
    return pd.DataFrame(rows, columns=cols)


def rank_analysis(df, keys, metric, crit, tag):
    """Friedman + Nemenyi (autorank) on per-problem means; saves a critical-difference diagram."""
    from autorank import autorank, plot_stats
    d = df[df.select_by == crit]
    tab = problem_means(d, keys, metric).dropna(axis=1)
    tab.columns = [LABEL[c] for c in tab.columns]
    res = autorank(tab, alpha=0.05, verbose=False, order="ascending", approach="frequentist",
                   force_mode="nonparametric")
    ax = plot_stats(res, allow_insignificant=True)
    fig = ax.get_figure()
    fig.set_size_inches(7.5, 2.8)
    fig.savefig(FIGURES / f"cd_{tag}_{metric}_{crit}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    r = res.rankdf[["meanrank"]].copy()
    r["n_problems"] = len(tab)
    r["friedman_p"] = res.pvalue
    r["cd"] = res.cd
    return r


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(which="both", colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def curves(df, panel_key, panels, x, metric, crit, body, path, ylabel, ref=None, logy=False, titles=None,
           series=SERIES):
    d = df[(df.select_by == crit) & (df.body == body)]
    if d.empty:
        return
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 2.6), squeeze=False)
    for ax, pnl in zip(axes[0], panels):
        g = d[d[panel_key] == pnl]
        if g.empty:
            ax.set_visible(False)
            continue
        tab = g.groupby([x, "method"])[metric].agg(["mean", "sem"]).reset_index()
        base = tab[tab.method == ref].set_index(x)["mean"] if ref else None
        for m in series:
            c = COLOR_OF[m]
            t = tab[tab.method == m].set_index(x).sort_index()
            if t.empty:
                continue
            y = t["mean"] - (base.reindex(t.index) if ref else 0)
            lw = 2.4 if m == PROPOSED else 1.6
            ax.plot(t.index, y, color=c, lw=lw, marker="o", ms=4, label=LABEL[m], zorder=3 if m == PROPOSED else 2)
            ax.fill_between(t.index, y - t["sem"], y + t["sem"], color=c, alpha=0.12, lw=0)
        if ref:
            ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xscale("log")
        xs = sorted(g[x].unique())
        ax.set_xticks(xs)
        # blank a label that would collide with the next (larger) tick on the log axis
        ax.set_xticklabels(["" if len(xs) > 2 and j + 1 < len(xs) and xs[j + 1] / v < 2 else f"{v:,}"
                            for j, v in enumerate(xs)], fontsize=7)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        if logy:
            ax.set_yscale("log")
        ax.set_title(titles.get(pnl, pnl) if titles else pnl, fontsize=9, color=INK)
        ax.set_xlabel("training items", fontsize=8, color=INK2)
        style(ax)
    axes[0][0].set_ylabel(ylabel, fontsize=8, color=INK2)
    h, lab = next((ax.get_legend_handles_labels() for ax in axes[0] if ax.get_legend_handles_labels()[0]),
                  ([], []))
    fig.legend(h, lab, loc="lower center", ncol=len(lab), fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


DATASET_TITLE = {"cb": "CommitmentBank (K=7)", "scut": "SCUT-FBP5500 (K=5)", "humicroedit": "Humicroedit (K=4)",
                 "mhs": "MHS sentiment (K=5)", "wikitox": "Wiki Toxicity (K=5)"}
FAMILY_TITLE = {"po": "proportional odds", "cr": "continuation ratio", "softmax": "multinomial logit",
                "locscale": "location-scale", "bimodal": "polarised mixture"}


def m_ablation(real, lines, sizes=(512, 2048)):
    """Annotators per set (m = 2, 3, 5), linear heads, training sizes {512, 2048, full}. m = 3 comes
    from the main runs restricted to the same sizes. Pairwise tests use the matched selection rule."""
    abl_dir = RUNS / "m_ablation"
    if not (abl_dir.exists() and any(abl_dir.glob("*.npz"))):
        return
    ab = collect(abl_dir)
    ab.to_parquet(RUNS / "m_ablation_selected.parquet")
    main3 = real[(real.body == "linear") & (real.n_train.isin(sizes) | (real.n_train == real.pool))]
    df = pd.concat([ab, main3], ignore_index=True)
    df = df[df.body == "linear"]
    # compare m on the datasets that have enough raters for every m (MHS has too few items at m = 5)
    common = set.intersection(*(set(g.dataset) for _, g in df.groupby("m")))
    df = df[df.dataset.isin(common)]
    lines += ["## Annotators per set (m ablation, linear heads, n ∈ {512, 2048, full})\n"]
    comp = []
    for m, g in df.groupby("m"):
        comp.append(dict(m=m, datasets=", ".join(sorted(g.dataset.unique())), seeds=g.seed.nunique(),
                         problems=g.groupby(["dataset", "n_train"]).ngroups))
    lines += [pd.DataFrame(comp).to_markdown(index=False), ""]
    rows = []
    rivals = ["soft_coral", "ce_soft", "soft_or", "oldl_qfd", "pll_corn"]
    for m, g in df.groupby("m"):
        if g.seed.nunique() < 2:
            continue
        for metric in ("nll", "ho_nll", "set_mae_med", "nll_gapped", "set_mae_med_gapped"):
            crit = "set_mae" if metric.startswith("set_mae") else "nll"
            if metric not in g or g[metric].isna().all():
                continue
            pw = pairwise(g.dropna(subset=[metric]), ["dataset", "n_train"], metric, crit, rivals)
            for r in pw.to_dict("records"):
                rows.append(dict(m=m, metric=metric, **r))
    tab = pd.DataFrame(rows)
    if tab.empty:
        return
    tab["baseline"] = tab.baseline.map(LABEL)
    tab.to_csv(TABLES / "m_ablation_pairwise.csv", index=False)
    for metric in ("nll", "ho_nll", "set_mae_med", "nll_gapped", "set_mae_med_gapped"):
        t = tab[tab.metric == metric]
        if t.empty:
            continue
        t = t.assign(cell=t.apply(lambda r: f"{r.mean_diff:+.3f} ({r.wins}/{r.ties}/{r.losses}; "
                                            f"P↑ {r.p_better:.2f})", axis=1))
        piv = t.pivot(index="baseline", columns="m", values="cell")
        lines += [f"### soft-CORN minus baseline on {metric} by m (W/T/L; P↑ = Bayesian P(soft-CORN better))\n",
                  piv.to_markdown(), ""]
    # set-type composition of the test sets, from the per-type metrics' availability is not enough:
    # recompute from the processed labels for each m
    import numpy as np
    from ordbench.paths import PROCESSED
    from ordbench.metrics import set_types
    import torch
    mix = []
    for name in sorted(df.dataset.unique()):
        z = np.load(PROCESSED / f"{name}.npz", allow_pickle=True)
        for m in sorted(df[df.dataset == name].m.unique()):
            v = z[f"valid_m{m}"]
            t = set_types(torch.as_tensor(z[f"mask_m{m}"][v])).numpy()
            mix.append(dict(dataset=name, m=m, single=(t == 0).mean(), contiguous=(t == 1).mean(), gapped=(t == 2).mean()))
    mix = pd.DataFrame(mix)
    mix.to_csv(TABLES / "m_ablation_set_types.csv", index=False)
    lines += ["### Set-type mix of the labels by m\n", mix.to_markdown(index=False, floatfmt=".2f"), ""]
    # figure: soft-CORN minus its closest rivals as m grows, on all items and on gapped sets
    panels = (("nll", "NLL, all items"), ("nll_gapped", "NLL, gapped sets"),
              ("set_mae_med", "set-MAE, all items"), ("set_mae_med_gapped", "set-MAE, gapped sets"))
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 2.9))
    for ax, (metric, title) in zip(axes, panels):
        for k, b in enumerate(("soft_coral", "ce_soft", "soft_or")):
            t = tab[(tab.metric == metric) & (tab.baseline == LABEL[b])].sort_values("m")
            if t.empty:
                continue
            x = t.m + (k - 1) * 0.06            # small dodge so error bars do not overlap
            ax.errorbar(x, t.mean_diff, yerr=t.se_diff, marker="o", ms=5, lw=1.8, capsize=2.5,
                        color=COLOR_OF[b], label=f"soft-CORN − {LABEL[b]}")
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks([2, 3, 5])
        ax.set_xlabel("annotators per set (m)", fontsize=8, color=INK2)
        ax.set_title(title, fontsize=9, color=INK)
        style(ax)
    axes[0].set_ylabel("difference (< 0 favours soft-CORN)", fontsize=8, color=INK2)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    fig.savefig(FIGURES / "m_ablation_linear.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def finetune_section(real, lines):
    """Fine-tuned backbones (scripts/06_finetune.py) against the frozen linear probe at the same
    training size (2,048 items; CommitmentBank's full 840)."""
    ft_dir = RUNS / "finetune"
    if not (ft_dir.exists() and any(ft_dir.glob("*.npz"))):
        return
    ft = collect(ft_dir)
    ft.to_parquet(RUNS / "finetune_selected.parquet")
    methods = sorted(ft.method.unique(), key=lambda m: list(LABEL).index(m))
    keys = ft[["dataset", "n_train"]].drop_duplicates()
    fr = real[(real.body == "linear") & real.method.isin(methods)].merge(keys, on=["dataset", "n_train"])
    df = pd.concat([ft, fr], ignore_index=True)
    sizes = ", ".join(f"{ds} {', '.join(map(str, sorted(g.n_train.unique())))}" for ds, g in ft.groupby("dataset"))
    lines += [f"## Fine-tuned backbones (training sizes: {sizes}; seeds: {ft.seed.nunique()})\n",
              "Backbone and loss trained end to end (all-mpnet-base-v2 for text, OpenCLIP ViT-B/16 for SCUT), "
              "compared with the frozen linear probe on the same training items. Learning rate and epoch "
              "selected on validation; each metric read at its matched selection.\n"]
    for metric in ("nll", "ho_nll", "set_mae_med"):
        crit = MATCH[metric]
        d = df[df.select_by == crit]
        g = d.groupby(["dataset", "n_train", "body", "method"])[metric].agg(["mean", "std"]).reset_index()
        g["cell"] = g.apply(lambda r: fmt(r["mean"], r["std"]), axis=1)
        g["method"] = g.method.map(LABEL)
        piv = g.pivot_table(index=["dataset", "n_train", "method"], columns="body", values="cell", aggfunc="first")
        piv.to_csv(TABLES / f"finetune_{metric}.csv")
        lines += [f"### {metric}: frozen linear probe vs fine-tuned (mean ± sd over seeds)\n", md_table(piv), ""]
    ftt = ft.copy()
    for metric in ("nll", "ho_nll", "set_mae_med", "rps"):
        pw = pairwise(ftt, ["dataset", "n_train"], metric, MATCH[metric], [m for m in methods if m != PROPOSED])
        pw["baseline"] = pw.baseline.map(LABEL)
        pw.to_csv(TABLES / f"pairwise_finetune_{metric}.csv", index=False)
        lines += [f"### fine-tuned: soft-CORN vs baselines on {metric} (problems = datasets; W/T/L per dataset by "
                  "corrected resampled t-test over 3 seeds)\n", pw.to_markdown(index=False, floatfmt=".3f"), ""]
    # per-dataset differences, so the direction on each dataset is visible
    rows = []
    for metric in ("nll", "set_mae_med"):
        d = ft[ft.select_by == MATCH[metric]].groupby(["dataset", "n_train", "method"])[metric].mean().unstack()
        for b in methods:
            if b != PROPOSED:
                for ds in d.index:
                    rows.append(dict(metric=metric, dataset=f"{ds[0]} (n={ds[1]})", baseline=LABEL[b],
                                     diff=d.loc[ds, PROPOSED] - d.loc[ds, b]))
    if rows:
        per = pd.DataFrame(rows).pivot_table(index=["metric", "baseline"], columns="dataset", values="diff")
        lines += ["### fine-tuned: soft-CORN minus baseline per dataset (< 0 favours soft-CORN)\n", md_table(per), ""]
    # figure: frozen vs fine-tuned test NLL, one panel per dataset, at each dataset's most-run size
    d = df[df.select_by == "nll"]
    main_n = ft.groupby("dataset").n_train.agg(lambda s: s.value_counts().idxmax())
    d = d[d.n_train == d.dataset.map(main_n)]
    datasets = [x for x in DATASET_TITLE if x in set(d.dataset)]
    fig, axes = plt.subplots(1, len(datasets), figsize=(2.3 * len(datasets), 2.8), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        g = d[d.dataset == ds].groupby(["body", "method"]).nll.agg(["mean", "sem"]).reset_index()
        for k, m in enumerate(methods):
            t = g[g.method == m].set_index("body").reindex(["linear", "finetune"])
            x = np.array([0, 1]) + (k - 1) * 0.06
            ax.errorbar(x, t["mean"], yerr=t["sem"], marker="o", ms=5, lw=1.8, capsize=2.5, color=COLOR_OF[m],
                        label=LABEL[m], zorder=3 if m == PROPOSED else 2)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["frozen", "fine-tuned"], fontsize=8)
        ax.set_xlim(-0.35, 1.35)
        ax.set_title(DATASET_TITLE[ds], fontsize=9, color=INK)
        style(ax)
    axes[0][0].set_ylabel("test NLL", fontsize=8, color=INK2)
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=len(lab), fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    fig.savefig(FIGURES / "finetune_vs_frozen_nll.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    FIGURES.mkdir(exist_ok=True)
    TABLES.mkdir(exist_ok=True)
    lines = ["# Benchmark results (auto-generated by scripts/05_analyze.py)\n"]
    baselines = [m for m in LABEL if m != PROPOSED]

    real_dir, syn_dir = RUNS / "real", RUNS / "synthetic"
    if real_dir.exists() and any(real_dir.glob("*.npz")):
        df = collect(real_dir)
        df.to_parquet(RUNS / "real_selected.parquet")
        steps_max = df.step.max()
        df["problem"] = df.dataset + "/n" + df.n_train.astype(str) + "/" + df.body
        lines += ["## Real data\n", f"{df[['dataset', 'n_train', 'body', 'seed']].drop_duplicates().shape[0]} "
                  f"runs, {df.method.nunique()} methods.\n",
                  "### Selection at grid edges (share of selected configs, NLL selection)\n",
                  md_table(edge_report(df, steps_max), ".2f"), ""]
        full = df[df.n_train == df.pool]
        lines += ["Each test metric is read at the step and configuration selected by its validation "
                  "counterpart (set-MAE for set-MAE/accuracy, NLL for NLL/held-out NLL, RPS for RPS).\n"]
        for body in ("linear", "mlp"):
            d = full[full.body == body]
            if d.empty:
                continue
            cols = {}
            for metric in ["set_mae_med", "set_acc_med", "nll", "ho_nll", "rps", "ho_rps"]:
                g = d[d.select_by == MATCH[metric]].groupby(["dataset", "method"])[metric].agg(["mean", "std"])
                cols[metric] = g.apply(lambda r: fmt(r["mean"], r["std"]), axis=1)
            tab = pd.DataFrame(cols).reset_index()
            tab["method"] = tab.method.map(LABEL)
            tab.to_csv(TABLES / f"real_full_{body}.csv", index=False)
            lines += [f"### Full training pool, {body} head (mean ± sd over seeds)\n", tab.to_markdown(index=False), ""]
            prior = d.groupby("dataset")[["prior_nll", "prior_ho_nll", "prior_set_mae_med"]].mean()
            lines += ["Feature-free reference (training-label marginal):\n", md_table(prior), ""]
        # the thesis protocol: every metric at the set-MAE-selected checkpoint
        d = full[(full.body == "linear") & (full.select_by == "set_mae")]
        thesis = d.groupby(["dataset", "method"])[["set_mae_med", "nll", "ho_nll"]].mean().unstack("dataset")
        thesis.index = thesis.index.map(LABEL)
        thesis.to_csv(TABLES / "real_full_linear_thesis_protocol.csv")
        lines += ["### Thesis protocol (all metrics at the set-MAE-selected checkpoint), linear, full pool\n",
                  md_table(thesis), ""]
        for body in ("linear", "mlp"):
            db = df[df.body == body]
            if db.seed.nunique() < 2:
                continue
            for metric in METRICS:
                pw = pairwise(db, ["dataset", "n_train"], metric, MATCH[metric], baselines)
                pw["baseline"] = pw.baseline.map(LABEL)
                pw.to_csv(TABLES / f"pairwise_real_{body}_{metric}.csv", index=False)
                if metric in ("set_mae_med", "nll", "ho_nll", "rps"):
                    lines += [f"### {body}: soft-CORN vs baselines on {metric}; diff < 0 favours soft-CORN; W/T/L by "
                              "corrected resampled t-test per (dataset, n) problem; P(better/rope/worse) from the "
                              "Bayesian signed-rank test across problems\n", pw.to_markdown(index=False, floatfmt=".3f"), ""]
            # sample-size regimes for the two closest competitors
            for metric in ("nll", "ho_nll", "set_mae_med"):
                for regime, sel in (("n <= 512", db.n_train <= 512), ("n >= 2048", db.n_train >= 2048)):
                    pw = pairwise(db[sel], ["dataset", "n_train"], metric, MATCH[metric], ["soft_coral", "ce_soft"])
                    if pw.empty:
                        continue
                    pw["baseline"] = pw.baseline.map(LABEL)
                    lines += [f"#### {body}, {regime}: soft-CORN vs closest competitors on {metric}\n",
                              pw.to_markdown(index=False, floatfmt=".3f"), ""]
            for metric in ("set_mae_med", "nll", "ho_nll", "rps"):
                r = rank_analysis(db, ["dataset", "n_train"], metric, MATCH[metric], f"real_{body}")
                r.to_csv(TABLES / f"ranks_real_{body}_{metric}.csv")
                lines += [f"### {body}: mean ranks on {metric} (Friedman p = {r.friedman_p.iloc[0]:.2g}, "
                          f"Nemenyi CD = {r.cd.iloc[0]:.2f}, {r.n_problems.iloc[0]} problems)\n",
                          md_table(r[["meanrank"]], ".2f"), ""]
        # set-type subgroups (full pool)
        sub = []
        for body in ("linear", "mlp"):
            for st in ("single", "contig", "gapped"):
                for metric, crit in ((f"nll_{st}", "nll"), (f"set_mae_med_{st}", "set_mae")):
                    d = full[(full.select_by == crit) & (full.body == body)]
                    if d.empty or metric not in d:
                        continue
                    pm = d.groupby(["dataset", "method"])[metric].mean().unstack()
                    for b in ("soft_coral", "ce_soft", "pll_corn"):
                        for ds in pm.index:
                            sub.append(dict(body=body, dataset=ds, set_type=st, metric=metric.rsplit("_", 1)[0],
                                            baseline=LABEL[b], diff=pm.loc[ds, PROPOSED] - pm.loc[ds, b]))
        sub = pd.DataFrame(sub)
        if not sub.empty:
            sub.to_csv(TABLES / "real_set_type_diffs.csv", index=False)
            piv = sub.pivot_table(index=["body", "metric", "baseline"], columns="set_type", values="diff", aggfunc="mean")
            lines += ["### soft-CORN minus baseline by set type (full pool, mean over datasets; < 0 favours soft-CORN)\n",
                      md_table(piv), ""]
        m_ablation(df, lines)
        finetune_section(df, lines)
        datasets = [d for d in DATASET_TITLE if d in set(df.dataset)]
        for body in ("linear", "mlp"):
            curves(df, "dataset", datasets, "n_train", "nll", "nll", body,
                   FIGURES / f"real_nll_vs_softcoral_{body}.png", "test NLL − soft-CORAL", ref="soft_coral",
                   titles=DATASET_TITLE, series=SOFT_SERIES)
            curves(df, "dataset", datasets, "n_train", "ho_nll", "nll", body,
                   FIGURES / f"real_honll_vs_softcoral_{body}.png", "held-out-rater NLL − soft-CORAL",
                   ref="soft_coral", titles=DATASET_TITLE, series=SOFT_SERIES)
            curves(df, "dataset", datasets, "n_train", "set_mae_med", "set_mae", body,
                   FIGURES / f"real_setmae_vs_softcoral_{body}.png", "set-MAE − soft-CORAL", ref="soft_coral",
                   titles=DATASET_TITLE, series=SOFT_SERIES + ["pll_corn"])

    if syn_dir.exists() and any(syn_dir.glob("*.npz")):
        sy = collect(syn_dir)
        sy.to_parquet(RUNS / "synthetic_selected.parquet")
        lines += ["## Synthetic data\n", "Excess risk KL(t || p̂) against the exact population target, "
                  "mean over seeds (selection on validation NLL of observed sets).\n"]
        for body in ("linear", "mlp"):
            d = sy[(sy.select_by == "nll") & (sy.body == body)]
            if d.empty:
                continue
            tab = d.groupby(["family", "n_train", "method"]).kl_t.mean().unstack("method")
            tab.columns = [LABEL[c] for c in tab.columns]
            tab.to_csv(TABLES / f"synthetic_klt_{body}.csv")
            lines += [f"### KL(t || p̂), {body} head\n", md_table(tab, ".4f"), ""]
            curves(sy, "family", [f for f in FAMILY_TITLE if f in set(sy.family)], "n_train", "kl_t", "nll", body,
                   FIGURES / f"synthetic_klt_{body}.png", "KL(target ‖ prediction)", logy=True, titles=FAMILY_TITLE)
            curves(sy, "family", [f for f in FAMILY_TITLE if f in set(sy.family)], "n_train", "kl_pi", "nll", body,
                   FIGURES / f"synthetic_klpi_{body}.png", "KL(raters ‖ prediction)", logy=True, titles=FAMILY_TITLE)
        sy["n_test"] = 0   # independent data per seed: plain paired t-test (no overlap correction)
        for body in ("linear", "mlp"):
            db = sy[sy.body == body]
            if db.seed.nunique() < 2:
                continue
            for metric in ("kl_t", "kl_pi", "set_mae_med"):
                crit = "set_mae" if metric == "set_mae_med" else "nll"
                pw = pairwise(db, ["family", "n_train"], metric, crit, baselines)
                pw["baseline"] = pw.baseline.map(LABEL)
                pw.to_csv(TABLES / f"pairwise_synthetic_{body}_{metric}.csv", index=False)
                lines += [f"### {body}: soft-CORN vs baselines on synthetic {metric} (all families and sizes)\n",
                          pw.to_markdown(index=False, floatfmt=".4f"), ""]
            for fam in FAMILY_TITLE:
                d = db[db.family == fam]
                if d.empty:
                    continue
                pw = pairwise(d, ["n_train"], "kl_t", "nll", ["soft_coral", "ce_soft", "soft_or", "oldl_qfd"])
                pw["baseline"] = pw.baseline.map(LABEL)
                lines += [f"#### {body}, {FAMILY_TITLE[fam]}: soft-CORN vs key baselines on KL(t || p̂) "
                          "(problems = training sizes)\n", pw.to_markdown(index=False, floatfmt=".4f"), ""]

    (TABLES / "summary.md").write_text("\n".join(lines), encoding="utf8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
