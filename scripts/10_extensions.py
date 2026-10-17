"""Analysis of the experiments beyond the core benchmark (scripts/run_extras.sh).

  targets     what to train on: the uniform-over-set label, the empirical distribution of the m ratings, a 50/50 mixture
              and the exact set likelihood, for the CORN / softmax / CORAL parameterisations; checkpoints selected by the
              NLL against uniform-over-set labels or by the target-agnostic exact set likelihood (xnll)
  baselines   SORD, Gaussian label distributions, CDW-CE, binomial / Poisson heads, cumulative link models
  variants    richer synthetic study (K, dimension, signal, set mechanism, nonlinear generators, noise)
  datasets    VADER sentiment (K = 9) and ConvAbuse severity (K = 5)
  shift       leave one platform / namespace / year / demographic subset out
  personal    per-rater offsets on SCUT: own rating against the population consensus
  selection   what selecting on set-MAE, NLL or RPS does to every method, across all real-data cells
  effects     random-effects summary of the main comparisons across problems

    python scripts/10_extensions.py
Writes results/extensions.md, results/ext_*.csv and figures/ext_*.png.
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

from ordbench.analysis import collect, pairwise_table, random_effects  # noqa: E402

warnings.filterwarnings("ignore")
LABEL = {"soft_corn": "soft-CORN", "soft_coral": "soft-CORAL", "soft_or": "soft-OR", "ce_soft": "soft CE",
         "ce_hard": "CE (hard)", "corn_hard": "CORN (hard)", "coral_hard": "CORAL (hard)", "oldl_cad": "OLDL-CAD",
         "oldl_qfd": "OLDL-QFD", "oldl_cjs": "OLDL-CJS", "pll_softmax": "PLL-softmax", "pll_corn": "PLL-CORN",
         "soft_corn_emp": "soft-CORN, empirical", "ce_emp": "soft CE, empirical", "soft_coral_emp": "soft-CORAL, empirical",
         "soft_corn_mix50": "soft-CORN, 50/50", "ce_mix50": "soft CE, 50/50", "soft_coral_mix50": "soft-CORAL, 50/50",
         "exact_corn": "CORN, exact likelihood", "exact_ce": "softmax, exact likelihood",
         "exact_coral": "CORAL, exact likelihood", "sord": "SORD (Laplace)", "sord_sq": "SORD (Gaussian)",
         "cdwce": "CDW-CE", "binom": "binomial head", "poisson": "Poisson head", "clm_logit": "CLM logit",
         "clm_probit": "CLM probit", "clm_cloglog": "CLM cloglog", "clm_ao": "CLM learned link"}
TARGETS = {"corn": ["soft_corn", "soft_corn_emp", "soft_corn_mix50", "exact_corn"],
           "softmax": ["ce_soft", "ce_emp", "ce_mix50", "exact_ce"],
           "coral": ["soft_coral", "soft_coral_emp", "soft_coral_mix50", "exact_coral"]}
NEW_BASELINES = ["sord", "sord_sq", "cdwce", "binom", "poisson", "clm_logit", "clm_probit", "clm_cloglog", "clm_ao"]
KEY = ["soft_corn", "soft_coral", "soft_or", "ce_soft", "oldl_qfd", "pll_corn", "corn_hard"]
ROPE = {"nll": 0.005, "ho_nll": 0.005, "kl_t": 0.002, "kl_pi": 0.002, "xnll": 0.005, "set_mae_med": 0.005, "ho_ece": 0.005}
LINES = []


def md(df, fmt=".4f", index=True):
    return df.to_markdown(floatfmt=fmt, index=index)


def have(folder):
    return folder.exists() and any(folder.glob("*.npz"))


def lab(index):
    return index.map(lambda m: LABEL.get(m, m))


# --------------------------------------------------------------------------------------------
# training targets
# --------------------------------------------------------------------------------------------

def targets_section():
    global LINES
    for kind, folder, metrics, keys in (("synthetic", RUNS / "targets_synthetic", ("kl_pi", "kl_t"), ["family", "n_train", "m"]),
                                        ("real", RUNS / "targets_real", ("ho_nll", "nll", "xnll"), ["dataset", "n_train", "m"])):
        if not have(folder):
            continue
        df = collect(folder)
        if kind == "synthetic":
            df["n_test"] = 0                                  # independent data per seed: plain paired t-test
        if "xnll" not in df:
            continue
        LINES.append(f"## Training targets, {kind} data\n")
        LINES.append("Uniform-over-set labels (the benchmark's soft targets) against the empirical distribution of the m "
                     "ratings, a 50/50 mixture of the two, and the exact likelihood of the observed set under m independent "
                     "ratings. Checkpoints are selected on validation NLL against uniform-over-set labels (`nll`) or on the "
                     "exact set likelihood (`xnll`), which does not favour any target.\n")
        for crit in ("nll", "xnll"):
            d = df[df.select_by == crit]
            for metric in metrics:
                if metric not in d or d[metric].isna().all():
                    continue
                tab = d.groupby(["m", "n_train", "method"])[metric].mean().unstack("method")
                cols = [m for g in TARGETS.values() for m in g if m in tab]
                t = tab[cols].rename(columns=LABEL)
                t.to_csv(TABLES / f"ext_targets_{kind}_{crit}_{metric}.csv")
                LINES += [f"### {kind}: {metric}, checkpoints selected by {crit} (mean over "
                          f"{'families' if kind == 'synthetic' else 'datasets'} and seeds; lower is better)\n", md(t), ""]
        rows = []
        main_metric = "kl_pi" if kind == "synthetic" else "ho_nll"
        for crit in ("nll", "xnll"):
            for param, group in TARGETS.items():
                base = group[0]
                for other in [m for m in group[1:] if m in set(df.method)]:
                    # `other` is the proposed side, so mean_diff < 0 and p_better refer to the other target being better
                    pw = pairwise_table(df, keys, main_metric, crit, [base], proposed=other, rope=ROPE[main_metric])
                    for r in pw.to_dict("records"):
                        rows.append(dict(selected_by=crit, parameterisation=param, target=LABEL[other], **{
                            k: r[k] for k in ("problems", "mean_diff", "wins", "ties", "losses", "p_better", "p_rope",
                                              "p_worse")}))
        tab = pd.DataFrame(rows)
        tab.to_csv(TABLES / f"ext_targets_{kind}_pairwise.csv", index=False)
        LINES += [f"### {kind}: other target minus uniform-over-set target on {main_metric} (< 0 favours the other target)\n"
                  f"W/T/L of the other target over problems (dataset or family x size x m); p_better = Bayesian "
                  f"P(other target lower), p_worse = P(uniform-over-set target lower).\n",
                  tab.to_markdown(index=False, floatfmt=".4f"), ""]


# --------------------------------------------------------------------------------------------
# extra baselines
# --------------------------------------------------------------------------------------------

def baselines_section():
    global LINES
    for kind, core_dir, base_dir, keys, metrics in (
            ("real", RUNS / "real", RUNS / "baselines_real", ["dataset", "n_train"], ("nll", "ho_nll", "set_mae_med", "ho_ece")),
            ("synthetic", RUNS / "synthetic", RUNS / "baselines_synthetic", ["family", "n_train"], ("kl_t", "kl_pi", "set_mae_med"))):
        if not (have(core_dir) and have(base_dir)):
            continue
        core, new = collect(core_dir), collect(base_dir)
        df = pd.concat([core[core.method.isin(KEY)], new], ignore_index=True)
        if kind == "synthetic":
            df["n_test"] = 0
        LINES.append(f"## Further baselines, {kind} data\n")
        for body in ("linear", "mlp"):
            d = df[df.body == body]
            if d.empty or not d.method.isin(NEW_BASELINES).any():
                continue
            full = d[d.n_train == d.n_train.max()] if kind == "synthetic" else d[d.n_train == d.pool]
            for metric in metrics:
                crit = "set_mae" if metric.startswith("set_mae") else "nll"
                if metric not in d or d[metric].isna().all():
                    continue
                f = full[full.select_by == crit]
                tab = f.groupby(["method"] + (["family"] if kind == "synthetic" else []))[metric].mean()
                tab = tab.unstack("family") if kind == "synthetic" else tab.to_frame(metric)
                tab = tab.loc[[m for m in KEY + NEW_BASELINES if m in tab.index]]
                tab.index = lab(tab.index)
                tab.to_csv(TABLES / f"ext_baselines_{kind}_{body}_{metric}.csv")
                LINES += [f"### {kind}, {body} heads: {metric} "
                          f"({'n = ' + str(int(d.n_train.max())) if kind == 'synthetic' else 'full pool, mean over datasets'}; "
                          "lower is better)\n", md(tab), ""]
                pw = pairwise_table(d, keys, metric, crit, [m for m in NEW_BASELINES if m in set(d.method)],
                                    rope=ROPE.get(metric, 0.005))
                if not pw.empty:
                    pw["baseline"] = pw.baseline.map(LABEL)
                    pw.to_csv(TABLES / f"ext_baselines_{kind}_{body}_{metric}_pairwise.csv", index=False)
                    LINES += [f"#### soft-CORN minus baseline on {metric}, {body} heads (< 0 favours soft-CORN)\n",
                              pw.to_markdown(index=False, floatfmt=".4f"), ""]


# --------------------------------------------------------------------------------------------
# synthetic variants
# --------------------------------------------------------------------------------------------

VARIANT_TITLE = {"base": "main benchmark", "K3": "K = 3", "K8": "K = 8", "K10": "K = 10", "d3": "d = 3", "d50": "d = 50",
                 "d200": "d = 200", "sig05": "signal 0.5", "sig2": "signal 2", "interval": "interval sets",
                 "threshold": "threshold sets", "gapped": "mirror (gapped) sets", "nonlinear": "nonlinear generator",
                 "noise": "20% uniform label noise", "hetero": "rater heterogeneity", "m2": "m = 2 raters", "m5": "m = 5 raters"}
FAMILY = {"po": "proportional odds", "cr": "continuation ratio", "softmax": "multinomial logit", "locscale": "location-scale",
          "bimodal": "polarised mixture"}


def variants_section():
    global LINES
    folders = {"base": RUNS / "synthetic", **{p.name[6:]: p for p in sorted(RUNS.glob("synth_*")) if p.is_dir()}}
    frames = {}
    for name, folder in folders.items():
        if have(folder):
            df = collect(folder)
            frames[name] = df[(df.select_by == "nll") & (df.n_train == 16384)]
    if len(frames) < 2:
        return
    order = [v for v in VARIANT_TITLE if v in frames]
    LINES.append("## Richer synthetic study (n = 16,384; excess risk KL(t || p_hat), lower is better)\n")
    rows, ratio_coral, ratio_ce = [], {}, {}
    for v in order:
        for body in ("linear", "mlp"):
            d = frames[v][frames[v].body == body]
            if d.empty:
                continue
            tab = d.groupby(["family", "method"]).kl_t.mean().unstack("method")
            keep = [m for m in ("soft_corn", "soft_coral", "soft_or", "ce_soft") if m in tab]
            win = tab[keep].idxmin(axis=1).map(LABEL)
            for fam in tab.index:
                rows.append(dict(variant=VARIANT_TITLE[v], body=body, family=FAMILY[fam], winner=win[fam],
                                 **{LABEL[m]: tab.loc[fam, m] for m in keep}))
            if body == "linear":
                ratio_coral[v] = (tab["soft_coral"] / tab["soft_corn"]).reindex(FAMILY)
                ratio_ce[v] = (tab["ce_soft"] / tab["soft_corn"]).reindex(FAMILY)
    full = pd.DataFrame(rows)
    full.to_csv(TABLES / "ext_variants.csv", index=False)
    LINES += ["Winner among soft-CORN, soft-CORAL, soft-OR and soft CE per family, and their excess risks:\n",
              full.to_markdown(index=False, floatfmt=".4f"), ""]
    wins = full[full.body == "linear"].groupby(["variant", "winner"]).size().unstack(fill_value=0)
    wins = wins.loc[[VARIANT_TITLE[v] for v in order if VARIANT_TITLE[v] in wins.index]]
    LINES += ["Number of families (of 5) each loss wins, linear heads:\n", md(wins, ".0f"), ""]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 0.28 * len(order) + 1.6))
    for ax, ratios, title in ((axes[0], ratio_coral, "soft-CORAL / soft-CORN"), (axes[1], ratio_ce, "soft CE / soft-CORN")):
        mat = np.log10(np.array([ratios[v].values for v in order], float))
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
        ax.set_xticks(range(5))
        ax.set_xticklabels([FAMILY[f] for f in FAMILY], rotation=35, ha="right", fontsize=7)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([VARIANT_TITLE[v] for v in order] if ax is axes[0] else [], fontsize=7)
        ax.set_title(f"excess-risk ratio {title} (log10)", fontsize=8)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:+.1f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=axes, shrink=0.8, label="log10 ratio (> 0: soft-CORN has lower risk)")
    fig.savefig(FIGURES / "ext_variants_ratio.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------------------------
# selection rule and random-effects summaries (core real-data runs)
# --------------------------------------------------------------------------------------------

def selection_section():
    global LINES
    path = RUNS / "real_selected.parquet"
    if not path.exists():
        return
    df = pd.read_parquet(path)
    rows = []
    for body in ("linear", "mlp"):
        d = df[(df.body == body) & (df.n_train == df.pool)]
        key = ["dataset", "method"]
        by = {c: d[d.select_by == c].groupby(key).mean(numeric_only=True) for c in ("nll", "set_mae", "rps")}
        for method in sorted(d.method.unique(), key=list(LABEL).index):
            sel = lambda c, m: by[c].xs(method, level="method")[m]  # noqa: E731
            rows.append(dict(body=body, method=LABEL[method],
                             nll_cost_of_setmae_selection=(sel("set_mae", "nll") - sel("nll", "nll")).mean(),
                             nll_cost_of_rps_selection=(sel("rps", "nll") - sel("nll", "nll")).mean(),
                             setmae_cost_of_nll_selection=(sel("nll", "set_mae_med") - sel("set_mae", "set_mae_med")).mean(),
                             ho_nll_cost_of_setmae_selection=(sel("set_mae", "ho_nll") - sel("nll", "ho_nll")).mean(),
                             ece_cost_of_setmae_selection=(sel("set_mae", "ho_ece") - sel("nll", "ho_ece")).mean()
                             if "ho_ece" in d else np.nan))
    tab = pd.DataFrame(rows)
    tab.to_csv(TABLES / "ext_selection.csv", index=False)
    LINES += ["## Effect of the selection rule (full pools, mean over datasets; cost = metric at the other rule's checkpoint "
              "minus metric at its own)\n", tab.to_markdown(index=False, floatfmt=".4f"), ""]


def problem_diffs(df, keys, metric, crit, proposed, baseline):
    """Per problem: mean seed difference and its Nadeau-Bengio corrected variance."""
    d = df[df.select_by == crit]
    out_d, out_v = [], []
    for _, g in d.groupby(keys):
        a = g[g.method == proposed].set_index("seed")[metric]
        c = g[g.method == baseline].set_index("seed")[metric]
        j = a.index.intersection(c.index)
        if len(j) < 3:
            continue
        diff = (a[j] - c[j]).values
        n_tr, n_te = g.n_train.iloc[0], g.n_test.iloc[0] if "n_test" in g else g.n_train.iloc[0]
        out_d.append(diff.mean())
        out_v.append(max(diff.var(ddof=1), 1e-12) * (1 / len(diff) + n_te / n_tr))
    return np.array(out_d), np.array(out_v)


def effects_section():
    global LINES
    rows = []
    sources = []
    if (RUNS / "real_selected.parquet").exists():
        sources.append(("real", pd.read_parquet(RUNS / "real_selected.parquet"), ["dataset", "n_train"]))
    if (RUNS / "finetune_selected.parquet").exists():
        sources.append(("finetune", pd.read_parquet(RUNS / "finetune_selected.parquet"), ["dataset", "n_train"]))
    for name, df, keys in sources:
        bodies = ("linear", "mlp") if name == "real" else ("finetune",)
        for body in bodies:
            d = df[df.body == body]
            for metric, crit in (("nll", "nll"), ("ho_nll", "nll"), ("set_mae_med", "set_mae"), ("ho_ece", "nll")):
                if metric not in d or d[metric].isna().all():
                    continue
                for b in ("soft_coral", "ce_soft", "soft_or"):
                    dd, vv = problem_diffs(d, keys, metric, crit, "soft_corn", b)
                    if len(dd) >= 3:
                        rows.append(dict(source=name, body=body, metric=metric, baseline=LABEL[b],
                                         **random_effects(dd, vv, rope=ROPE.get(metric, 0.005))))
    if not rows:
        return
    tab = pd.DataFrame(rows)
    tab.to_csv(TABLES / "ext_random_effects.csv", index=False)
    LINES += ["## Random-effects summary across problems, soft-CORN minus baseline\n",
              "d_i ~ N(mu, tau^2 + v_i) with v_i the corrected variance of problem i's seed differences; mu < 0 favours "
              "soft-CORN; tau is the between-problem standard deviation of the true difference. The probabilities are from the "
              "normal posterior of mu under a flat prior (ROPE as in the main tables). A stand-in for the Stan-based hierarchical "
              "test of Corani et al. (2017), which needs PyStan.\n", tab.to_markdown(index=False, floatfmt=".4f"), ""]


# --------------------------------------------------------------------------------------------
# distribution shift and personalisation
# --------------------------------------------------------------------------------------------

def shift_section():
    global LINES
    folder = RUNS / "shift"
    if not have(folder):
        return
    df = collect(folder)
    df["family"] = df.dataset.str.split("/").str[0] + "/" + df.dataset.str.split("/").str[1]
    d = df[df.select_by == "nll"]
    LINES.append("## Distribution shift: leave one group out (frozen linear heads)\n")
    LINES.append("Train on every group but one, select checkpoints on in-distribution validation items, read the held-out "
                 "group (`ood`) and an in-distribution test split (`id`). Groups: MHS platform codes 0 / 2 / 3, Wikipedia "
                 "Toxicity namespace and year terciles, SCUT demographic subsets.\n")
    cols = {"ood_nll": "nll", "id_nll": "idtest_nll", "ood_ho_nll": "ho_nll", "id_ho_nll": "idtest_ho_nll",
            "ood_set_mae": "set_mae_med", "id_set_mae": "idtest_set_mae_med"}
    full = d[d.n_train == d.groupby("dataset").n_train.transform("max")]       # largest training size
    tab = full.groupby(["family", "method"])[[c for c in cols.values() if c in full]].mean()
    tab.columns = [k for k, v in cols.items() if v in full]
    tab["shift_cost_nll"] = tab["ood_nll"] - tab["id_nll"]
    tab = tab.loc[(slice(None), [m for m in KEY if m in set(full.method)]), :]
    tab.index = pd.MultiIndex.from_tuples([(f, LABEL[m]) for f, m in tab.index])
    tab.to_csv(TABLES / "ext_shift.csv")
    LINES += ["### Largest training size (the full pool of the training groups), mean over held-out groups and seeds\n",
              md(tab, ".3f"), ""]
    rows = []
    for metric, crit in (("nll", "nll"), ("ho_nll", "nll"), ("set_mae_med", "set_mae")):
        pw = pairwise_table(df, ["dataset", "n_train"], metric, crit, [m for m in KEY if m != "soft_corn"],
                            rope=ROPE.get(metric, 0.005))
        pw["metric"] = metric
        rows.append(pw)
    pw = pd.concat(rows)
    pw["baseline"] = pw.baseline.map(LABEL)
    pw.to_csv(TABLES / "ext_shift_pairwise.csv", index=False)
    LINES += ["### Held-out group: soft-CORN minus baseline (< 0 favours soft-CORN); problems = held-out group x training size\n",
              pw.to_markdown(index=False, floatfmt=".4f"), ""]


def personal_section():
    global LINES
    folder = RUNS / "personal"
    if not have(folder):
        return
    df = collect(folder)
    d = df[df.select_by == "nll"].copy()
    d["offsets"] = d.offsets.astype(bool)
    LINES.append("## Personalisation on SCUT: per-rater offsets (45 known raters, 15 held-out raters)\n")
    LINES.append("Per-rating NLL on test faces. `known` = the raters used for training, `new` = raters never seen (their "
                 "offset is zero). alpha is the weight of the population consensus (the uniform distribution over the set of "
                 "3 known raters' ratings) in the training target, 1 - alpha that of the rater's own rating; hard-label "
                 "losses use the own rating only. ratings = training faces per rater.\n")
    t = d.groupby(["ratings_per_rater", "alpha", "offsets", "method"])[["nll", "newrater_nll"]].mean()
    t.columns = ["known raters", "new raters"]
    t = t.reset_index()
    t["method"] = t.method.map(LABEL)
    t["personalised"] = t.offsets.map({True: "per-rater offsets", False: "population model"})
    piv = t.pivot_table(index=["ratings_per_rater", "alpha", "method"], columns="personalised",
                        values=["known raters", "new raters"])
    piv.to_csv(TABLES / "ext_personal.csv")
    LINES += [md(piv, ".3f"), ""]
    own = d[(d.alpha == 0) & d.offsets]
    base = d[(d.alpha == 0) & ~d.offsets]
    gain = (own.groupby(["ratings_per_rater", "method"]).nll.mean() - base.groupby(["ratings_per_rater", "method"]).nll.mean()).unstack(0)
    gain.index = lab(gain.index)
    LINES += ["### Gain of per-rater offsets on known raters (alpha = 0; negative = better), NLL difference\n", md(gain, ".3f"), ""]


def datasets_section():
    global LINES
    folder = RUNS / "real_extra"
    if not have(folder):
        return
    df = collect(folder)
    title = {"vader_tweets": "VADER tweets (K=9)", "vader_nyt": "VADER NYT (K=9)", "vader_movie": "VADER movie (K=9)",
             "vader_amazon": "VADER Amazon (K=9)", "convabuse": "ConvAbuse (K=5)"}
    LINES.append("## Further multi-annotator datasets (larger K: VADER sentiment, K = 9; ConvAbuse severity, K = 5)\n")
    LINES.append("Same protocol as the core benchmark (frozen mpnet features, 12 methods, 5 seeds, checkpoints selected "
                 "offline on validation). ConvAbuse has only 2-7 annotators per exchange, so with m = 3 most items have "
                 "no held-out rater (the held-out metrics rest on the items with more); m = 2 is added for it.\n")
    for body in ("linear", "mlp"):
        d = df[df.body == body]
        if d.empty:
            continue
        full = d[d.n_train == d.pool]
        for metric, crit in (("nll", "nll"), ("ho_nll", "nll"), ("set_mae_med", "set_mae"), ("ho_ece", "nll")):
            if metric not in full or full[metric].isna().all():
                continue
            t = full[full.select_by == crit].groupby(["dataset", "m", "method"])[metric].mean().unstack("method")
            t = t[[m for m in KEY if m in t]].rename(columns=LABEL)
            t.index = [f"{title.get(a, a)}, m={b}" for a, b in t.index]
            t.to_csv(TABLES / f"ext_datasets_{body}_{metric}.csv")
            LINES += [f"### {body} heads: {metric} at the full pool (lower is better)\n", md(t), ""]
        for metric, crit in (("nll", "nll"), ("ho_nll", "nll"), ("set_mae_med", "set_mae"), ("ho_ece", "nll")):
            if metric not in d or d[metric].isna().all():
                continue
            pw = pairwise_table(d, ["dataset", "m", "n_train"], metric, crit,
                                ["soft_coral", "ce_soft", "soft_or", "oldl_qfd", "pll_corn", "corn_hard"],
                                rope=ROPE.get(metric, 0.005))
            pw["baseline"] = pw.baseline.map(LABEL)
            pw.to_csv(TABLES / f"ext_datasets_{body}_{metric}_pairwise.csv", index=False)
            LINES += [f"#### {body} heads: soft-CORN minus baseline on {metric} (< 0 favours soft-CORN; problems = "
                      "dataset x m x size)\n", pw.to_markdown(index=False, floatfmt=".4f"), ""]


def main():
    LINES.append("# Experiments beyond the core benchmark (auto-generated by scripts/10_extensions.py)\n")
    for f in (targets_section, baselines_section, variants_section, datasets_section, shift_section, personal_section,
              selection_section, effects_section):
        f()
    (TABLES / "extensions.md").write_text("\n".join(LINES), encoding="utf8")
    print("\n".join(LINES))


if __name__ == "__main__":
    main()
