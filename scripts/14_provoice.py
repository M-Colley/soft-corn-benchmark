"""Natively set-valued labels: the ProVoice in-vehicle study (FUTURE_WORK item 3).

12 drivers answered 1,446 prompts with the set of levels of autonomy (LoA 0..4) they would accept; 6.8% of the
answers are sets. Splits keep a driver's prompts together (3 test drivers, 2 validation drivers, the rest train), 20
random draws, checkpoints selected on validation as in the main benchmark. There are no held-out raters, so the
metrics are read against the accepted sets: NLL against the uniform-over-set label, the log of the probability mass
placed inside the accepted set (`pll`), set-MAE of the predicted median (0 inside the set), RPS and EMD.

    python scripts/14_provoice.py
Reads $SCB_DATA/results/provoice[_task|_centred|_known|_baselines]; writes results/provoice.md, results/provoice_*.csv and
figures/provoice_pairwise*.png.
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ordbench.paths import FIGURES, PROCESSED, RUNS, TABLES  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from ordbench.analysis import collect  # noqa: E402

warnings.filterwarnings("ignore")
PROPOSED = "soft_corn"
LABEL = {"soft_corn": "soft-CORN", "soft_coral": "soft-CORAL", "soft_or": "soft-OR", "ce_soft": "soft CE",
         "ce_hard": "CE (hard)", "corn_hard": "CORN (hard)", "coral_hard": "CORAL (hard)", "oldl_cad": "OLDL-CAD",
         "oldl_qfd": "OLDL-QFD", "oldl_cjs": "OLDL-CJS", "pll_softmax": "PLL-softmax", "pll_corn": "PLL-CORN",
         "sord": "SORD (Laplace)", "sord_sq": "SORD (Gaussian)", "cdwce": "CDW-CE", "binom": "binomial head",
         "poisson": "Poisson head", "clm_logit": "CLM logit", "clm_probit": "CLM probit", "clm_cloglog": "CLM cloglog",
         "clm_ao": "CLM learned link"}
CORE = ["soft_corn", "soft_coral", "soft_or", "ce_soft", "oldl_qfd", "oldl_cad", "oldl_cjs", "pll_corn", "pll_softmax",
        "corn_hard", "coral_hard", "ce_hard"]
# lower is better for all; each is read at the checkpoint chosen by its validation counterpart
METRICS = {"nll": ("nll", "NLL against the uniform-over-set label"),
           "pll": ("nll", "-log P(accepted set)"),
           "set_mae_med": ("set_mae", "set-MAE of the predicted median"),
           "rps": ("rps", "ranked probability score"),
           "emd": ("rps", "earth mover's distance")}
SET_METRICS = ["nll_contig", "pll_contig", "set_mae_med_contig"]       # only items whose answer is a set
HIGHER_BETTER = {"set_acc_med"}
COLORS = {"soft_corn": "#2a78d6", "soft_coral": "#eb6834", "ce_soft": "#1baf7a", "oldl_qfd": "#eda100", "pll_corn": "#e87ba4",
          "corn_hard": "#008300", "soft_or": "#4a3aa7"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LINES = []


def md(df, fmt=".3f", index=True):
    return df.to_markdown(floatfmt=fmt, index=index)


def load():
    frames = []
    for tag, folder in (("all features", "provoice"), ("task only", "provoice_task"), ("driver-centred", "provoice_centred"),
                        ("known driver", "provoice_known"), ("all features", "provoice_baselines")):
        d = RUNS / folder
        if d.exists() and any(d.glob("*.npz")):
            f = collect(d)
            f["features"] = tag
            frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    df["size"] = np.where(df.n_train >= df.pool, "full pool", df.n_train.astype(str))
    return df


def at_match(df, metric):
    crit = METRICS.get(metric, ("nll",))[0]
    return df[df.select_by == crit]


def corrected(diff, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled t-test: mean, standard error, p, and a 95% interval."""
    diff = np.asarray(diff, float)
    diff = diff[np.isfinite(diff)]
    J = len(diff)
    if J < 3:
        return dict(J=J, mean=np.nan, se=np.nan, p=np.nan, lo=np.nan, hi=np.nan)
    se = np.sqrt((1 / J + n_test / n_train) * diff.var(ddof=1))
    t = stats.t.ppf(0.975, J - 1)
    p = 2 * stats.t.sf(abs(diff.mean() / se), J - 1) if se > 0 else np.nan
    return dict(J=J, mean=diff.mean(), se=se, p=p, lo=diff.mean() - t * se, hi=diff.mean() + t * se)


def pairwise(df, metric, features, body, size):
    d = at_match(df[(df.features == features) & (df.body == body) & (df["size"] == size)], metric)
    if metric not in d or d[metric].isna().all():
        return pd.DataFrame()
    wide = d.pivot_table(index="seed", columns="method", values=metric)
    meta = d.drop_duplicates("seed").set_index("seed")
    n_train, n_test = meta.n_train.mean(), meta.n_test.mean()
    rows = []
    for m in wide:
        if m == PROPOSED or PROPOSED not in wide:
            continue
        diff = (wide[PROPOSED] - wide[m]).dropna()
        r = corrected(diff.values, n_train, n_test)
        r.update(baseline=m, better_seeds=int((diff < 0).sum()), features=features, body=body, size=size, metric=metric)
        rows.append(r)
    return pd.DataFrame(rows)


def data_section():
    global LINES
    z = np.load(PROCESSED / "provoice.npz", allow_pickle=True)
    mask, group = z["mask_m0"], z["group"]
    sizes = mask.sum(1)
    gapped = np.array([np.ptp(np.flatnonzero(r)) + 1 > r.sum() for r in mask])
    levels = pd.Series(mask.sum(0), index=[f"LoA {k}" for k in range(mask.shape[1])])
    LINES.append("## Data\n")
    LINES.append(f"{len(mask):,} answers from {len(set(group))} drivers, {mask.shape[1]} levels (LoA 0 = do nothing ... 4 = act "
                 f"autonomously). {np.mean(sizes == 1):.1%} of the answers are single levels, {np.sum(sizes > 1)} are sets "
                 f"({int(np.sum((sizes > 1) & ~gapped))} contiguous, {int(gapped.sum())} with a gap). {z['X'].shape[1]} features per answer: "
                 "per-window summaries of the driver-state and vehicle frames (means, standard deviations, event fractions) and "
                 "a one-hot code of the task asked about.\n")
    LINES += ["Accepted levels (an answer counts for every level in its set):\n", md(levels.to_frame("answers"), ".0f"), ""]
    per = pd.Series(group).value_counts().sort_index()
    LINES += [f"Answers per driver: {per.min()}-{per.max()} (median {int(per.median())}).\n"]


def summary_section(df):
    global LINES
    LINES.append("## Test performance (mean over 20 driver-level splits; lower is better except set accuracy)\n")
    rows = []
    for features in sorted(df.features.unique()):
        for body in ("linear", "mlp"):
            for size in ("512", "full pool"):
                sub = df[(df.features == features) & (df.body == body) & (df["size"] == size)]
                if sub.empty:
                    continue
                for metric in METRICS:
                    d = at_match(sub, metric)
                    if metric not in d or d[metric].isna().all():
                        continue
                    g = d.groupby("method")[metric].agg(["mean", "std", "median"])
                    for m, r in g.iterrows():
                        rows.append(dict(features=features, body=body, size=size, metric=metric, method=m, mean=r["mean"], sd=r["std"],
                                         median=r["median"], seeds=int(d[d.method == m].seed.nunique())))
                prior = sub.drop_duplicates("seed")
                rows.append(dict(features=features, body=body, size=size, metric="nll", method="prior", mean=prior.prior_nll.mean(),
                                 sd=prior.prior_nll.std(), seeds=prior.seed.nunique()))
                rows.append(dict(features=features, body=body, size=size, metric="set_mae_med", method="prior",
                                 mean=prior.prior_set_mae_med.mean(), sd=prior.prior_set_mae_med.std(), seeds=prior.seed.nunique()))
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "provoice_summary.csv", index=False)
    for features in sorted(out.features.unique()):
        for body in ("linear", "mlp"):
            for size in ("512", "full pool"):
                t = out[(out.features == features) & (out.body == body) & (out["size"] == size)]
                if t.empty:
                    continue
                piv = t.pivot_table(index="method", columns="metric", values="mean")
                med = t.pivot_table(index="method", columns="metric", values="median")
                piv["nll median"] = med["nll"]                         # the mean is dominated by the few splits that overfit
                order = [m for m in CORE if m in piv.index] + [m for m in piv.index if m not in CORE and m != "prior"] + \
                        (["prior"] if "prior" in piv.index else [])
                piv = piv.loc[order][[m for m in METRICS if m in piv]]
                piv.index = [LABEL.get(m, "feature-free prior") for m in piv.index]
                LINES += [f"### {features}, {body} head, {size} (n_train = "
                          f"{int(df[(df.features == features) & (df.body == body) & (df['size'] == size)].n_train.mean())})\n",
                          md(piv), ""]


def pairwise_section(df):
    global LINES
    LINES.append("## soft-CORN minus each other loss, paired over the 20 splits\n")
    LINES.append("Negative favours soft-CORN. Mean difference with the Nadeau-Bengio corrected resampled t-test (p) and its 95% "
                 "interval; `seeds` is the number of splits (of 20) in which soft-CORN is better. The splits share drivers, so the "
                 "correction matters: the 20 test sets of 3 drivers overlap heavily.\n")
    allrows = []
    for features in sorted(df.features.unique()):
        for body in ("linear", "mlp"):
            for size in ("512", "full pool"):
                for metric in METRICS:
                    allrows.append(pairwise(df, metric, features, body, size))
    pw = pd.concat([r for r in allrows if len(r)], ignore_index=True)
    pw.to_csv(TABLES / "provoice_pairwise.csv", index=False)
    key = ["soft_coral", "soft_or", "ce_soft", "oldl_qfd", "pll_corn", "corn_hard"]
    for features in sorted(df.features.unique()):
        for body in ("linear", "mlp"):
            for size in ("512", "full pool"):
                sub = pw[(pw.features == features) & (pw.body == body) & (pw["size"] == size)]
                if sub.empty:
                    continue
                tab = sub[sub.baseline.isin(key)].copy()
                tab["baseline"] = tab.baseline.map(LABEL)
                tab = tab.pivot_table(index="baseline", columns="metric", values=["mean", "p", "better_seeds"])
                cols = []
                for metric in METRICS:
                    if ("mean", metric) in tab:
                        cols += [("mean", metric), ("p", metric), ("better_seeds", metric)]
                tab = tab[cols]
                tab.columns = [f"{m} {'diff' if a == 'mean' else a.replace('better_seeds', 'seeds')}" for a, m in tab.columns]
                tab = tab.loc[[LABEL[k] for k in key if LABEL[k] in tab.index]]
                LINES += [f"### {features}, {body} head, {size}\n", md(tab, ".3f"), ""]
    return pw


def prior_section(df):
    """Every loss against the feature-free predictor that outputs the mean training soft label."""
    global LINES
    LINES.append("## Against the feature-free prior (the mean training label), paired over the 20 splits\n")
    LINES.append("Method minus prior on the test drivers; negative means the model beats the label marginal. `beats` is the number of "
                 "splits (of 20) in which it does, p the Nadeau-Bengio corrected test, median the median difference.\n")
    rows = []
    for features in sorted(df.features.unique()):
        for body in ("linear", "mlp"):
            for size in ("512", "full pool"):
                for metric, prior_col in (("nll", "prior_nll"), ("set_mae_med", "prior_set_mae_med")):
                    d = at_match(df[(df.features == features) & (df.body == body) & (df["size"] == size)], metric)
                    if d.empty or metric not in d:
                        continue
                    for m, g in d.groupby("method"):
                        diff = (g.set_index("seed")[metric] - g.set_index("seed")[prior_col]).dropna()
                        r = corrected(diff.values, g.n_train.mean(), g.n_test.mean())
                        r.update(features=features, body=body, size=size, metric=metric, method=m, median=float(diff.median()),
                                 beats=int((diff < 0).sum()))
                        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "provoice_vs_prior.csv", index=False)
    key = [m for m in CORE if m in set(out.method)]
    for features in sorted(out.features.unique()):
        for metric in ("nll", "set_mae_med"):
            sub = out[(out.features == features) & (out.metric == metric) & out.method.isin(key)]
            if sub.empty:
                continue
            tab = sub.pivot_table(index="method", columns=["body", "size"], values=["mean", "median", "beats"])
            tab = tab.loc[key]
            tab.index = tab.index.map(LABEL)
            tab.columns = [f"{b} {s}: {a}" for a, b, s in tab.columns]
            LINES += [f"### {features}, {metric}\n", md(tab, ".3f"), ""]


def set_items_section(df):
    global LINES
    cols = [c for c in SET_METRICS if c in df]
    if not cols:
        return
    LINES.append("## Items whose answer is a set (contiguous sets; about 25 per test split, so noisy)\n")
    LINES.append("Test metrics restricted to the answers that are sets, at the checkpoint selected on validation NLL; mean over the "
                 "splits whose test drivers have at least one set (the count is in `splits`).\n")
    rows = []
    for features in sorted(df.features.unique()):
        for body in ("linear", "mlp"):
            d = df[(df.features == features) & (df.body == body) & (df["size"] == "full pool") & (df.select_by == "nll")]
            if d.empty or "nll_contig" not in d:
                continue
            g = d.groupby("method")[cols].mean()
            g["splits"] = d[d.method == PROPOSED].nll_contig.notna().sum()
            g = g.loc[[m for m in CORE if m in g.index]]
            g.index = g.index.map(LABEL)
            LINES += [f"### {features}, {body} head, full pool\n", md(g, ".3f"), ""]
            wide = d.pivot_table(index="seed", columns="method", values="nll_contig")
            for m in ("corn_hard", "soft_coral", "ce_soft", "pll_corn"):
                if m not in wide:
                    continue
                diff = (wide[PROPOSED] - wide[m]).dropna()
                r = corrected(diff.values, d.n_train.mean(), d.n_test.mean())
                rows.append(dict(features=features, body=body, baseline=LABEL[m], **r))
    if rows:
        out = pd.DataFrame(rows)
        out.to_csv(TABLES / "provoice_set_items_pairwise.csv", index=False)
        LINES += ["### soft-CORN minus rival on the set-valued test answers only (NLL; < 0 favours soft-CORN)\n",
                  "Paired over the splits that have set answers among their test drivers (`J`), Nadeau-Bengio corrected. "
                  "CORN (hard) differs from soft-CORN only on the 98 set answers, so this row measures what the set "
                  "information is worth to the CORN head.\n",
                  md(out[["features", "body", "baseline", "J", "mean", "lo", "hi", "p"]], ".3f", index=False), ""]


def figure(pw, features, path, split):
    sub = pw[(pw.features == features) & pw.baseline.isin(["soft_coral", "soft_or", "ce_soft", "oldl_qfd", "pll_corn", "corn_hard"])]
    metrics = [m for m in ("nll", "pll", "set_mae_med") if m in set(sub.metric)]
    bodies = [b for b in ("linear", "mlp") if b in set(sub.body)]
    if sub.empty:
        return
    order = ["soft_coral", "soft_or", "ce_soft", "oldl_qfd", "pll_corn", "corn_hard"]
    fig, axes = plt.subplots(len(bodies), len(metrics), figsize=(3.1 * len(metrics) + 1.2, 2.5 * len(bodies) + 0.6), sharey=True,
                             squeeze=False)
    for i, body in enumerate(bodies):
        for j, metric in enumerate(metrics):
            ax = axes[i][j]
            for k, m in enumerate(order):
                for size, dy, filled in (("full pool", -0.12, True), ("512", 0.12, False)):
                    r = sub[(sub.body == body) & (sub.metric == metric) & (sub.baseline == m) & (sub["size"] == size)]
                    if r.empty:
                        continue
                    r = r.iloc[0]
                    ax.plot([r.lo, r.hi], [k + dy] * 2, color=COLORS[m], lw=1.4, solid_capstyle="round")
                    ax.plot(r["mean"], k + dy, "o", ms=5.5, color=COLORS[m] if filled else "white", mec=COLORS[m], mew=1.4, zorder=3)
            ax.axvline(0, color=INK2, lw=0.8)
            ax.set_yticks(range(len(order)))
            ax.set_yticklabels([LABEL[m] for m in order], fontsize=8, color=INK)
            ax.invert_yaxis()
            ax.grid(axis="x", color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            for s in ("top", "right", "left"):
                ax.spines[s].set_visible(False)
            ax.tick_params(axis="y", length=0)
            ax.tick_params(axis="x", labelsize=7, colors=INK2)
            ax.set_title(f"{METRICS[metric][1]}\n{body} head", fontsize=8, color=INK, loc="left")
    fig.supxlabel(f"soft-CORN minus rival, mean with 95% interval over 20 {split} splits (< 0: soft-CORN better).   "
                  "filled: full pool, open: 512 answers", fontsize=7.5, color=INK2)
    fig.tight_layout()
    fig.savefig(FIGURES / path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    df = load()
    LINES.append("# ProVoice study 1: natively set-valued labels (auto-generated by scripts/14_provoice.py)\n")
    data_section()
    summary_section(df)
    pw = pairwise_section(df)
    prior_section(df)
    set_items_section(df)
    figure(pw, "all features", "provoice_pairwise.png", "driver-level")
    figure(pw, "known driver", "provoice_pairwise_known.png", "window-level (known driver)")
    (TABLES / "provoice.md").write_text("\n".join(LINES), encoding="utf8")
    print(f"wrote {TABLES / 'provoice.md'}")


if __name__ == "__main__":
    main()
