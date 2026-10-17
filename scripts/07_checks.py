"""Grid-edge robustness checks (REPORT.md, section 6), from the extra run folders that
scripts/run_frozen.sh writes next to the main benchmark.

  wd_check      CommitmentBank and Humicroedit, linear heads, L2 grid extended to 100 and 1,000
  mlp_lr_check  MLP heads, soft-CORN / soft CE / soft-CORAL, learning rates 3e-5 ... 1e-3, evaluated every 50 steps
  finetune_lr_check  fine-tuning (n = 2,048), soft-CORN / soft-CORAL / soft CE, backbone learning rate 1e-4,
                above the main grid {1e-5, 3e-5}
  finetune_lpft      the same fine-tuning with the head first fitted on frozen features (LP-FT)

For each problem the main-grid selection is compared with the selection over the union of both grids
(validation NLL decides, as everywhere else), so "does a wider grid change the result?" is answered
on the quantity the benchmark reports: test NLL at the NLL-selected checkpoint.

It also measures hyperparameter sensitivity from the main runs' validation logs: how much of the
(learning rate, L2) grid lies within 0.01 and 0.05 nats of a method's best validation NLL.

    python scripts/07_checks.py
Writes results/check_wd.csv, check_mlp_lr.csv, check_finetune_lr.csv, check_finetune_init.csv,
sensitivity.csv and checks.md.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ordbench.paths import RUNS, TABLES  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ordbench.analysis import collect  # noqa: E402
from ordbench.losses import METHODS  # noqa: E402
from ordbench.train import LRS, WDS  # noqa: E402

KEY = ["dataset", "n_train", "seed", "method"]
LABEL = dict(zip(METHODS, ["CE (hard)", "CORN (hard)", "CORAL (hard)", "soft CE", "soft-CORAL", "soft-OR", "OLDL-CAD",
                           "OLDL-QFD", "OLDL-CJS", "PLL-softmax", "PLL-CORN", "soft-CORN"]))


def union_change(main, ext, methods):
    """Per problem: test NLL under the main grid, under the extended grid alone, and under the union."""
    a = main[(main.select_by == "nll") & main.method.isin(methods)].set_index(KEY)
    b = ext[(ext.select_by == "nll") & ext.method.isin(methods)].set_index(KEY)
    j = a.index.intersection(b.index)
    a, b = a.loc[j], b.loc[j]
    use_b = b.val_score.values < a.val_score.values
    u = a.copy()
    u.loc[use_b, ["nll", "lr", "wd", "step"]] = b.loc[use_b, ["nll", "lr", "wd", "step"]].values
    out = pd.DataFrame({"nll_main": a.nll, "nll_ext": b.nll, "nll_union": u.nll,
                        "lr_main": a.lr, "wd_main": a.wd, "lr_ext": b.lr, "wd_ext": b.wd, "step_ext": b.step,
                        "lr_union": u.lr, "wd_union": u.wd}, index=j)
    out["d_union"] = out.nll_union - out.nll_main
    return out.reset_index()


def sensitivity(run_dir, tols=(0.01, 0.05)):
    """Per (cell, method): share of the hyperparameter grid whose best validation NLL (over training
    steps) is within `tol` of the best configuration, and the median excess over the best."""
    import json
    rows = []
    for f in sorted(Path(run_dir).glob("*.npz")):
        with np.load(f, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            for method in meta["methods"]:
                v = np.nan_to_num(z[f"{method}/val/nll"], nan=np.inf).min(0)          # (configs,)
                row = dict(dataset=meta["dataset"], n_train=meta["n_train"], body=meta["body"], seed=meta["seed"],
                           method=method, median_excess=float(np.median(v - v.min())))
                row.update({f"within_{t:g}": float((v <= v.min() + t).mean()) for t in tols})
                rows.append(row)
    return pd.DataFrame(rows)


def finetune_check(base_tag, ext_tag, csv, title, ext_name):
    """Fine-tuning runs of `ext_tag` (another learning rate, or another head initialisation) against the
    main protocol `base_tag`: test NLL of the variant alone, and of the better of the two per (dataset,
    seed, method) by validation NLL, with soft-CORN minus soft-CORAL under each."""
    base_dir, ext_dir = RUNS / base_tag, RUNS / ext_tag
    if not (base_dir.exists() and ext_dir.exists() and any(ext_dir.glob("*.npz"))):
        return []
    base, ext = collect(base_dir), collect(ext_dir)
    methods = sorted(set(ext.method) & set(base.method))
    d = union_change(base, ext, methods)
    d["d_ext"] = d.nll_ext - d.nll_main
    d.to_csv(TABLES / csv, index=False)
    sel = d.groupby("method").agg(cells=("d_union", "size"), mean_change_alone=("d_ext", "mean"),
                                  mean_change_best_of_both=("d_union", "mean"),
                                  best_of_both_picks_variant=("nll_union", lambda s: float("nan")))
    pick = d.assign(picks=(d.nll_union != d.nll_main)).groupby("method").picks.mean()
    sel["best_of_both_picks_variant"] = pick
    sel.index = sel.index.map(lambda m: LABEL.get(m, m))
    out = [f"## {title} ({d.groupby(['dataset', 'n_train']).ngroups} problems, {d.seed.nunique()} seeds)\n",
           f"Change in test NLL (negative = better than the main protocol) when {ext_name} is used alone, and when "
           "the better of the two is chosen per dataset, seed and method on validation NLL:\n",
           sel.to_markdown(floatfmt=".4f"), ""]
    gap = d.pivot_table(index=["dataset", "n_train", "seed"], columns="method", values=["nll_main", "nll_ext", "nll_union"])
    if {"soft_corn", "soft_coral"} <= set(d.method):
        for tag, name in (("main", "main protocol"), ("ext", f"{ext_name} alone"), ("union", "better of the two")):
            diff = (gap[f"nll_{tag}"]["soft_corn"] - gap[f"nll_{tag}"]["soft_coral"]).groupby("dataset").mean()
            out.append(f"* soft-CORN minus soft-CORAL, test NLL, {name}: mean {diff.mean():+.4f} "
                       f"({', '.join(f'{k} {v:+.3f}' for k, v in diff.items())})")
        out.append("")
    return out


def finetune_best_of_all(tags=("finetune", "finetune_lr_check", "finetune_lpft", "finetune_lpft_tuned"),
                         csv="check_finetune_best_of_all.csv", title="Fine-tuning"):
    """The fairest fine-tuning comparison: every method may use every protocol variant (head
    initialisation, learning rate) and, per (dataset, training size, seed), the variant with the best
    validation NLL is read. soft-CORN minus the other soft losses under that selection."""
    frames = []
    for tag in tags:
        d = RUNS / tag
        if d.exists() and any(d.glob("*.npz")):
            c = collect(d)
            frames.append(c[c.select_by == "nll"].assign(variant=tag))
    if len(frames) < 2:
        return []
    df = pd.concat(frames, ignore_index=True)
    methods = ["soft_corn", "soft_coral", "ce_soft"]
    df = df[df.method.isin(methods)]
    # only seeds that every variant present for a (dataset, size) has: a method must not lose its best variant
    # merely because that variant was run on fewer seeds
    present = df.groupby(["dataset", "n_train"]).variant.agg(set)
    seeds_of = df.groupby(["dataset", "n_train", "variant"]).seed.agg(set)
    kept = []
    for (ds, n), variants in present.items():
        common = set.intersection(*(seeds_of[(ds, n, v)] for v in variants))
        kept.append(df[(df.dataset == ds) & (df.n_train == n) & df.seed.isin(common)])
    df = pd.concat(kept, ignore_index=True)
    best = df.loc[df.groupby(KEY).val_score.idxmin()]
    best.to_csv(TABLES / csv, index=False)
    tab = best.pivot_table(index=["dataset", "n_train"], columns="method", values="nll")
    out = [f"## {title}, best variant per method by validation NLL ({', '.join(frames_tag for frames_tag in tags if (RUNS / frames_tag).exists())}; "
           f"{best.seed.nunique()} seeds)\n",
           "Every method may use any head initialisation and backbone learning rate in the variants; for each dataset, "
           "training size and seed the variant with the best validation NLL is read.\n",
           tab[methods].to_markdown(floatfmt=".3f"), ""]
    for tier, sel in (("main size (2,048 items; CommitmentBank 840)", tab.index.get_level_values("n_train") > 512),
                      ("512 items", tab.index.get_level_values("n_train") == 512)):
        for b in ("soft_coral", "ce_soft"):
            diff = (tab["soft_corn"] - tab[b])[sel].dropna()
            if len(diff):
                out.append(f"* soft-CORN minus {LABEL[b]}, {tier}: mean {diff.mean():+.4f} "
                           f"({', '.join(f'{k[0]} {v:+.3f}' for k, v in diff.items())})")
    share = best.groupby(["method", "variant"]).size().unstack(fill_value=0)
    share = share.div(share.sum(1), axis=0)
    share.index = share.index.map(lambda m: LABEL.get(m, m))
    out += ["", "Share of (dataset, seed) cells in which each variant was selected:\n", share.to_markdown(floatfmt=".2f"), ""]
    return out


def main():
    lines =["# Grid-edge and sensitivity checks (auto-generated by scripts/07_checks.py)\n"]
    main_dir = RUNS / "real"
    real = collect(main_dir) if main_dir.exists() and any(main_dir.glob("*.npz")) else None

    wd_dir = RUNS / "wd_check"
    if real is not None and wd_dir.exists() and any(wd_dir.glob("*.npz")):
        ext = collect(wd_dir)
        methods = sorted(set(ext.method))
        d = union_change(real[real.body == "linear"], ext, methods)
        d.to_csv(TABLES / "check_wd.csv", index=False)
        sel = d.groupby("method").agg(cells=("d_union", "size"), mean_change=("d_union", "mean"),
                                      max_abs_change=("d_union", lambda s: s.abs().max()),
                                      main_at_wd_max=("wd_main", lambda s: (s == max(WDS)).mean()),
                                      ext_at_new_max=("wd_ext", lambda s: (s == ext.wd.max()).mean()),
                                      union_at_new_max=("wd_union", lambda s: (s == ext.wd.max()).mean()))
        lines += [f"## Extended L2 grid (to {ext.wd.max():g}); {d.dataset.nunique()} datasets, "
                  f"{d.seed.nunique()} seeds, {d.groupby(['dataset', 'n_train']).ngroups} problems per method\n",
                  "Change in test NLL (union of both grids minus main grid), and how often a selected configuration "
                  "sits at a grid maximum:\n", sel.to_markdown(floatfmt=".4f"), ""]
        gap = d.pivot_table(index=["dataset", "n_train", "seed"], columns="method",
                            values=["nll_main", "nll_union"]).dropna()
        if {"soft_corn", "soft_coral"} <= set(d.method):
            for tag in ("main", "union"):
                diff = (gap[f"nll_{tag}"]["soft_corn"] - gap[f"nll_{tag}"]["soft_coral"]).mean()
                lines.append(f"* soft-CORN minus soft-CORAL, test NLL, {tag} grid: {diff:+.4f}")
            lines.append("")

    lr_dir = RUNS / "mlp_lr_check"
    if real is not None and lr_dir.exists() and any(lr_dir.glob("*.npz")):
        ext = collect(lr_dir)
        methods = sorted(set(ext.method))
        d = union_change(real[real.body == "mlp"], ext, methods)
        d.to_csv(TABLES / "check_mlp_lr.csv", index=False)
        first = ext[(ext.select_by == "nll")].step.min()
        e = ext[ext.select_by == "nll"]
        sel = d.groupby("method").agg(cells=("d_union", "size"), mean_change=("d_union", "mean"),
                                      max_abs_change=("d_union", lambda s: s.abs().max()),
                                      main_at_lowest_lr=("lr_main", lambda s: (s == min(LRS)).mean()),
                                      ext_at_lowest_lr=("lr_ext", lambda s: (s == e.lr.min()).mean()),
                                      ext_at_highest_lr=("lr_ext", lambda s: (s == e.lr.max()).mean()))
        sel["ext_at_first_checkpoint"] = e.groupby("method").step.apply(lambda s: (s == first).mean())
        lines += [f"## Lower MLP learning rates ({e.lr.min():g} to {e.lr.max():g}, checkpoints every {first} steps); "
                  f"{d.groupby(['dataset', 'n_train']).ngroups} problems per method, {d.seed.nunique()} seeds\n",
                  "Change in test NLL (union of both grids minus main grid), and where the selected learning rate "
                  "sits:\n", sel.to_markdown(floatfmt=".4f"), ""]
        main_sel = real[(real.body == "mlp") & (real.select_by == "nll") & real.method.isin(methods)]
        lines += ["Share of main-grid selections at the lowest learning rate (3e-4) by method:\n",
                  main_sel.groupby("method").lr.apply(lambda s: (s == min(LRS)).mean()).to_frame("share")
                  .to_markdown(floatfmt=".2f"), ""]
    lines += finetune_check("finetune", "finetune_lr_check", "check_finetune_lr.csv",
                            "Fine-tuning learning rate above the main grid", "new rate")
    lines += finetune_check("finetune", "finetune_lpft", "check_finetune_init.csv",
                            "Head initialisation: linear probe then fine-tune (LP-FT) against the training-label marginal",
                            "LP-FT")
    lines += finetune_check("finetune", "finetune_lpft_tuned", "check_finetune_init_tuned.csv",
                            "Head initialisation: LP-FT with a tuned probe (L2 up to 100, two learning rates, 1,000 steps)",
                            "tuned LP-FT")
    lines += finetune_best_of_all()
    for tag in ("deberta", "roberta_large", "dinov2", "siglip"):
        lines += finetune_check(f"finetune_{tag}", f"finetune_{tag}_lpft", f"check_finetune_init_{tag}.csv",
                                f"Head initialisation on the {tag} backbone: LP-FT against the label marginal", "LP-FT")
        lines += finetune_best_of_all((f"finetune_{tag}", f"finetune_{tag}_lpft"), f"check_finetune_best_{tag}.csv",
                                      f"Fine-tuning {tag}")
    if real is not None:
        sens = sensitivity(main_dir)
        sens.to_csv(TABLES / "sensitivity.csv", index=False)
        for body, g in sens.groupby("body"):
            t = g.groupby("method")[["within_0.01", "within_0.05", "median_excess"]].mean()
            t.index = t.index.map(lambda m: LABEL.get(m, m))
            lines += [f"## Hyperparameter sensitivity, {body} heads ({g.groupby(['dataset', 'n_train', 'seed']).ngroups} "
                      "cells; share of the (lr, L2) grid within 0.01 / 0.05 nats of the best validation NLL, "
                      "median excess validation NLL over the grid)\n", t.to_markdown(floatfmt=".3f"), ""]
    (TABLES / "checks.md").write_text("\n".join(lines), encoding="utf8")
    print("\n".join(lines))
    if len(lines) == 1:
        print("no check runs found under", RUNS)


if __name__ == "__main__":
    main()
