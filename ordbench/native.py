"""Natively set-valued ordinal labels (FUTURE_WORK item 3): datasets in which a person accepts several levels at once,
such as the accepted proactivity levels of the thesis's in-vehicle study, instead of sets built from independent raters.

`build_native` turns a table with one row per labelled instance (a situation, or a user-situation pair) into the
processed feature file that scripts/03_run.py reads with `--m 0`:

    columns   id (optional; row number by default), a set column listing the accepted levels ("1|3|4", 0- or 1-based),
              the features (either columns f0, f1, ... or a .npy file with one row per table row), optionally a
              group column (situation id, user id) that must not be split across training and test items
    output    $SCB_DATA/processed/<name>.npz with X, K, item_id, mask_m0 (accepted sets), ho_m0 (no held-out raters: zeros),
              valid_m0, group, and empty rating lists, so every runner, metric and analysis works unchanged. Without
              the ratings behind the sets, the empirical-distribution and exact-likelihood targets are not available,
              and the held-out-rater metrics are skipped.

    python -m ordbench.native --csv study.csv --name vehicle --K 5 --features features.npy --group situation
    python scripts/03_run.py --datasets vehicle --m 0 --n_train full --bodies linear mlp
"""
import argparse

import numpy as np
import pandas as pd

from .paths import PROCESSED


def parse_sets(values, K, base=0, sep="|"):
    """Boolean (n, K) mask from strings such as "1|3|4" (levels numbered from `base`)."""
    mask = np.zeros((len(values), K), bool)
    for i, v in enumerate(values):
        levels = [int(t) - base for t in str(v).split(sep) if t.strip() != ""]
        if not levels or min(levels) < 0 or max(levels) >= K:
            raise ValueError(f"row {i}: accepted levels {v!r} are not within 0..{K - 1} (base {base})")
        mask[i, levels] = True
    return mask


def build_native(csv, name, K, features=None, set_col="accepted", id_col=None, group_col=None, base=0, sep="|",
                 feature_prefix="f", out=None):
    df = pd.read_csv(csv)
    mask = parse_sets(df[set_col], K, base, sep)
    if features is not None:
        X = np.load(features)
        if len(X) != len(df):
            raise ValueError(f"{len(X)} feature rows for {len(df)} table rows")
    else:
        cols = [c for c in df.columns if c.startswith(feature_prefix) and c[len(feature_prefix):].isdigit()]
        if not cols:
            raise ValueError(f"no feature columns {feature_prefix}0, {feature_prefix}1, ... and no features file")
        X = df[cols].to_numpy()
    item_id = df[id_col].to_numpy() if id_col else np.arange(len(df))
    group = df[group_col].astype(str).to_numpy() if group_col else None
    return save_native(name, X, mask, K, item_id, group, out)


def save_native(name, X, mask, K, item_id=None, group=None, out=None, extra=None):
    """Write the processed file for `mask` (n, K) accepted sets and features `X` (n, d); see the module docstring."""
    n = len(X)
    item_id = np.arange(n) if item_id is None else np.asarray(item_id)
    group = item_id.astype(str) if group is None else np.asarray(group).astype(str)
    path = out or PROCESSED / f"{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=X.astype(np.float32), K=K, item_id=item_id, ratings_flat=np.zeros(0, np.int64),
                        ratings_ptr=np.zeros(n + 1, np.int64), group=group, mask_m0=mask,
                        ho_m0=np.zeros((n, K), np.int32), valid_m0=np.ones(n, bool), **(extra or {}))
    sizes = mask.sum(1)
    print(f"[{name}] rows={n} groups={len(set(group))} d={X.shape[1]} K={K} set size mean={sizes.mean():.2f} "
          f"singletons={np.mean(sizes == 1):.0%} gapped={np.mean([np.ptp(np.flatnonzero(r)) + 1 > r.sum() for r in mask]):.0%}")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--features", help=".npy file with one feature row per table row (default: columns f0, f1, ...)")
    ap.add_argument("--set_col", default="accepted")
    ap.add_argument("--id_col")
    ap.add_argument("--group_col", help="rows of one group are kept together when splitting into train / validation / test")
    ap.add_argument("--base", type=int, default=0, help="number of the lowest level in the file (0 or 1)")
    ap.add_argument("--sep", default="|")
    a = ap.parse_args()
    build_native(a.csv, a.name, a.K, a.features, a.set_col, a.id_col, a.group_col, a.base, a.sep)


if __name__ == "__main__":
    main()
