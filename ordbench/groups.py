"""Item groups for distribution-shift experiments (leave one group out).

The processed feature files keep `item_id` (the raw comment id, revision id or image file name); the groups are read
from the raw releases and aligned to that order, so the splits of the shift experiment use the same items and set labels
as the main benchmark.

    mhs       platform code of the comment (youtube has only 70 items and is never held out)
    wikitox   namespace of the page (article / user talk) and the year of the comment (terciles: early / mid / late)
    scut      demographic subset of the face (AF / AM Asian female / male, CF / CM Caucasian female / male)
"""
import numpy as np
import pandas as pd

from .paths import RAW

SCHEMES = {"mhs": {"platform": ["0", "2", "3"]},
           "wikitox": {"ns": ["article", "user"], "year": ["early", "mid", "late"]},
           "scut": {"subset": ["AF", "AM", "CF", "CM"]}}


def item_groups(name, scheme, item_id):
    """Group label (str) of every item in `item_id`, for one of the schemes in SCHEMES[name]."""
    if name == "mhs":
        df = pd.read_parquet(RAW / "mhs" / "measuring-hate-speech.parquet", columns=["comment_id", "platform"])
        g = df.groupby("comment_id")["platform"].first()
        return np.array([str(int(g.loc[i])) for i in item_id])
    if name == "wikitox":
        com = pd.read_csv(RAW / "wikitox" / "toxicity_annotated_comments.tsv", sep="\t", index_col=0)
        if scheme == "ns":
            return com.loc[item_id, "ns"].to_numpy().astype(str)
        year = com.loc[item_id, "year"].to_numpy()
        lo, hi = np.quantile(year, [1 / 3, 2 / 3])
        return np.where(year <= lo, "early", np.where(year <= hi, "mid", "late"))
    if name == "scut":
        return np.array([str(f)[:2] for f in item_id])
    raise ValueError(f"no groups for {name}")


def leave_one_group_out(groups, held_out, seed, test=0.2, val=0.1, max_ood=8000):
    """Indices (into the items) of the training pool, the in-distribution validation and test sets, and the
    out-of-distribution test set. Everything outside the held-out group is split at random into pool / validation /
    in-distribution test (20% / 10% as in the main benchmark); the held-out group is the out-of-distribution test
    set, a random subset of at most `max_ood` items."""
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    ood = np.flatnonzero(groups == held_out)
    if len(ood) > max_ood:
        ood = np.sort(rng.choice(ood, max_ood, replace=False))
    rest = rng.permutation(np.flatnonzero(groups != held_out))
    nt, nv = round(test * len(rest)), round(val * len(rest))
    return rest[nt + nv:], rest[nt:nt + nv], rest[:nt], ood
