"""Personalisation (FUTURE_WORK item 12) on SCUT-FBP5500, where all 60 raters rated all 5,500 faces.

Task: predict a given rater's rating of a face. The raters are split into known raters (45), whose ratings are used for
training, and held-out raters (15), who are never seen. Faces are split as in scripts/03_run.py. Models are heads on the
frozen OpenCLIP features, optionally with a learned offset of the output logits per known rater (rater bias, shrunk
by the L2 penalty; the held-out raters get a zero offset). A training row is one (face, known rater) pair, and each rater
contributes a random `--ratings` faces of the training pool.

The soft target of a row mixes the personal rating with the population consensus of the face, the uniform
distribution over the set of ratings of m = 3 known raters (the set-valued label of the main benchmark):
    target = (1 - alpha) * onehot(own rating) + alpha * u_set,     alpha in {0, 0.5, 1}
Hard-label losses (CE, CORN, CORAL) use the own rating only. Checkpoints are selected on validation faces x known
raters; reported are the test faces x known raters (`test`) and the test faces x held-out raters (`newrater`).

    python scripts/13_personal.py
Output: $SCB_DATA/results/personal/scut_<offsets|none>_a<alpha>_r<ratings>_s<seed>.npz (finished cells are skipped)
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ordbench.paths import PROCESSED, RAW, RUNS  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from ordbench.losses import Target  # noqa: E402
from ordbench.train import CONFIGS, flatten, make_evalset, run_cell  # noqa: E402

_spec = importlib.util.spec_from_file_location("run03", Path(__file__).with_name("03_run.py"))
run03 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run03)

N_KNOWN, M_SET = 45, 3
HARD = ["ce_hard", "corn_hard", "coral_hard"]
SOFT = ["ce_soft", "soft_corn", "soft_coral"]


def rating_matrix(item_id):
    """(items, raters) ratings 0..4 in the order of the processed SCUT items."""
    cache = PROCESSED / "scut_matrix.npz"
    if not cache.exists():
        df = pd.read_excel(RAW / "scut" / "SCUT-FBP5500_v2" / "All_Ratings.xlsx")
        mat = df.pivot(index="Filename", columns="Rater", values="Rating")
        np.savez_compressed(cache, R=mat.to_numpy() - 1, files=mat.index.to_numpy().astype(str))
    z = np.load(cache, allow_pickle=True)
    pos = {f: i for i, f in enumerate(z["files"])}
    return z["R"][[pos[str(f)] for f in item_id]].astype(np.int64)


def onehot(y, K):
    return np.eye(K, dtype=np.float32)[y]


def pair_target(items, raters, R, S, K, alpha):
    """Target of the (item, rater) pairs: own rating mixed with the population consensus set of the item."""
    own = onehot(R[items, raters], K)
    consensus = S[items] / S[items].sum(1, keepdims=True)
    p = (1 - alpha) * own + alpha * consensus
    y = R[items, raters]
    mask = own > 0
    return Target(torch.as_tensor(p), torch.as_tensor(mask), torch.as_tensor(y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--ratings", nargs="+", default=["50", "400", "full"], help="training faces per known rater")
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.5, 1.0])
    ap.add_argument("--out", default=str(RUNS / "personal"))
    a = ap.parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True

    z = np.load(PROCESSED / "scut.npz", allow_pickle=True)
    X, K = z["X"], int(z["K"])
    R = rating_matrix(z["item_id"])
    n_items, n_raters = R.shape
    for seed in a.seeds:
        te, va, pool = run03.split(n_items, seed)
        rng = np.random.default_rng(seed)
        order = rng.permutation(n_raters)
        known, new = np.sort(order[:N_KNOWN]), np.sort(order[N_KNOWN:])
        slot = {int(r): i for i, r in enumerate(known)}                  # offset-table row of every known rater
        # population consensus: the set of ratings of M_SET random known raters, fixed per item
        S = np.zeros((n_items, K), np.float32)
        for i in range(n_items):
            S[i, R[i, rng.choice(known, M_SET, replace=False)]] = 1
        mu, sd = X[pool].mean(0), X[pool].std(0) + 1e-6
        Z = (X - mu) / sd

        def pairs(items, raters):
            it = np.repeat(items, len(raters))
            rt = np.tile(raters, len(items))
            return it, rt

        vi, vr = pairs(va, known)
        ti, tr_ = pairs(te, known)
        ni, nr = pairs(te, new)

        def evalset(items, raters, aux):
            rating = R[items, raters]
            return make_evalset(Z[items], onehot(rating, K) > 0, device=dev, aux=aux)

        for ratings in a.ratings:
            per = len(pool) if ratings == "full" else int(ratings)
            sel = [(i, r) for r in known for i in rng.choice(pool, min(per, len(pool)), replace=False)]
            ii, rr = np.array([s[0] for s in sel]), np.array([s[1] for s in sel])
            aux_known = np.array([slot[int(r)] for r in rr])
            for alpha in a.alphas:
                for offsets in ("offsets", "none"):
                    f = out_dir / f"scut_{offsets}_a{alpha:g}_r{ratings}_s{seed}.npz"
                    methods = SOFT if alpha > 0 else HARD + SOFT
                    if f.exists():
                        continue
                    t0 = time.time()
                    use = offsets == "offsets"
                    evalsets = {"val": evalset(vi, vr, np.array([slot[int(r)] for r in vr]) if use else None),
                                "test": evalset(ti, tr_, np.array([slot[int(r)] for r in tr_]) if use else None),
                                "newrater": evalset(ni, nr, np.full(len(nr), N_KNOWN) if use else None)}
                    res, steps = run_cell(Z[ii], None, evalsets, K, methods, body="linear", seed=seed, device=dev,
                                          steps=2000, eval_every=100, aux_tr=aux_known if use else None,
                                          n_aux=N_KNOWN + 1 if use else 0, target_tr=pair_target(ii, rr, R, S, K, alpha))
                    meta = dict(dataset="scut-personal", m=M_SET, seed=seed, n_train=len(ii), pool=len(pool) * N_KNOWN, body="linear",
                                K=K, n_val=len(vi), n_test=len(ti), configs=CONFIGS, methods=methods, prior=None,
                                offsets=use, alpha=alpha, ratings_per_rater=ratings, known=len(known), new=len(new))
                    np.savez_compressed(f, steps=steps, meta=json.dumps(meta), **flatten(res))
                    print(f"{f.name}  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
