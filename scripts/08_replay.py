"""Replay finished real-data cells and keep the predictions at the selected checkpoints.

The benchmark logs metrics for every (step, configuration) but not the predictions, because selection is
done offline. Everything that needs per-item predictions (paired item-level tests, conformal prediction
sets, reliability diagrams, post-hoc temperature scaling) is computed from this replay: each cell is
trained again exactly as in scripts/03_run.py (same split, standardisation, batch order, initialisation)
and the validation and test log-PMFs of the member and step that the NLL criterion selected are stored.
Heads are trained independently per member, so the replay reproduces the logged metrics; the file
records the largest deviation between the replayed and the logged test NLL as a check.

    python scripts/08_replay.py --bodies linear
Output: $SCB_DATA/results/replay/<cell>.npz (finished cells are skipped)
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ordbench.paths import RUNS  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ordbench.analysis import select  # noqa: E402
from ordbench.train import make_evalset, run_cell  # noqa: E402

_spec = importlib.util.spec_from_file_location("run03", Path(__file__).with_name("03_run.py"))
run03 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run03)


def replay(path, crit="nll", device="cuda"):
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        steps = z["steps"]
    name, m, seed, n, body = meta["dataset"], meta["m"], meta["seed"], meta["n_train"], meta["body"]
    configs = [tuple(c) for c in meta["configs"]]
    rows = {r["method"]: r for r in select(path) if r["select_by"] == crit}
    X, mask, ho, K, cnt = run03.load(name, m)
    te, va, pool = run03.split(len(X), seed, groups=run03.load_groups(name, m))
    tr = pool[:n]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Z = (X - mu) / sd
    evalsets = {"val": make_evalset(Z[va], mask[va], device=device, m=m),
                "test": make_evalset(Z[te], mask[te], ho=ho[te], device=device, m=m)}
    capture = {meth: [(configs.index((r["lr"], r["wd"])), r["step"])] for meth, r in rows.items()}
    res, _ = run_cell(Z[tr], mask[tr], evalsets, K, meta["methods"], body=body, seed=seed, steps=int(steps[-1]),
                      eval_every=int(steps[0]), configs=configs, device=device, counts_tr=cnt[tr], m=m,
                      capture=capture)
    out = dict(meta=json.dumps(meta), va=va, te=te, test_mask=mask[te], test_ho=ho[te], val_mask=mask[va],
               val_counts=cnt[va], test_counts=cnt[te], steps=steps)
    worst = 0.0
    for meth, r in rows.items():
        j, step = capture[meth][0]
        for split in ("val", "test"):
            out[f"{meth}/{split}"] = capture[("pmf", meth, split, j, step)].astype(np.float32)
        out[f"{meth}/sel"] = np.array([r["lr"], r["wd"], r["step"]])
        e = int(np.where(steps == step)[0][0])
        worst = max(worst, abs(float(res[meth]["test"]["nll"][e, j]) - float(r["nll"])))
    out["max_nll_deviation"] = np.array(worst)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(RUNS / "real"))
    ap.add_argument("--out", default=str(RUNS / "replay"))
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--bodies", nargs="+", default=["linear"])
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--crit", default="nll")
    a = ap.parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    for f in sorted(Path(a.src).glob("*.npz")):
        ds, _, s, _, b = f.stem.split("_")
        if (a.datasets and ds not in a.datasets) or b not in a.bodies or (a.seeds and int(s[1:]) not in a.seeds):
            continue
        dest = out_dir / f.name
        if dest.exists():
            continue
        t0 = time.time()
        res = replay(f, a.crit, dev)
        np.savez_compressed(dest, **res)
        print(f"{f.name}  {time.time() - t0:.0f}s  max |test NLL replay - log| = {float(res['max_nll_deviation']):.2e}",
              flush=True)


if __name__ == "__main__":
    main()
