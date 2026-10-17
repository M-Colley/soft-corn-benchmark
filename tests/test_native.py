"""Natively set-valued labels go through the loader, the runner and the selection unchanged."""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from ordbench.analysis import collect  # noqa: E402
from ordbench.native import parse_sets  # noqa: E402


def _run03():
    spec = importlib.util.spec_from_file_location("run03_native", REPO / "scripts" / "03_run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_parse_sets_numbering_and_errors():
    assert parse_sets(["1|3", "0"], 4).tolist() == [[False, True, False, True], [True, False, False, False]]
    assert parse_sets(["1|3", "4"], 4, base=1).tolist() == [[True, False, True, False], [False, False, False, True]]
    for bad in ("5", "", "-1|2"):
        with pytest.raises(ValueError):
            parse_sets([bad], 4)


def test_group_split_keeps_groups_together_and_is_reproducible():
    run03 = _run03()
    groups = np.repeat(np.arange(100), 4)                    # 100 situations rated by 4 users each
    te, va, pool = run03.split(len(groups), 2, groups=groups)
    parts = [set(groups[x]) for x in (te, va, pool)]
    assert not (parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2])
    assert len(te) + len(va) + len(pool) == 400 and len(parts[0]) == 20 and len(parts[1]) == 10
    assert all(np.array_equal(a, b) for a, b in zip((te, va, pool), run03.split(len(groups), 2, groups=groups)))
    assert not np.array_equal(pool, np.sort(pool))           # the pool is shuffled, so nested subsets are random


def test_group_split_fractions_can_be_set_for_few_groups():
    """The ProVoice study has 12 drivers: 25% / 17% of the groups are 3 test and 2 validation drivers."""
    run03 = _run03()
    groups = np.repeat(np.arange(12), 7)
    te, va, pool = run03.split(len(groups), 0, 0.25, 0.17, groups=groups)
    assert [len(set(groups[x])) for x in (te, va, pool)] == [3, 2, 7]


def test_native_table_through_the_runner(tmp_path):
    g = np.random.default_rng(0)
    n, d, K = 240, 6, 5
    X = g.normal(size=(n, d))
    centre = np.clip(np.round(2 + X[:, 0] * 1.2 + g.normal(0, 0.5, n)), 0, K - 1).astype(int)
    width = g.integers(0, 2, n)
    sets = ["|".join(str(k) for k in range(max(0, c - w), min(K - 1, c + w) + 1)) for c, w in zip(centre, width)]
    df = pd.DataFrame({"situation": np.repeat(np.arange(n // 4), 4), "accepted": [s for s in sets]})
    df[[f"f{j}" for j in range(d)]] = X
    csv = tmp_path / "study.csv"
    df.to_csv(csv, index=False)
    data = tmp_path / "data"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "SCB_DATA": str(data)}
    build = [sys.executable, "-m", "ordbench.native", "--csv", str(csv), "--name", "toy", "--K", str(K),
             "--group_col", "situation"]
    proc = subprocess.run(build, capture_output=True, text=True, env=env, cwd=REPO)
    assert proc.returncode == 0, proc.stderr[-2000:]
    z = np.load(data / "processed" / "toy.npz", allow_pickle=True)
    assert z["mask_m0"].shape == (n, K) and z["X"].shape == (n, d) and z["ho_m0"].sum() == 0
    out = tmp_path / "runs"
    run = [sys.executable, str(REPO / "scripts" / "03_run.py"), "--datasets", "toy", "--m", "0", "--seeds", "0",
           "--n_train", "full", "--bodies", "linear", "--methods", "soft_corn", "soft_coral", "ce_soft", "--steps", "40",
           "--eval_every", "20", "--out", str(out)]
    proc = subprocess.run(run, capture_output=True, text=True, env=env, cwd=REPO)
    assert proc.returncode == 0, proc.stderr[-2000:]
    df = collect(out)
    assert sorted(df.method.unique()) == ["ce_soft", "soft_coral", "soft_corn"]
    assert np.isfinite(df[["nll", "set_mae_med", "rps"]].to_numpy()).all()
    assert "ho_nll" not in df or df.ho_nll.isna().all()     # no held-out raters in natively set-valued data
