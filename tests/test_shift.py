"""Leave-one-group-out splits for the distribution-shift experiment, and the extra evaluation splits in selection."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ordbench.analysis import select  # noqa: E402
from ordbench.groups import leave_one_group_out  # noqa: E402


def test_leave_one_group_out_is_disjoint_complete_and_reproducible():
    groups = np.array(["a"] * 500 + ["b"] * 300 + ["c"] * 200)
    pool, va, idte, ood = leave_one_group_out(groups, "b", seed=3)
    assert set(groups[ood]) == {"b"} and len(ood) == 300
    others = np.concatenate([pool, va, idte])
    assert set(groups[others]) == {"a", "c"} and len(others) == 700 and len(set(others)) == 700
    assert (len(idte), len(va), len(pool)) == (140, 70, 490)
    again = leave_one_group_out(groups, "b", seed=3)
    assert all(np.array_equal(x, y) for x, y in zip((pool, va, idte, ood), again))
    other_seed = leave_one_group_out(groups, "b", seed=4)
    assert not np.array_equal(pool, other_seed[0])


def test_out_of_distribution_set_is_capped():
    groups = np.array(["x"] * 100 + ["y"] * 9000)
    _, _, _, ood = leave_one_group_out(groups, "y", seed=0, max_ood=8000)
    assert len(ood) == 8000 and set(groups[ood]) == {"y"} and (np.diff(ood) > 0).all()


def test_select_reads_extra_splits(tmp_path):
    steps, M = np.array([100, 200]), 2
    rng = np.random.default_rng(0)
    arrays = {"steps": steps}
    meta = dict(dataset="d", m=3, seed=0, n_train=10, pool=10, body="linear", K=4, n_val=5, n_test=5, methods=["soft_corn"],
                configs=[[1e-3, 1.0], [1e-2, 1.0]], prior=None)
    for split in ("val", "test", "idtest"):
        for k in ("nll", "set_mae_med", "rps"):
            arrays[f"soft_corn/{split}/{k}"] = rng.random((2, M)) + (10 if split == "idtest" else 0)
    np.savez(tmp_path / "cell.npz", meta=json.dumps(meta), **arrays)
    rows = select(tmp_path / "cell.npz")
    row = [r for r in rows if r["select_by"] == "nll"][0]
    assert "nll" in row and 10 < row["idtest_nll"] < 11 and "val_nll" not in row
