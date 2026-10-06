"""End-to-end checks that run on CPU in CI: the splits are reproducible, and a tiny synthetic run
goes through the real runner script and back out of the offline model selection."""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from ordbench.analysis import CRITERIA, collect  # noqa: E402


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), REPO / "scripts" / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_splits_are_reproducible_and_disjoint():
    """Results from different machines can only be merged if every machine draws the same splits."""
    run03 = _load_script("03_run.py")
    te, va, pool = run03.split(1000, seed=3)
    assert (len(te), len(va), len(pool)) == (200, 100, 700)
    assert len(set(te) | set(va) | set(pool)) == 1000
    te2, va2, pool2 = run03.split(1000, seed=3)
    assert np.array_equal(te, te2) and np.array_equal(va, va2) and np.array_equal(pool, pool2)
    # the first permuted items of seed 0 are fixed by numpy's PCG64 generator on every platform
    assert run03.split(10, seed=0)[0].tolist() == np.random.default_rng(0).permutation(10)[:2].tolist()


def test_synthetic_runner_and_selection_smoke(tmp_path):
    methods = ["soft_corn", "soft_coral", "ce_soft"]
    cmd = [sys.executable, str(REPO / "scripts" / "04_synthetic.py"), "--families", "cr", "--seeds", "0",
           "--n_train", "128", "--bodies", "linear", "--methods", *methods, "--n_val", "100",
           "--n_test", "200", "--steps", "40", "--eval_every", "20", "--out", str(tmp_path)]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "SCB_DATA": str(tmp_path / "data")}
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=REPO)
    assert proc.returncode == 0, proc.stderr[-2000:]
    files = list(tmp_path.glob("cr_m3_s0_n128_linear.npz"))
    assert len(files) == 1
    df = collect(tmp_path)
    assert sorted(df.method.unique()) == sorted(methods)
    assert len(df) == len(methods) * len(CRITERIA)
    assert np.isfinite(df[["nll", "set_mae_med", "rps", "kl_t", "kl_pi"]].to_numpy()).all()
    assert (df.kl_t > -1e-4).all()                  # excess risk against the exact target is non-negative
    assert set(df.step) <= {20, 40}
