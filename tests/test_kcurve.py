"""The K-shot split must give the test driver and the validation drivers the same K calibration answers and no leakage."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ordbench import kcurve as KC  # noqa: E402


def make_rows(n_drivers=6, per_driver=20):
    drivers = np.repeat([f"{i + 1:03d}" for i in range(n_drivers)], per_driver)
    windows = np.array([f"{d}|win{(i % per_driver) // 2:03d}" for i, d in enumerate(drivers)])     # two prompts per window
    return drivers, windows


def test_split_roles_and_disjointness():
    drivers, windows = make_rows()
    train, val, test, cal, val_drivers, personal = KC.kshot_split(drivers, windows, "003", 0, 5)
    assert personal[0] == "003" and len(val_drivers) == 2 and "003" not in val_drivers
    assert set(drivers[test]) == {"003"} and set(drivers[val]) == set(val_drivers)
    assert not (set(train) & set(val)) and not (set(train) & set(test)) and not (set(val) & set(test))
    full = [d for d in set(drivers) if d not in personal]
    assert all((drivers[train] == d).sum() == 20 for d in full)                                  # full drivers in training
    assert all((drivers[train] == d).sum() == 5 for d in personal)                               # K answers of the others
    assert all(len(cal[d]) == 5 for d in personal)


def test_calibration_prefixes_are_nested_and_evaluation_has_no_twin_of_a_calibration_window():
    drivers, windows = make_rows()
    splits = {k: KC.kshot_split(drivers, windows, "004", 1, k) for k in (0, 2, 5, 10)}
    for small, big in ((0, 2), (2, 5), (5, 10)):
        for d in splits[small][5]:
            assert set(splits[small][3][d]) <= set(splits[big][3][d])
        assert splits[small][4] == splits[big][4]                                               # same validation drivers
    for k, (train, val, test, cal, val_drivers, personal) in splits.items():
        for d in personal:
            used = set(windows[cal[d]])
            evaluated = set(windows[np.concatenate([val, test])][np.isin(drivers[np.concatenate([val, test])], [d])])
            assert not (used & evaluated)
    # K = 0 evaluates every answer of the personalised drivers
    train, val, test, cal, *_ = splits[0]
    assert len(test) == 20


def test_zero_code_for_k_zero_only_and_personal_prior_interpolates():
    drivers, windows = make_rows()
    code = np.eye(6, dtype=np.float32)[np.searchsorted(sorted(set(drivers)), drivers)]
    train, val, test, cal, val_drivers, personal = KC.kshot_split(drivers, windows, "002", 0, 0)
    c0 = KC.driver_code(code, drivers, personal, 0)
    assert c0[np.isin(drivers, personal)].sum() == 0 and c0[~np.isin(drivers, personal)].sum() == (~np.isin(drivers, personal)).sum()
    assert np.array_equal(KC.driver_code(code, drivers, personal, 5), code)
    soft = np.zeros((len(drivers), 3))
    soft[np.arange(len(drivers)), 2] = 1                                                          # everyone answers level 2
    pop = np.array([0.6, 0.3, 0.1])
    _, _, test5, cal5, _, _ = KC.kshot_split(drivers, windows, "002", 0, 5)
    lp = KC.personal_prior(soft, drivers, test5, cal5, pop, alpha=5.0)[0]
    assert np.allclose(np.exp(lp[0]), (np.array([0, 0, 5.0]) + 5 * pop) / 10)
    lp0 = KC.personal_prior(soft, drivers, test, {d: np.array([], int) for d in personal}, pop, alpha=5.0)[0]
    assert np.allclose(np.exp(lp0[0]), pop)


def test_collapse_sets_keeps_one_level_inside_each_set():
    mask = np.array([[0, 1, 1, 0, 0], [1, 0, 0, 0, 0], [0, 1, 1, 1, 0], [1, 1, 0, 0, 0]], bool)
    lower, upper = KC.collapse_sets(mask, "lower"), KC.collapse_sets(mask, "upper")
    assert lower.sum(1).tolist() == [1, 1, 1, 1] and upper.sum(1).tolist() == [1, 1, 1, 1]
    assert [int(np.flatnonzero(r)[0]) for r in lower] == [1, 0, 2, 0]
    assert [int(np.flatnonzero(r)[0]) for r in upper] == [2, 0, 2, 1]
    rand = KC.collapse_sets(mask, "random", 3)
    assert rand.sum(1).tolist() == [1, 1, 1, 1] and not (rand & ~mask).any()
