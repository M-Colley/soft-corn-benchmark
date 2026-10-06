import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ordbench import synthetic as S  # noqa: E402


@pytest.mark.parametrize("family", S.FAMILIES)
def test_rater_pmf_valid(family):
    X, mask, pi = S.make(family, 500, 3, seed=0)
    assert pi.shape == (500, 5) and np.allclose(pi.sum(1), 1) and (pi >= 0).all()
    assert mask.any(1).all() and (mask.sum(1) <= 3).all()


def test_set_target_matches_monte_carlo():
    rng = np.random.default_rng(0)
    pi = rng.dirichlet(np.ones(5), size=4)
    t = S.set_target(pi, 3)
    assert np.allclose(t.sum(1), 1)
    for i in range(4):
        r = S.sample_ratings(np.repeat(pi[i:i + 1], 200_000, 0), 3, rng)
        mask = S.sets_from_ratings(r, 5)
        mc = (mask / mask.sum(1, keepdims=True)).mean(0)
        assert np.abs(mc - t[i]).max() < 5e-3


def test_set_target_of_point_mass_is_point_mass():
    pi = np.eye(5)
    assert np.allclose(S.set_target(pi, 3), pi)


def test_problem_parameters_shared_across_splits():
    _, _, pi_a = S.make("cr", 10, 3, seed=1, param_seed=7)
    X_b, _, pi_b = S.make("cr", 10, 3, seed=1, param_seed=7)
    assert np.allclose(pi_a, pi_b)
    _, _, pi_c = S.make("cr", 10, 3, seed=1, param_seed=8)
    assert not np.allclose(pi_a, pi_c)
