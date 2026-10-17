"""The generalised synthetic generator reproduces the original one by default, and its new options and set
mechanisms have exactly the targets that sampling produces."""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit, log_softmax
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ordbench import synthetic as S  # noqa: E402


def _unit(rng, d, scale):
    return rng.normal(size=d) * scale / np.sqrt(d)


def original_rater_pmf(family, X, rng, K=5):
    """The generator of the first benchmark run, verbatim."""
    n, d = X.shape
    if family == "po":
        s = X @ _unit(rng, d, 3.0)
        theta = np.linspace(-2.5, 2.5, K - 1)
        gt = expit(s[:, None] - theta)
        pi = np.concatenate([1 - gt[:, :1], gt[:, :-1] - gt[:, 1:], gt[:, -1:]], 1)
    elif family == "cr":
        W = rng.normal(size=(d, K - 1)) * 1.5 / np.sqrt(d)
        b = np.log((K - 1 - np.arange(K - 1)) / 1.0)
        h = expit(X @ W + b)
        surv = np.concatenate([np.ones((n, 1)), np.cumprod(h, 1)], 1)
        pi = surv * np.concatenate([1 - h, np.ones((n, 1))], 1)
    elif family == "softmax":
        B = rng.normal(size=(d, K)) * 2.0 / np.sqrt(d)
        pi = np.exp(log_softmax(X @ B, 1))
    elif family == "locscale":
        mu = (K - 1) / 2 + 1.2 * (X @ _unit(rng, d, 1.0))
        sd = 0.8 * np.exp(0.8 * (X @ _unit(rng, d, 1.0)))
        cdf = norm.cdf((np.arange(K - 1) + 0.5 - mu[:, None]) / sd[:, None])
        pi = np.diff(np.concatenate([np.zeros((n, 1)), cdf, np.ones((n, 1))], 1), axis=1)
    else:
        low = np.array([0.55, 0.30, 0.10, 0.04, 0.01])
        mid = np.array([0.05, 0.20, 0.50, 0.20, 0.05])
        a = expit(2.0 * (X @ _unit(rng, d, 1.0)))[:, None]
        c = expit(2.0 * (X @ _unit(rng, d, 1.0)) - 0.5)[:, None]
        pi = c * mid + (1 - c) * (a * low + (1 - a) * low[::-1])
    pi = np.clip(pi, 0, None)
    return pi / pi.sum(1, keepdims=True)


@pytest.mark.parametrize("family", S.FAMILIES)
def test_default_generator_reproduces_the_original(family):
    X = np.random.default_rng(0).normal(size=(300, 10))
    new = S.rater_pmf(family, X, np.random.default_rng(7))
    old = original_rater_pmf(family, X, np.random.default_rng(7))
    assert np.allclose(new, old, atol=1e-12)


@pytest.mark.parametrize("family", S.FAMILIES)
@pytest.mark.parametrize("K,nonlinear,signal", [(3, False, 1.0), (8, False, 1.0), (10, True, 1.0), (5, True, 2.0),
                                                (5, False, 0.5)])
def test_variants_give_valid_distributions(family, K, nonlinear, signal):
    X = np.random.default_rng(1).normal(size=(200, 12))
    pi = S.rater_pmf(family, X, np.random.default_rng(3), K=K, signal=signal, nonlinear=nonlinear)
    assert pi.shape == (200, K) and (pi >= 0).all() and np.allclose(pi.sum(1), 1)


def test_nonlinear_predictor_has_the_scale_of_the_linear_one_and_is_mostly_not_linear():
    X = np.random.default_rng(2).normal(size=(40000, 10))
    lin = S._Predictor(np.random.default_rng(0), 10, 1, 3.0)(X)
    net = S._Predictor(np.random.default_rng(0), 10, 1, 3.0, nonlinear=True)(X)
    assert 0.5 < net.std() / lin.std() < 2.0
    Xc = np.c_[X, np.ones(len(X))]
    coef, *_ = np.linalg.lstsq(Xc, net[:, 0], rcond=None)
    r2 = 1 - np.var(net[:, 0] - Xc @ coef) / np.var(net[:, 0])
    assert r2 < 0.5                                       # a linear head sees at most the linear quarter


def test_noise_and_heterogeneity_keep_a_distribution_and_move_mass_as_stated():
    pi = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0]])
    out = S._noisy(pi, 0.0, (0.2, 0.6, 0.2))
    assert np.allclose(out.sum(1), 1)
    assert np.allclose(out[0], [0.8, 0.2, 0, 0])            # down-shift clipped at level 0
    assert np.allclose(out[1], [0, 0, 0.2, 0.8])            # up-shift clipped at the top level
    assert np.allclose(out[2], [0.2, 0.6, 0.2, 0])
    assert np.allclose(S._noisy(pi, 0.4, None)[0], [0.7, 0.1, 0.1, 0.1])


@pytest.mark.parametrize("mechanism", S.MECHANISMS[1:])
def test_mechanism_target_matches_sampling(mechanism):
    K, d = 5, 4
    rng = np.random.default_rng(0)
    x = rng.normal(size=d)
    X = np.tile(x, (60000, 1))
    pi = np.tile(rng.dirichlet(np.ones(K)), (60000, 1))
    vw = rng.normal(size=(d, 3))
    sets = S.mechanism_sets(mechanism, pi, X, vw, np.random.default_rng(1))
    emp = (sets / sets.sum(1, keepdims=True)).mean(0)
    exact = S.mechanism_sets(mechanism, pi[:1], X[:1], vw)[0]
    assert np.abs(emp - exact).max() < 0.01
    assert np.isclose(exact.sum(), 1)


def test_mechanism_sets_have_the_advertised_shapes():
    K = 6
    pi = np.full((4000, K), 1 / K)
    X = np.zeros((4000, 3))
    vw = np.zeros((3, 3))
    thr = S.mechanism_sets("threshold", pi, X, vw, np.random.default_rng(0))
    assert all(row[np.argmax(row):].all() for row in thr)                        # contiguous upper sets
    gap = S.mechanism_sets("gapped", pi, X, vw, np.random.default_rng(0))
    assert all(np.array_equal(row, row[::-1]) for row in gap)                    # mirror sets
    iv = S.mechanism_sets("interval", pi, X, vw, np.random.default_rng(0))
    assert all(np.diff(np.flatnonzero(row)).max(initial=1) == 1 for row in iv)   # no gaps


def test_make_target_and_options_are_consistent():
    X, mask, pi, t = S.make("cr", 500, 3, seed=2, K=7, d=6, return_target=True)
    assert X.shape == (500, 6) and mask.shape == (500, 7) and np.allclose(t.sum(1), 1)
    assert np.allclose(t, S.set_target(pi, 3))
    X, mask, pi, t = S.make("po", 300, 3, seed=2, mechanism="interval", return_target=True)
    assert np.allclose(t.sum(1), 1) and mask.any(1).all()
    with pytest.raises(ValueError):
        S.make("po", 10, 3, seed=0, mechanism="gapped", return_ratings=True)
