"""Synthetic set-valued ordinal data with a known population target.

Each item x ~ N(0, I_d) has a rater distribution pi(x) over K levels drawn from one of five
families. Its set-valued label is the set of distinct levels chosen by m independent raters
from pi(x), the same mechanism used to build the real-data labels. Every soft loss trained on
uniform-over-set labels u_S has the same population minimiser,
    t(x) = E[u_S | x],
which is computed exactly by enumerating all K^m rating tuples. Excess risk KL(t || p_hat) is
then a noise-free measure of how well a loss + parameterisation recovers the target.

Families (all logits linear in x, so the comparison is about the output parameterisation):
    po        proportional odds, P(y > k) = sigma(w.x - theta_k)       (CORAL's model)
    cr        continuation ratio with level-specific weights            (CORN's model)
    softmax   multinomial logit                                         (softmax's model)
    locscale  discretised normal latent with x-dependent mean and spread
    bimodal   x-dependent mixture of a consensus peak and two polarised peaks,
              so raters often disagree across the scale (gapped sets)
"""
import itertools

import numpy as np
from scipy.special import expit, log_softmax
from scipy.stats import norm

FAMILIES = ("po", "cr", "softmax", "locscale", "bimodal")


def _unit(rng, d, scale):
    return rng.normal(size=d) * scale / np.sqrt(d)


def rater_pmf(family, X, rng, K=5):
    """pi(x) for every row of X. Family parameters are drawn from `rng`."""
    n, d = X.shape
    if family == "po":
        s = X @ _unit(rng, d, 3.0)
        theta = np.linspace(-2.5, 2.5, K - 1)
        gt = expit(s[:, None] - theta)                       # P(y > k)
        pi = np.concatenate([1 - gt[:, :1], gt[:, :-1] - gt[:, 1:], gt[:, -1:]], 1)
    elif family == "cr":
        W = rng.normal(size=(d, K - 1)) * 1.5 / np.sqrt(d)
        b = np.log((K - 1 - np.arange(K - 1)) / 1.0)         # uniform marginal at x = 0
        h = expit(X @ W + b)                                 # P(y > k | y >= k)
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
    elif family == "bimodal":
        low = np.array([0.55, 0.30, 0.10, 0.04, 0.01])
        mid = np.array([0.05, 0.20, 0.50, 0.20, 0.05])
        assert K == 5
        a = expit(2.0 * (X @ _unit(rng, d, 1.0)))[:, None]   # which pole dominates
        c = expit(2.0 * (X @ _unit(rng, d, 1.0)) - 0.5)[:, None]  # consensus vs polarised
        pi = c * mid + (1 - c) * (a * low + (1 - a) * low[::-1])
    else:
        raise ValueError(family)
    pi = np.clip(pi, 0, None)
    return pi / pi.sum(1, keepdims=True)


def sample_ratings(pi, m, rng):
    """(n, m) integer ratings, m independent draws per row from pi."""
    c = np.cumsum(pi, 1)
    u = rng.random((len(pi), m, 1))
    return (u > c[:, None, :-1]).sum(-1)


def sets_from_ratings(r, K):
    mask = np.zeros((len(r), K), bool)
    np.put_along_axis(mask, r, True, axis=1)
    return mask


def set_target(pi, m):
    """t(x) = E[u_S | x] exactly, by enumerating all K^m rating tuples."""
    K = pi.shape[1]
    tuples = np.array(list(itertools.product(range(K), repeat=m)))           # (K^m, m)
    U = sets_from_ratings(tuples, K).astype(float)
    U /= U.sum(1, keepdims=True)                                             # u_S per tuple
    prob = np.ones((len(pi), len(tuples)))
    for j in range(m):
        prob *= pi[:, tuples[:, j]]
    return prob @ U


def make(family, n, m, seed, d=10, K=5, param_seed=None):
    """Draw n items. The family parameters depend only on `param_seed` (defaults to seed),
    so train / val / test sets of one problem share the same pi(.)."""
    prng = np.random.default_rng(10_000 + (seed if param_seed is None else param_seed))
    drng = np.random.default_rng(seed)
    X = drng.normal(size=(n, d))
    pi = rater_pmf(family, X, prng, K)
    mask = sets_from_ratings(sample_ratings(pi, m, drng), K)
    return X.astype(np.float32), mask, pi
