"""Synthetic set-valued ordinal data with a known population target.

Each item x ~ N(0, I_d) has a rater distribution pi(x) over K levels drawn from one of five
families. Its set-valued label is, by default, the set of distinct levels chosen by m independent raters
from pi(x), the same mechanism used to build the real-data labels. Every soft loss trained on
uniform-over-set labels u_S has the same population minimiser,
    t(x) = E[u_S | x],
which is computed exactly (by enumerating the sets that can occur and their probabilities). Excess
risk KL(t || p_hat) is then a noise-free measure of how well a loss + parameterisation recovers the target.

Families (logits linear in x by default, so the comparison is about the output parameterisation):
    po        proportional odds, P(y > k) = sigma(w.x - theta_k)       (CORAL's model)
    cr        continuation ratio with level-specific weights            (CORN's model)
    softmax   multinomial logit                                         (softmax's model)
    locscale  discretised normal latent with x-dependent mean and spread
    bimodal   x-dependent mixture of a consensus peak and two polarised peaks,
              so raters often disagree across the scale (gapped sets)

Variations (all default to the settings above, which reproduce the original generator exactly):
    K          number of levels (3-10; the polarised mixture builds its templates for any K)
    d, signal  input dimension and the scale of the predictors (signal-to-noise of the problem)
    nonlinear  the predictors are a quarter linear and three quarters pairwise interactions, which a linear head
               cannot represent
    eps, shift label noise: every rating is uniform with probability eps, and shifted by -1 / 0 / +1
               levels with probabilities `shift` (rater heterogeneity); pi(x) is the resulting
               distribution of a single rating, so the sets stay exactly characterised
    mechanism  how the set is produced from the latent level y ~ pi(x):
                 raters     the set of m independent ratings (default)
                 interval   [y - w, y + w] clipped to the scale, with an x-dependent half-width w in {0, 1, 2}
                 threshold  every level from y upwards (a user who accepts everything above a personal threshold)
                 gapped     {y, K - 1 - y}, an adversarial mirror set that is gapped by construction
"""
import itertools

import numpy as np
from scipy.special import expit, log_softmax
from scipy.stats import norm

FAMILIES = ("po", "cr", "softmax", "locscale", "bimodal")
MECHANISMS = ("raters", "interval", "threshold", "gapped")
HIDDEN = 16                       # interaction terms per predictor


def _unit(rng, d, scale):
    return rng.normal(size=d) * scale / np.sqrt(d)


class _Predictor:
    """(n, n_out) predictors of x. Linear case: a linear map with N(0, scale^2 / d) weights (for n_out = 1 it draws
    exactly the numbers of `_unit`, so the original generator is reproduced). Nonlinear case: the same scale, made of
    a quarter of linear signal and three quarters of pairwise interactions (x.u_j)(x.v_j), which are uncorrelated with
    any linear function of Gaussian inputs, so a linear head can recover at most the linear quarter."""

    def __init__(self, rng, d, n_out, scale, nonlinear=False):
        self.nonlinear = nonlinear
        if nonlinear:
            self.W = rng.normal(size=(d, n_out)) * 0.5 * scale / np.sqrt(d)
            self.U, self.V = (rng.normal(size=(d, HIDDEN)) for _ in range(2))
            self.U /= np.linalg.norm(self.U, axis=0)
            self.V /= np.linalg.norm(self.V, axis=0)
            self.C = rng.normal(size=(HIDDEN, n_out)) * scale * np.sqrt(0.75 / HIDDEN)
        else:
            self.W = rng.normal(size=(d, n_out)) * scale / np.sqrt(d)

    def __call__(self, X):
        if self.nonlinear:
            return X @ self.W + ((X @ self.U) * (X @ self.V)) @ self.C
        return X @ self.W


def _bimodal_templates(K):
    if K == 5:
        return np.array([0.55, 0.30, 0.10, 0.04, 0.01]), np.array([0.05, 0.20, 0.50, 0.20, 0.05])
    k = np.arange(K)
    low = 0.55 ** k
    mid = np.exp(-0.5 * ((k - (K - 1) / 2) / (0.45 * K ** 0.5)) ** 2)
    return low / low.sum(), mid / mid.sum()


def rater_pmf(family, X, rng, K=5, signal=1.0, nonlinear=False, eps=0.0, shift=None):
    """pi(x) for every row of X. Family parameters are drawn from `rng`."""
    n, d = X.shape
    if family == "po":
        s = _Predictor(rng, d, 1, 3.0 * signal, nonlinear)(X)[:, 0]
        theta = np.linspace(-2.5, 2.5, K - 1)
        gt = expit(s[:, None] - theta)                       # P(y > k)
        pi = np.concatenate([1 - gt[:, :1], gt[:, :-1] - gt[:, 1:], gt[:, -1:]], 1)
    elif family == "cr":
        z = _Predictor(rng, d, K - 1, 1.5 * signal, nonlinear)(X)
        b = np.log((K - 1 - np.arange(K - 1)) / 1.0)         # uniform marginal at x = 0
        h = expit(z + b)                                     # P(y > k | y >= k)
        surv = np.concatenate([np.ones((n, 1)), np.cumprod(h, 1)], 1)
        pi = surv * np.concatenate([1 - h, np.ones((n, 1))], 1)
    elif family == "softmax":
        pi = np.exp(log_softmax(_Predictor(rng, d, K, 2.0 * signal, nonlinear)(X), 1))
    elif family == "locscale":
        mu = (K - 1) / 2 + 1.2 * signal * _Predictor(rng, d, 1, 1.0, nonlinear)(X)[:, 0]
        sd = 0.8 * np.exp(0.8 * signal * _Predictor(rng, d, 1, 1.0, nonlinear)(X)[:, 0])
        cdf = norm.cdf((np.arange(K - 1) + 0.5 - mu[:, None]) / sd[:, None])
        pi = np.diff(np.concatenate([np.zeros((n, 1)), cdf, np.ones((n, 1))], 1), axis=1)
    elif family == "bimodal":
        low, mid = _bimodal_templates(K)
        a = expit(2.0 * signal * _Predictor(rng, d, 1, 1.0, nonlinear)(X)[:, 0])[:, None]       # which pole dominates
        c = expit(2.0 * signal * _Predictor(rng, d, 1, 1.0, nonlinear)(X)[:, 0] - 0.5)[:, None]  # consensus vs polarised
        pi = c * mid + (1 - c) * (a * low + (1 - a) * low[::-1])
    else:
        raise ValueError(family)
    pi = np.clip(pi, 0, None)
    pi = pi / pi.sum(1, keepdims=True)
    return _noisy(pi, eps, shift)


def _noisy(pi, eps, shift):
    """Distribution of a single rating after rater heterogeneity (shift) and label noise (eps)."""
    if shift is not None:
        down, stay, up = shift
        lower = np.concatenate([pi[:, 1:], np.zeros((len(pi), 1))], 1)       # mass shifted down one level
        upper = np.concatenate([np.zeros((len(pi), 1)), pi[:, :-1]], 1)
        out = stay * pi + down * lower + up * upper
        out[:, 0] += down * pi[:, 0]                                         # clipped at the ends
        out[:, -1] += up * pi[:, -1]
        pi = out
    if eps:
        pi = (1 - eps) * pi + eps / pi.shape[1]
    return pi


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
    """t(x) = E[u_S | x] exactly for sets of m independent ratings, by enumerating all K^m rating tuples."""
    K = pi.shape[1]
    tuples = np.array(list(itertools.product(range(K), repeat=m)))           # (K^m, m)
    U = sets_from_ratings(tuples, K).astype(float)
    U /= U.sum(1, keepdims=True)                                             # u_S per tuple
    prob = np.ones((len(pi), len(tuples)))
    for j in range(m):
        prob *= pi[:, tuples[:, j]]
    return prob @ U


# --------------------------------------------------------------------------------------------
# Set mechanisms other than m independent raters
# --------------------------------------------------------------------------------------------

def _mechanism_sets(mechanism, K):
    """The (latent level y, auxiliary index w) combinations of a mechanism and the set each produces (bool (K,))."""
    out = []
    if mechanism == "interval":
        for y in range(K):
            for w in range(3):
                s = np.zeros(K, bool)
                s[max(0, y - w):min(K - 1, y + w) + 1] = True
                out.append((y, w, s))
    elif mechanism == "threshold":
        for y in range(K):
            s = np.zeros(K, bool)
            s[y:] = True
            out.append((y, 0, s))
    elif mechanism == "gapped":
        for y in range(K):
            s = np.zeros(K, bool)
            s[[y, K - 1 - y]] = True
            out.append((y, 0, s))
    else:
        raise ValueError(mechanism)
    return out


def _width_probs(X, vw):
    """P(half-width w | x) for the interval mechanism: softmax over w of a linear score (an x-dependent width)."""
    return np.exp(log_softmax(np.array([1.0, 0.0, -1.0]) + X @ vw, 1))


def mechanism_sets(mechanism, pi, X, vw, rng=None):
    """Sample one set per item (rng given) or return the exact target E[u_S | x] (rng None)."""
    n, K = pi.shape
    combos = _mechanism_sets(mechanism, K)
    wp = _width_probs(X, vw) if mechanism == "interval" else None
    probs = np.stack([pi[:, y] * (wp[:, w] if wp is not None else 1.0) for y, w, _ in combos], 1)      # (n, combos)
    sets = np.stack([s for _, _, s in combos])                                                         # (combos, K)
    if rng is None:
        return probs @ (sets / sets.sum(1, keepdims=True))
    pick = (rng.random((n, 1)) > np.cumsum(probs, 1)[:, :-1]).sum(1)
    return sets[pick]


def make(family, n, m, seed, d=10, K=5, param_seed=None, return_ratings=False, signal=1.0, nonlinear=False,
         eps=0.0, shift=None, mechanism="raters", return_target=False):
    """Draw n items. The family parameters depend only on `param_seed` (defaults to seed),
    so train / val / test sets of one problem share the same pi(.). With return_ratings the (n, m)
    ratings behind the sets are returned as a fourth value (mechanism "raters" only); with return_target
    the exact target E[u_S | x] is appended."""
    prng = np.random.default_rng(10_000 + (seed if param_seed is None else param_seed))
    drng = np.random.default_rng(seed)
    X = drng.normal(size=(n, d))
    pi = rater_pmf(family, X, prng, K, signal, nonlinear, eps, shift)
    out = [X.astype(np.float32)]
    if mechanism == "raters":
        ratings = sample_ratings(pi, m, drng)
        out += [sets_from_ratings(ratings, K), pi]
        if return_ratings:
            out.append(ratings)
        if return_target:
            out.append(set_target(pi, m))
        return tuple(out)
    vw = prng.normal(size=(d, 3)) * 1.0 / np.sqrt(d)          # drawn after the family parameters
    out += [mechanism_sets(mechanism, pi, X, vw, drng), pi]
    if return_ratings:
        raise ValueError("sets from this mechanism are not built from ratings")
    if return_target:
        out.append(mechanism_sets(mechanism, pi, X, vw))
    return tuple(out)
