"""Losses compared in the benchmark.

Every loss acts on the raw outputs of one of several output parameterisations and on a batch
`Target` that carries the set-valued label in equivalent views:

    p     (N, K)  uniform distribution over the accepted set (the soft label)
    mask  (N, K)  boolean indicator of the accepted set
    y     (N,)    hard label obtained by collapsing the set to its lower median
    c     (N, K)  optional: counts of the m ratings the set was built from (empirical-distribution and
                  exact-likelihood targets); m is the number of raters per set (0 if unknown)

Parameterisations (`HEAD_OF[name]`):
    softmax  K logits, PMF = softmax
    corn     K-1 conditional logits, P(y>k | y>k-1) = sigmoid(z_k)            (Shi et al., 2023)
    coral    K-1 logits s(x) + b_k with one shared weight vector              (Cao et al., 2020)
    or       K-1 independent logits for P(y>k), no rank constraint            (Niu et al., 2016)
    binom    one score, PMF = Binomial(K-1, sigmoid(s))                       (unimodal; Beckham & Pal, 2017)
    poisson  one score, PMF = truncated Poisson(exp(s))                       (unimodal; Beckham & Pal, 2017)
    clm_*    cumulative link model, P(y>k) = F(s(x) + b_k) with ordered thresholds b_k and link F in
             {logit, probit, cloglog}; clm_ao learns the link (Aranda-Ordaz family, one extra parameter)

Hard-label CORAL and soft-CORAL / soft-OR call `coral_pytorch.coral_loss`; hard-label CORN uses
an identity with soft-CORN that tests/test_losses.py checks against `coral_pytorch.corn_loss`. The OLDL
objectives port the reference implementation of Wen et al. (ICCV 2023, github.com/downdric/OLDL,
MIT) from its hard-coded K in {5, 10, 85} to arbitrary K.

`METHODS` is the benchmark's core set of twelve; `TARGET_METHODS` and `BASELINE_METHODS` are the
extensions that vary the training target (empirical ratings, exact set likelihood, mixtures) and add
further baselines (SORD, CDW-CE, unimodal heads, cumulative link models). They are run into their own
result folders so the core benchmark keeps its twelve methods.
"""
import itertools
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from coral_pytorch.losses import coral_loss

EPS = 1e-7


@dataclass
class Target:
    p: torch.Tensor                 # (N, K) float
    mask: torch.Tensor              # (N, K) bool
    y: torch.Tensor                 # (N,) long
    c: torch.Tensor | None = None   # (N, K) float counts of the m ratings behind each set
    m: int = 0                      # raters per set (0: unknown)

    def index(self, idx):
        return Target(self.p[idx], self.mask[idx], self.y[idx], None if self.c is None else self.c[idx], self.m)

    def to(self, device):
        return Target(self.p.to(device), self.mask.to(device), self.y.to(device),
                      None if self.c is None else self.c.to(device), self.m)

    @property
    def empirical(self):
        """Empirical distribution of the m ratings behind each set."""
        if self.c is None:
            raise ValueError("this loss needs the ratings behind each set (Target.c)")
        return self.c / self.c.sum(-1, keepdim=True)


# --------------------------------------------------------------------------------------------
# Decoding outputs to a PMF
# --------------------------------------------------------------------------------------------

def corn_log_pmf(z):
    """log P(y=k) for CORN conditional logits z (..., K-1), computed in log space.

    log p_k = sum_{j<k} log sigma(z_j) + log(1 - sigma(z_k)),  log p_{K-1} = sum_{j<K-1} log sigma(z_j).
    """
    ls = F.logsigmoid(z)
    lns = F.logsigmoid(-z)
    zeros = torch.zeros_like(z[..., :1])
    cum = torch.cat([zeros, torch.cumsum(ls, dim=-1)], dim=-1)          # log q_{k-1}, k = 0..K-1
    return cum + torch.cat([lns, zeros], dim=-1)


def cumulative_to_pmf(gt):
    """P(y>k) (..., K-1) -> PMF (..., K). Differences can be negative for models without a
    rank-consistency guarantee (OR; CORAL with unordered biases); they are clipped and the
    PMF renormalised, as is standard when such models are scored with proper scoring rules."""
    ones = torch.ones_like(gt[..., :1])
    zeros = torch.zeros_like(gt[..., :1])
    upper = torch.cat([ones, gt], dim=-1)
    lower = torch.cat([gt, zeros], dim=-1)
    pmf = (upper - lower).clamp_min(EPS)
    return pmf / pmf.sum(-1, keepdim=True)


def binom_log_pmf(s, K):
    """Binomial(K-1, q) over levels 0..K-1 with q = sigmoid(s); s (..., 1)."""
    k = torch.arange(K, device=s.device, dtype=s.dtype)
    log_coef = math.lgamma(K) - torch.lgamma(k + 1) - torch.lgamma(K - k)        # log C(K-1, k)
    return log_coef + k * F.logsigmoid(s) + (K - 1 - k) * F.logsigmoid(-s)


def poisson_log_pmf(s, K):
    """Poisson(exp(s)) truncated to levels 0..K-1 and renormalised; s (..., 1)."""
    k = torch.arange(K, device=s.device, dtype=s.dtype)
    return F.log_softmax(k * s - torch.lgamma(k + 1), dim=-1)


def link_cdf(name, z, lam=None):
    """Inverse link F(z) of a cumulative link model: P(y > k) = F(s + b_k)."""
    if name == "logit":
        return torch.sigmoid(z)
    if name == "probit":
        return 0.5 * (1 + torch.erf(z / 2 ** 0.5))
    if name == "cloglog":
        return -torch.expm1(-torch.exp(z.clamp(max=20)))
    if name == "ao":                                   # Aranda-Ordaz: logit at lam = 1, cloglog as lam -> 0
        return -torch.expm1(-torch.log1p(lam * torch.exp(z.clamp(max=20))) / lam)
    raise ValueError(name)


def log_pmf(head, out, K=None):
    """Log-PMF (..., K) of raw outputs `out` of a head type. The single-score unimodal heads need K."""
    if head == "softmax":
        return F.log_softmax(out, dim=-1)
    if head == "corn":
        return corn_log_pmf(out)
    if head in ("coral", "or"):
        return cumulative_to_pmf(torch.sigmoid(out)).log()
    if head in ("binom", "poisson"):
        if K is None:
            raise ValueError(f"{head} needs the number of levels K")
        return (binom_log_pmf if head == "binom" else poisson_log_pmf)(out, K)
    if head.startswith("clm_"):
        link = head[4:]
        if link == "ao":
            z, lam = out[..., :-1], F.softplus(out[..., -1:]) + 1e-3
            return cumulative_to_pmf(link_cdf("ao", z, lam)).log()
        return cumulative_to_pmf(link_cdf(link, out)).log()
    raise ValueError(head)


def rank_violation(head, out):
    """Fraction of examples whose implied P(y>k) is not non-increasing in k."""
    if head not in ("coral", "or"):
        return torch.zeros(out.shape[:-1], device=out.device)
    gt = torch.sigmoid(out)
    return (gt[..., 1:] > gt[..., :-1] + 1e-6).any(-1).float()


# --------------------------------------------------------------------------------------------
# Proposed loss
# --------------------------------------------------------------------------------------------

def soft_corn_loss(z, p, normalise="example"):
    """soft-CORN: CORN with the hard subset membership 1[y >= k] and indicator 1[y > k] replaced
    by their expectations q_{k-1} = P(y >= k) and q_k = P(y > k) under the label distribution p.

        L = - sum_{k=0}^{K-2} [ q_k log sigma(z_k) + p_k log(1 - sigma(z_k)) ]

    normalise="example": mean over examples. Equals the cross-entropy between p and the
                         CORN-decoded PMF (a categorical likelihood).
    normalise="corn":    divide by the soft size of the conditional training sets,
                         sum_i sum_k q_{i,k-1}; equals coral_pytorch.corn_loss exactly on one-hot p.
    """
    q = 1.0 - torch.cumsum(p, dim=-1)[..., :-1]          # q_k = P(y > k), k = 0..K-2
    per = -(q * F.logsigmoid(z) + p[..., :-1] * F.logsigmoid(-z)).sum(-1)
    if normalise == "example":
        return per.mean(-1)
    if normalise == "corn":
        q_prev = torch.cat([torch.ones_like(q[..., :1]), q[..., :-1]], dim=-1)   # P(y >= k)
        return per.sum(-1) / q_prev.sum((-1, -2))
    raise ValueError(normalise)


# --------------------------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------------------------

def soft_coral_loss(logits, p):
    """soft-CORAL / soft-OR (Matton et al., 2026, Eq. 2): CORAL's binary cross-entropy with
    the extended binary label 1[y > k] replaced by p^{>k} = sum_{j>k} p_j. Delegates to
    coral_pytorch.coral_loss, which accepts fractional `levels`."""
    soft = (1.0 - torch.cumsum(p, dim=-1)[..., :-1]).expand(logits.shape)
    flat = coral_loss(logits.reshape(-1, logits.shape[-1]), soft.reshape(-1, soft.shape[-1]),
                      reduction=None)
    return flat.reshape(logits.shape[:-1]).mean(-1)


def ce_soft_loss(logits, p):
    """Soft-label cross-entropy (= KL label-distribution learning up to a constant)."""
    return -(p * F.log_softmax(logits, dim=-1)).sum(-1).mean(-1)


def pmf_ce_loss(logp, p):
    """Cross-entropy between the soft label p and any predicted log-PMF (the likelihood of every
    proper soft loss on its own parameterisation)."""
    return -(p * logp).sum(-1).mean(-1)


def ce_hard_loss(logits, y):
    flat = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.expand(logits.shape[:-1]).reshape(-1),
                           reduction="none")
    return flat.reshape(logits.shape[:-1]).mean(-1)


def levels(y, K):
    """Extended binary labels 1[y > k], k = 0..K-2: coral_pytorch.dataset.levels_from_labelbatch
    without its per-example .item() loop (equality is checked in tests/test_losses.py)."""
    return (torch.arange(K - 1, device=y.device) < y[:, None]).float()


def one_hot(y, K):
    """F.one_hot without the device synchronisation, so it can run inside a CUDA graph."""
    return (torch.arange(K, device=y.device) == y[:, None]).float()


def coral_hard_loss(logits, y):
    K = logits.shape[-1] + 1
    levels_ = levels(y, K).to(logits).expand(logits.shape)
    flat = coral_loss(logits.reshape(-1, K - 1), levels_.reshape(-1, K - 1), reduction=None)
    return flat.reshape(logits.shape[:-1]).mean(-1)


def corn_hard_loss(logits, y):
    """coral_pytorch.corn_loss, vectorised over ensemble members via the identity
    corn_loss(z, y) == soft_corn_loss(z, onehot(y), normalise="corn") (checked against the
    library in tests/test_losses.py)."""
    K = logits.shape[-1] + 1
    return soft_corn_loss(logits, one_hot(y, K).to(logits.dtype), normalise="corn")


def _cdf(pmf):
    return torch.cumsum(pmf, dim=-1)


def oldl_cad_loss(logits, p):
    """OLDL cumulative absolute distance (Wen et al., 2023; CAD.py): sum_k |F_hat_k - F_k|."""
    return (_cdf(F.softmax(logits, -1)) - _cdf(p)).abs().sum(-1).mean(-1)


def qfd_matrix(K, device=None, dtype=torch.float32):
    """A_ij = 1 - |i - j| / K, the similarity matrix used in QFDloss.py for K = 5, 10, 85."""
    i = torch.arange(K, device=device, dtype=dtype)
    return 1.0 - (i[:, None] - i[None, :]).abs() / K


def oldl_qfd_loss(logits, p):
    """OLDL quadratic form distance (QFDloss.py): |p_hat - p| A |p_hat - p|^T. The reference
    code takes the absolute difference before the quadratic form; kept as published."""
    d = (F.softmax(logits, -1) - p).abs()
    A = qfd_matrix(p.shape[-1], p.device, d.dtype)
    return torch.einsum("...k,kl,...l->...", d, A, d).mean(-1)


def oldl_cjs_loss(logits, p):
    """OLDL cumulative Jensen-Shannon (CJS.py): sum_k JS terms between cumulative values."""
    PD = _cdf(F.softmax(logits, -1)).clamp_min(EPS)
    GT = _cdf(p)
    M = (PD + GT) / 2
    t = 0.5 * PD * (PD.log() - M.log()) + 0.5 * GT * ((GT + 1e-8).log() - M.log())
    return t.sum(-1).mean(-1)


def pll_loss(logp, mask):
    """Partial-label (superset) likelihood: -log P(y in S) = -log sum_{k in S} p_hat_k
    (Jin & Ghahramani, 2002; Cour et al., 2011). Treats the set as a coarsened single label."""
    masked = logp.masked_fill(~mask, float("-inf"))
    return -torch.logsumexp(masked, dim=-1).mean(-1)


# --------------------------------------------------------------------------------------------
# Further baselines: soft ordinal targets, distance-weighted CE
# --------------------------------------------------------------------------------------------

def sord_matrix(K, power=1, device=None, dtype=torch.float32):
    """Row j is the SORD target of level j: softmax_k(-|k - j|^power) (Diaz & Marathe, 2019; power 1 is
    the Laplace-shaped label distribution, power 2 the Gaussian-shaped one)."""
    i = torch.arange(K, device=device, dtype=dtype)
    d = (i[:, None] - i[None, :]).abs() ** power
    return F.softmax(-d if power == 1 else -d / 2, dim=-1)


def sord_loss(logits, p, power=1):
    """SORD / Gaussian label-distribution cross-entropy with a set-valued label: the SORD target
    of every level in the set, averaged (the soft label p convolved with the ordinal kernel)."""
    target = p @ sord_matrix(p.shape[-1], power, p.device, p.dtype)
    return ce_soft_loss(logits, target)


def cdwce_loss(logits, p, alpha=2.0):
    """Class-distance-weighted cross-entropy (Polat et al., 2022): -sum_k |k - y|^alpha log(1 - p_k),
    averaged over the levels y of the set."""
    K = p.shape[-1]
    i = torch.arange(K, device=p.device, dtype=p.dtype)
    w = p @ ((i[:, None] - i[None, :]).abs() ** alpha)           # (N, K): expected distance weights
    probs = F.softmax(logits, -1)
    return -(w * torch.log1p(-probs.clamp(max=1 - EPS))).sum(-1).mean(-1)


# --------------------------------------------------------------------------------------------
# Exact set likelihood (sets built from m independent raters)
# --------------------------------------------------------------------------------------------

_SUBSETS = {}


def subset_tables(K, device):
    """Indicator matrix A (2^K - 1, K) of all non-empty subsets of the K levels, and their sizes."""
    key = (K, str(device))
    if key not in _SUBSETS:
        bits = (torch.arange(1, 2 ** K, device=device)[:, None] >> torch.arange(K, device=device)) & 1
        _SUBSETS[key] = (bits.bool(), bits.sum(-1))
    return _SUBSETS[key]


def exact_set_logp(logp, mask, m):
    """log P(S | pi_hat): probability that m independent ratings from the predicted PMF produce exactly
    the observed set S, by inclusion-exclusion, P(S) = sum_{T subset S} (-1)^{|S|-|T|} pi(T)^m.
    logp (..., N, K), mask (N, K) -> (..., N). Singletons give pi_k^m."""
    K = logp.shape[-1]
    A, size = subset_tables(K, logp.device)
    mass = logp.exp() @ A.T.to(logp.dtype)                                   # pi(T), (..., N, 2^K - 1)
    inside = ~((A[None] & ~mask[:, None, :]).any(-1))                        # T subset of S, (N, 2^K - 1)
    sign = 1.0 - 2.0 * ((mask.sum(-1)[:, None] - size[None, :]) % 2).to(logp.dtype)
    total = (inside * sign * mass.clamp_min(0) ** m).sum(-1)
    return total.clamp_min(1e-12).log()


def exact_set_loss(logp, mask, m):
    return -exact_set_logp(logp, mask, m).mean(-1)


def brute_force_set_prob(pmf, m):
    """P(set of m iid ratings == S) for every set S, by enumerating all K^m rating tuples (used by tests).
    pmf (K,) -> {frozenset: probability}."""
    out = {}
    for tup in itertools.product(range(len(pmf)), repeat=m):
        pr = 1.0
        for k in tup:
            pr *= float(pmf[k])
        key = frozenset(tup)
        out[key] = out.get(key, 0.0) + pr
    return out


# --------------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------------

CORE_METHODS = {
    "ce_hard": "softmax", "corn_hard": "corn", "coral_hard": "coral",
    "ce_soft": "softmax", "soft_coral": "coral", "soft_or": "or",
    "oldl_cad": "softmax", "oldl_qfd": "softmax", "oldl_cjs": "softmax",
    "pll_softmax": "softmax", "pll_corn": "corn",
    "soft_corn": "corn",
}

# training targets other than uniform-over-set (need Target.c / Target.m)
TARGET_METHODS = {
    "soft_corn_emp": "corn", "ce_emp": "softmax", "soft_coral_emp": "coral",
    "exact_corn": "corn", "exact_ce": "softmax", "exact_coral": "coral",
    "soft_corn_mix50": "corn", "ce_mix50": "softmax", "soft_coral_mix50": "coral",
}

BASELINE_METHODS = {
    "sord": "softmax", "sord_sq": "softmax", "cdwce": "softmax",
    "binom": "binom", "poisson": "poisson",
    "clm_logit": "clm_logit", "clm_probit": "clm_probit", "clm_cloglog": "clm_cloglog", "clm_ao": "clm_ao",
}

HEAD_OF = {**CORE_METHODS, **TARGET_METHODS, **BASELINE_METHODS}
METHODS = list(CORE_METHODS)
ALL_METHODS = list(HEAD_OF)

# EnsembleHead output layer of every head type
OUTPUT_OF = {"softmax": "full", "corn": "full", "or": "full", "coral": "coral", "binom": "coral",
             "poisson": "coral", "clm_logit": "clm", "clm_probit": "clm", "clm_cloglog": "clm", "clm_ao": "clm_ao"}


def compute_loss(method, out, t: Target):
    """Per-member loss, shape out.shape[:-2] (scalar for a single model)."""
    if method == "soft_corn":
        return soft_corn_loss(out, t.p)
    if method in ("soft_coral", "soft_or"):
        return soft_coral_loss(out, t.p)
    if method == "ce_soft":
        return ce_soft_loss(out, t.p)
    if method == "ce_hard":
        return ce_hard_loss(out, t.y)
    if method == "coral_hard":
        return coral_hard_loss(out, t.y)
    if method == "corn_hard":
        return corn_hard_loss(out, t.y)
    if method == "oldl_cad":
        return oldl_cad_loss(out, t.p)
    if method == "oldl_qfd":
        return oldl_qfd_loss(out, t.p)
    if method == "oldl_cjs":
        return oldl_cjs_loss(out, t.p)
    if method == "pll_softmax":
        return pll_loss(F.log_softmax(out, -1), t.mask)
    if method == "pll_corn":
        return pll_loss(corn_log_pmf(out), t.mask)
    if method == "soft_corn_emp":
        return soft_corn_loss(out, t.empirical)
    if method == "ce_emp":
        return ce_soft_loss(out, t.empirical)
    if method == "soft_coral_emp":
        return soft_coral_loss(out, t.empirical)
    if method == "soft_corn_mix50":
        return soft_corn_loss(out, 0.5 * t.p + 0.5 * t.empirical)
    if method == "ce_mix50":
        return ce_soft_loss(out, 0.5 * t.p + 0.5 * t.empirical)
    if method == "soft_coral_mix50":
        return soft_coral_loss(out, 0.5 * t.p + 0.5 * t.empirical)
    if method in ("exact_corn", "exact_ce", "exact_coral"):
        if t.m <= 0:
            raise ValueError("the exact set likelihood needs the number of raters per set (Target.m)")
        return exact_set_loss(log_pmf(HEAD_OF[method], out, t.p.shape[-1]), t.mask, t.m)
    if method == "sord":
        return sord_loss(out, t.p, 1)
    if method == "sord_sq":
        return sord_loss(out, t.p, 2)
    if method == "cdwce":
        return cdwce_loss(out, t.p)
    if method in ("binom", "poisson") or method.startswith("clm_"):
        return pmf_ce_loss(log_pmf(HEAD_OF[method], out, t.p.shape[-1]), t.p)
    raise ValueError(method)


def output_dim(method, K):
    head = HEAD_OF[method]
    if head == "softmax":
        return K
    if head in ("binom", "poisson"):
        return 1
    if head == "clm_ao":
        return K                        # K - 1 cumulative logits plus the link parameter
    return K - 1
