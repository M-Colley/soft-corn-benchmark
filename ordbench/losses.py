"""Losses compared in the benchmark.

Every loss acts on the raw outputs of one of four output parameterisations and on a batch
`Target` that carries the set-valued label in three equivalent views:

    p     (N, K)  uniform distribution over the accepted set (the soft label)
    mask  (N, K)  boolean indicator of the accepted set
    y     (N,)    hard label obtained by collapsing the set to its lower median

Parameterisations (`HEAD_OF[name]`):
    softmax  K logits, PMF = softmax
    corn     K-1 conditional logits, P(y>k | y>k-1) = sigmoid(z_k)            (Shi et al., 2023)
    coral    K-1 logits s(x) + b_k with one shared weight vector              (Cao et al., 2020)
    or       K-1 independent logits for P(y>k), no rank constraint            (Niu et al., 2016)

Hard-label CORAL and soft-CORAL / soft-OR call `coral_pytorch.coral_loss`; hard-label CORN uses
an identity with soft-CORN that tests/test_losses.py checks against `coral_pytorch.corn_loss`. The OLDL
objectives port the reference implementation of Wen et al. (ICCV 2023, github.com/downdric/OLDL,
MIT) from its hard-coded K in {5, 10, 85} to arbitrary K.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from coral_pytorch.losses import coral_loss

EPS = 1e-7


@dataclass
class Target:
    p: torch.Tensor      # (N, K) float
    mask: torch.Tensor   # (N, K) bool
    y: torch.Tensor      # (N,) long

    def index(self, idx):
        return Target(self.p[idx], self.mask[idx], self.y[idx])


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


def log_pmf(head, out):
    if head == "softmax":
        return F.log_softmax(out, dim=-1)
    if head == "corn":
        return corn_log_pmf(out)
    if head in ("coral", "or"):
        return cumulative_to_pmf(torch.sigmoid(out)).log()
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
# Registry
# --------------------------------------------------------------------------------------------

HEAD_OF = {
    "ce_hard": "softmax", "corn_hard": "corn", "coral_hard": "coral",
    "ce_soft": "softmax", "soft_coral": "coral", "soft_or": "or",
    "oldl_cad": "softmax", "oldl_qfd": "softmax", "oldl_cjs": "softmax",
    "pll_softmax": "softmax", "pll_corn": "corn",
    "soft_corn": "corn",
}

METHODS = list(HEAD_OF)


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
    raise ValueError(method)


def output_dim(method, K):
    return K if HEAD_OF[method] == "softmax" else K - 1
