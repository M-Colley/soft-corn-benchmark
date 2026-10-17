"""Evaluation metrics, vectorised over ensemble members.

Set-valued targets (built from the m annotators used for training):
    set_mae_*   distance from the point prediction to the nearest accepted level (thesis set-MAE)
    set_acc_*   point prediction inside the accepted set (thesis set-accuracy)
    nll         cross-entropy against the target distribution p (uniform-over-set on real data,
                the exact population target E[u_S | x] on synthetic data)
    pll         -log P(y in S), the partial-label likelihood of the observed set
    rps         ranked probability score against p (proper, ordinal)
    emd         earth mover's distance / CAD between predicted and target CDFs
    spread_r    Pearson correlation, across items, of predicted and target standard deviations
Held-out annotators (ratings never used to build any label or to select a model):
    ho_nll, ho_rps, ho_emd, ho_mae  scored against the empirical distribution of each item's
                held-out ratings, averaged over items that have any
xnll (only when the number m of raters per set is known): -log P(S | pi_hat), the exact likelihood of the
                observed sets under m independent ratings from the predicted PMF; a target-agnostic
                validation criterion for models trained on different soft targets
Point predictions use the median of the predicted PMF ("med", the CORN/CORAL decision rule)
or its mode ("mode").

Set-valued predictions and calibration (what an application that shows an acceptance set needs):
    int{50,80,90}_*   the narrowest contiguous interval of levels whose predicted mass is at least
                the nominal level (ties: larger mass, then lower start), scored by
                  width    number of levels in the interval
                  jdist    Jaccard distance to the observed set
                  hocov    share of held-out ratings inside the interval (calibrated: >= nominal)
                  iscore   width + (2 / (1 - level)) * expected distance of a held-out rating to the
                           interval (Winkler-type interval score; lower is better, and only
                           approximately proper because the interval is a highest-density one)
    ho_ece            ordinal expected calibration error: mean over k of the binned gap between the
                predicted P(y > k) and the held-out raters' share above k, items weighted equally

`prepare` precomputes everything that does not depend on the predictions, so `evaluate` runs
without host-device synchronisation.
"""
import torch

from .losses import exact_set_logp

SET_TYPES = ("single", "contig", "gapped")
LEVELS = (0.5, 0.8, 0.9)        # nominal mass of the predicted intervals
ECE_BINS = 10


def set_types(mask):
    """0 = singleton, 1 = contiguous multi-level, 2 = gapped."""
    mask = mask.bool()
    size = mask.sum(-1)
    K = mask.shape[-1]
    idx = torch.arange(K, device=mask.device)
    lo = torch.where(mask, idx, K).min(-1).values
    hi = torch.where(mask, idx, -1).max(-1).values
    span = hi - lo + 1
    t = torch.full_like(size, 1)
    t[size == 1] = 0
    t[(size > 1) & (span > size)] = 2
    return t


def _median(pmf):
    return (torch.cumsum(pmf, -1) < 0.5 - 1e-9).sum(-1).clamp_max(pmf.shape[-1] - 1)


def _set_dist(pred, mask):
    K = mask.shape[-1]
    k = torch.arange(K, device=mask.device)
    d = (pred[..., None] - k).abs().float()
    return d.masked_fill(~mask, float("inf")).min(-1).values


def _sd(pmf):
    k = torch.arange(pmf.shape[-1], device=pmf.device, dtype=pmf.dtype)
    return ((pmf * k ** 2).sum(-1) - (pmf * k).sum(-1) ** 2).clamp_min(0).sqrt()


def best_interval(pmf, level):
    """Narrowest contiguous interval [lo, hi] with mass >= level (ties: larger mass, then lower lo).
    pmf (..., K) -> lo, hi (...). The full range always qualifies, so a choice always exists."""
    K = pmf.shape[-1]
    F0 = torch.cat([torch.zeros_like(pmf[..., :1]), torch.cumsum(pmf, -1)], -1)
    mass = F0[..., None, 1:] - F0[..., :-1, None]                    # [..., a, b] = P(a <= y <= b)
    a = torch.arange(K, device=pmf.device)[:, None]
    b = torch.arange(K, device=pmf.device)[None, :]
    ok = (b >= a) & (mass >= level - 1e-6)
    score = torch.where(ok, (b - a + 1).to(pmf.dtype) - 0.5 * mass, float("inf"))
    idx = score.flatten(-2).argmin(-1)
    return idx // K, idx % K


def _interval_metrics(pmf, ev, out):
    mask, K = ev["mask"], pmf.shape[-1]
    k = torch.arange(K, device=pmf.device)
    for level in LEVELS:
        tag = f"int{round(100 * level)}"
        lo, hi = best_interval(pmf, level)
        inside = (k >= lo[..., None]) & (k <= hi[..., None])
        width = (hi - lo + 1).to(pmf.dtype)
        inter = (inside & mask).sum(-1).to(pmf.dtype)
        out[f"{tag}_width"] = width.mean(-1)
        out[f"{tag}_jdist"] = (1 - inter / (mask.sum(-1) + width - inter)).mean(-1)
        if "ho_w" in ev:
            w, item = ev["ho_w"], ev["ho_item"]
            out[f"{tag}_hocov"] = ((w * inside).sum(-1) * item).sum(-1)
            miss = torch.maximum(lo[..., None] - k, k - hi[..., None]).clamp_min(0).to(pmf.dtype)
            out[f"{tag}_iscore"] = ((width + 2 / (1 - level) * (w * miss).sum(-1)) * item).sum(-1)


def _ho_ece(F_hat, ev, bins=ECE_BINS):
    """sum_b (weight of bin b) * |mean predicted P(y > k) - mean held-out share above k| over the
    equal-width bins of the predicted probability, averaged over k = 0..K-2."""
    g = (1 - F_hat[..., :-1]).clamp(0, 1)                              # (M, N, K-1)
    r = 1 - ev["ho_F"][..., :-1]                                       # (N, K-1)
    cell = (g * bins).long().clamp_max(bins - 1)[..., None] == torch.arange(bins, device=g.device)
    w = cell * ev["ho_item"][None, :, None, None]                      # (M, N, K-1, bins)
    gap = (w * (g - r)[..., None]).sum(1).abs().sum(-1)                # (M, K-1)
    return gap.mean(-1)


def prepare(p, mask, ho_counts=None, min_group=5, m=0):
    mask = mask.bool()
    ev = dict(p=p, mask=mask, F=torch.cumsum(p, -1), types=set_types(mask), m=m)
    sd = _sd(p)
    ev["sd_c"] = sd - sd.mean()
    ev["groups"] = {}
    for g, name in enumerate(SET_TYPES):
        sel = (ev["types"] == g).float()
        if sel.sum() >= min_group:
            ev["groups"][name] = sel / sel.sum()
    if ho_counts is not None:
        n = ho_counts.sum(-1)
        if (n > 0).any():
            ev["ho_w"] = ho_counts / n.clamp_min(1)[:, None]
            ev["ho_item"] = (n > 0).float() / (n > 0).sum()
            ev["ho_F"] = torch.cumsum(ev["ho_w"], -1)
    return ev


@torch.no_grad()
def evaluate(logp, ev):
    """logp (M, N, K) log predicted PMF; ev from `prepare`. Returns {name: (M,) tensor}."""
    p, mask = ev["p"], ev["mask"]
    pmf = logp.exp()
    K = pmf.shape[-1]
    out = {}
    med = _median(pmf)
    d_med = _set_dist(med, mask)
    d_mode = _set_dist(pmf.argmax(-1), mask)
    out["set_mae_med"] = d_med.mean(-1)
    out["set_acc_med"] = (d_med == 0).float().mean(-1)
    out["set_mae_mode"] = d_mode.mean(-1)
    out["set_acc_mode"] = (d_mode == 0).float().mean(-1)

    nll_i = -(p * logp).sum(-1)
    pll_i = -torch.logsumexp(logp.masked_fill(~mask, float("-inf")), -1)
    F_hat = torch.cumsum(pmf, -1)
    rps_i = ((F_hat - ev["F"])[..., :-1] ** 2).sum(-1) / (K - 1)
    emd_i = (F_hat - ev["F"]).abs().sum(-1)
    out.update(nll=nll_i.mean(-1), pll=pll_i.mean(-1), rps=rps_i.mean(-1), emd=emd_i.mean(-1))
    if ev.get("m", 0) > 0:      # exact likelihood of the observed sets when they come from m independent raters
        out["xnll"] = -exact_set_logp(logp, mask, ev["m"]).mean(-1)

    sd_hat = _sd(pmf)
    a = sd_hat - sd_hat.mean(-1, keepdim=True)
    out["spread_r"] = (a * ev["sd_c"]).sum(-1) / (a.norm(dim=-1) * ev["sd_c"].norm() + 1e-12)

    if "ho_w" in ev:
        w, item = ev["ho_w"], ev["ho_item"]
        out["ho_nll"] = (-(w * logp).sum(-1) * item).sum(-1)
        out["ho_rps"] = ((((F_hat - ev["ho_F"])[..., :-1] ** 2).sum(-1) / (K - 1)) * item).sum(-1)
        out["ho_emd"] = ((F_hat - ev["ho_F"]).abs().sum(-1) * item).sum(-1)
        dist = (med[..., None] - torch.arange(K, device=pmf.device)).abs().float()
        out["ho_mae"] = ((w * dist).sum(-1) * item).sum(-1)
        out["ho_ece"] = _ho_ece(F_hat, ev)
    _interval_metrics(pmf, ev, out)

    for name, wg in ev["groups"].items():
        out[f"set_mae_med_{name}"] = (d_med * wg).sum(-1)
        out[f"set_acc_med_{name}"] = ((d_med == 0).float() * wg).sum(-1)
        out[f"nll_{name}"] = (nll_i * wg).sum(-1)
        out[f"pll_{name}"] = (pll_i * wg).sum(-1)
    return out
