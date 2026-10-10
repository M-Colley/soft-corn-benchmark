"""Correctness checks for the losses: every identity claimed for soft-CORN in the thesis,
agreement with coral_pytorch, and agreement of the OLDL ports with the reference code."""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from coral_pytorch.losses import corn_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ordbench import losses as L  # noqa: E402
from ordbench.metrics import best_interval, evaluate, prepare, set_types  # noqa: E402

K = 5


@pytest.fixture(autouse=True)
def float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def random_sets(N, K, gen):
    mask = torch.rand(N, K, generator=gen) < 0.35
    empty = ~mask.any(-1)
    mask[empty, torch.randint(0, K, (int(empty.sum()),), generator=gen)] = True
    p = mask.double() / mask.sum(-1, keepdim=True)
    return mask, p


@pytest.fixture
def batch():
    g = torch.Generator().manual_seed(0)
    N = 64
    z = torch.randn(N, K - 1, generator=g) * 2
    mask, p = random_sets(N, K, g)
    y = torch.randint(0, K, (N,), generator=g)
    return z, mask, p, y


def test_corn_pmf_normalised(batch):
    z, *_ = batch
    assert torch.allclose(L.corn_log_pmf(z).exp().sum(-1), torch.ones(len(z)))


def test_soft_corn_equals_corn_on_one_hot(batch):
    z, _, _, y = batch
    onehot = F.one_hot(y, K).double()
    ref = corn_loss(z, y, num_classes=K)
    assert torch.allclose(L.soft_corn_loss(z, onehot, normalise="corn"), ref, atol=1e-12)
    # per-example normalisation differs only by the constant N / sum_k |S_k|
    n_cond = sum(int((y > k - 1).sum()) for k in range(K - 1))
    assert torch.allclose(L.soft_corn_loss(z, onehot) * len(y) / n_cond, ref, atol=1e-12)


def test_soft_corn_is_cross_entropy_of_decoded_pmf(batch):
    z, _, p, _ = batch
    ce = -(p * L.corn_log_pmf(z)).sum(-1).mean()
    assert torch.allclose(L.soft_corn_loss(z, p), ce, atol=1e-12)


def test_soft_corn_gradient_and_hessian(batch):
    """Appendix D: dL/dz_k = q_{k-1} sigma(z_k) - q_k and d2L/dz_k^2 = q_{k-1} sigma(1 - sigma)."""
    z, _, p, _ = batch
    z = z[:1].clone().requires_grad_(True)
    pi = p[:1]
    loss = L.soft_corn_loss(z, pi) * 1  # one example: mean == sum
    (g,) = torch.autograd.grad(loss, z, create_graph=True)
    q = 1 - torch.cumsum(pi, -1)[:, :-1]
    q_prev = torch.cat([torch.ones(1, 1), q[:, :-1]], -1)
    s = torch.sigmoid(z)
    assert torch.allclose(g, q_prev * s - q, atol=1e-12)
    H = torch.stack([torch.autograd.grad(g[0, k], z, retain_graph=True)[0][0] for k in range(K - 1)])
    assert torch.allclose(H, torch.diag((q_prev * s * (1 - s))[0]), atol=1e-12)


def test_soft_coral_reduces_to_coral(batch):
    z, _, _, y = batch
    onehot = F.one_hot(y, K).double()
    assert torch.allclose(L.soft_coral_loss(z, onehot), L.coral_hard_loss(z, y), atol=1e-12)


def test_levels_match_library(batch):
    from coral_pytorch.dataset import levels_from_labelbatch
    _, _, _, y = batch
    assert torch.equal(L.levels(y, K), levels_from_labelbatch(y, num_classes=K).float())
    assert torch.equal(L.one_hot(y, K), F.one_hot(y, K).float())


def test_vectorised_corn_hard_matches_library(batch):
    z, _, _, y = batch
    zz = torch.stack([z, 2 * z, -z])
    got = L.corn_hard_loss(zz, y)
    want = torch.stack([corn_loss(v, y, num_classes=K) for v in zz])
    assert torch.allclose(got, want)


def test_pll_singletons_is_cross_entropy(batch):
    _, _, _, y = batch
    g = torch.Generator().manual_seed(1)
    logits = torch.randn(len(y), K, generator=g)
    mask = F.one_hot(y, K).bool()
    assert torch.allclose(L.pll_loss(F.log_softmax(logits, -1), mask), F.cross_entropy(logits, y))


# ---- reference OLDL implementations (github.com/downdric/OLDL), CPU, verbatim maths ----------

def ref_cad(pd, gt):
    s = 0.0
    for i in range(0, 10):
        m = i + 1
        s = s + torch.mean((pd[:, 0:m].sum(1) - gt[:, 0:m].sum(1)).abs())
    return s


def ref_qfd(pd, gt):
    A = torch.tensor([[1, 4 / 5, 3 / 5, 2 / 5, 1 / 5], [4 / 5, 1, 4 / 5, 3 / 5, 2 / 5],
                      [3 / 5, 4 / 5, 1, 4 / 5, 3 / 5], [2 / 5, 3 / 5, 4 / 5, 1, 4 / 5],
                      [1 / 5, 2 / 5, 3 / 5, 4 / 5, 1]])
    t = (pd - gt).abs()
    return 0.1 * torch.diag(t @ A @ t.T).sum()


def ref_cjs(pd, gt):
    s = 0.0
    for i in range(0, 10):
        PD = pd[:, :i + 1].sum(1)
        GT = gt[:, :i + 1].sum(1)
        M = (PD + GT) / 2
        t = 0.5 * PD * (torch.log(PD) - torch.log(M)) + 0.5 * GT * (torch.log(GT + 1e-8) - torch.log(M))
        s = s + torch.mean(t)
    return s


def test_oldl_ports_match_reference(batch):
    _, _, p, y = batch
    g = torch.Generator().manual_seed(2)
    logits = torch.randn(len(y), K, generator=g)
    pd = F.softmax(logits, -1)
    assert torch.allclose(L.oldl_cad_loss(logits, p), ref_cad(pd, p), atol=1e-10)
    assert torch.allclose(L.oldl_qfd_loss(logits, p), ref_qfd(pd, p) / (0.1 * len(y)), atol=1e-10)
    assert torch.allclose(L.oldl_cjs_loss(logits, p), ref_cjs(pd, p), atol=1e-6)


def test_set_types():
    m = torch.tensor([[0, 1, 0, 0, 0], [0, 1, 1, 1, 0], [1, 0, 1, 0, 0], [1, 1, 0, 0, 1]]).bool()
    assert set_types(m).tolist() == [0, 1, 2, 2]


def test_metrics_perfect_prediction():
    mask = torch.tensor([[0, 1, 0, 0, 0], [0, 0, 1, 1, 0]]).bool()
    p = mask.double() / mask.sum(-1, keepdim=True)
    logp = p.clamp_min(1e-12).log()[None]
    out = evaluate(logp, prepare(p, mask))
    assert out["set_mae_med"].item() == 0 and out["set_acc_mode"].item() == 1
    assert out["rps"].item() < 1e-9 and out["emd"].item() < 1e-9


def test_every_method_runs_and_backprops(batch):
    _, mask, p, y = batch
    t = L.Target(p, mask, y)
    for m in L.METHODS:
        out = torch.randn(3, len(y), L.output_dim(m, K), requires_grad=True)
        loss = L.compute_loss(m, out, t)
        assert loss.shape == (3,), m
        loss.sum().backward()
        assert torch.isfinite(out.grad).all(), m
        lp = L.log_pmf(L.HEAD_OF[m], out.detach())
        assert torch.allclose(lp.exp().sum(-1), torch.ones(3, len(y))), m


def test_metrics_match_direct_computation():
    """The sync-free weighted forms equal plain boolean-indexed means."""
    g = torch.Generator().manual_seed(3)
    N = 200
    mask, p = random_sets(N, K, g)
    ho = torch.randint(0, 3, (N, K), generator=g).double()
    ho[:50] = 0
    logp = torch.log_softmax(torch.randn(2, N, K, generator=g), -1)
    out = evaluate(logp, prepare(p, mask, ho))
    t = set_types(mask)
    has = ho.sum(-1) > 0
    w = ho[has] / ho[has].sum(-1, keepdim=True)
    assert torch.allclose(out["ho_nll"], -(w * logp[:, has]).sum(-1).mean(-1))
    nll_i = -(p * logp).sum(-1)
    for gi, name in enumerate(("single", "contig", "gapped")):
        assert torch.allclose(out[f"nll_{name}"], nll_i[:, t == gi].mean(-1))


def test_best_interval_matches_brute_force():
    g = torch.Generator().manual_seed(4)
    pmf = torch.softmax(torch.randn(300, K, generator=g) * 1.5, -1)
    for level in (0.5, 0.8, 0.9, 1.0):
        lo, hi = best_interval(pmf, level)
        for i in range(len(pmf)):
            best = min(((b - a, -float(pmf[i, a:b + 1].sum()), a) for a in range(K) for b in range(a, K)
                        if float(pmf[i, a:b + 1].sum()) >= level - 1e-6))
            assert (int(lo[i]), int(hi[i])) == (best[2], best[2] + best[0]), (level, i)


def test_interval_and_calibration_metrics_match_direct_computation():
    g = torch.Generator().manual_seed(5)
    N = 400
    mask, p = random_sets(N, K, g)
    ho = torch.randint(0, 4, (N, K), generator=g).double()
    ho[:60] = 0
    pmf = torch.softmax(torch.randn(2, N, K, generator=g), -1)
    out = evaluate(pmf.log(), prepare(p, mask, ho))
    has = ho.sum(-1) > 0
    w = ho / ho.sum(-1, keepdim=True).clamp_min(1)
    for level in (0.5, 0.8, 0.9):
        tag = f"int{round(100 * level)}"
        lo, hi = best_interval(pmf, level)
        for m in range(2):
            cov, score, jd, wd = [], [], [], []
            for i in range(N):
                ks = torch.arange(K)
                inside = (ks >= lo[m, i]) & (ks <= hi[m, i])
                width = float(hi[m, i] - lo[m, i] + 1)
                jd.append(1 - float((inside & mask[i]).sum()) / float((inside | mask[i]).sum()))
                wd.append(width)
                if has[i]:
                    miss = torch.clamp(torch.maximum(lo[m, i] - ks, ks - hi[m, i]), min=0).double()
                    cov.append(float((w[i] * inside).sum()))
                    score.append(width + 2 / (1 - level) * float((w[i] * miss).sum()))
            assert abs(float(out[f"{tag}_width"][m]) - sum(wd) / N) < 1e-9
            assert abs(float(out[f"{tag}_jdist"][m]) - sum(jd) / N) < 1e-9
            assert abs(float(out[f"{tag}_hocov"][m]) - sum(cov) / len(cov)) < 1e-6   # item weights are float32
            assert abs(float(out[f"{tag}_iscore"][m]) - sum(score) / len(score)) < 1e-6
    # ECE against a plain loop over bins
    Fh = torch.cumsum(pmf, -1)
    for m in range(2):
        per_k = []
        for k in range(K - 1):
            gk = (1 - Fh[m, has, k])
            rk = 1 - torch.cumsum(w[has], -1)[:, k]
            b = torch.clamp((gk * 10).long(), max=9)
            per_k.append(sum(float((b == j).sum()) / int(has.sum()) * abs(float(gk[b == j].mean() - rk[b == j].mean()))
                             for j in range(10) if (b == j).any()))
        assert abs(float(out["ho_ece"][m]) - sum(per_k) / (K - 1)) < 1e-6


def test_calibration_metrics_reference_cases():
    N = 50
    mask = torch.zeros(N, K).bool()
    mask[:, 2] = True
    p = mask.double()
    # held-out raters distributed exactly as the prediction: no calibration gap, intervals cover >= level
    pmf = torch.softmax(torch.randn(1, N, K, generator=torch.Generator().manual_seed(6)), -1)
    out = evaluate(pmf.log(), prepare(p, mask, pmf[0]))
    assert out["ho_ece"].item() < 1e-6
    for tag, level in (("int50", 0.5), ("int80", 0.8), ("int90", 0.9)):
        assert out[f"{tag}_hocov"].item() >= level - 1e-9
    # a constant, overconfident prediction against raters that split evenly: P(y > 2) is 1, share is 0.5
    pmf = torch.zeros(1, N, K, dtype=torch.float64)
    pmf[..., 4] = 1.0
    ho = torch.zeros(N, K, dtype=torch.float64)
    ho[:, [0, 4]] = 1.0
    out = evaluate(pmf.clamp_min(1e-12).log(), prepare(p, mask, ho))
    # P(y > k) = 1 for k < 4 and the raters' share above k is 0.5 for k < 4: gap 0.5 at 4 of 4 thresholds
    assert abs(out["ho_ece"].item() - 0.5) < 1e-6
    assert out["int80_width"].item() == 1 and abs(out["int80_hocov"].item() - 0.5) < 1e-6


def _argmin_pmf(fn, head, labels, w, gen):
    z = (torch.randn(1, K if head == "softmax" else K - 1, generator=gen) * 0.5).requires_grad_(True)
    opt = torch.optim.LBFGS([z], max_iter=400, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-11, tolerance_change=1e-13)

    def closure():
        opt.zero_grad()
        loss = (w * torch.stack([fn(z, labels[i:i + 1]) for i in range(len(w))])).sum()
        loss.backward()
        return loss
    for _ in range(3):
        opt.step(closure)
    return L.log_pmf(head, z.detach()).exp()[0]


def test_properness_for_the_mean_soft_label():
    """soft-CORN, soft-OR/CORAL and soft CE are minimised in expectation by the mean soft label
    t = E[u_S]; the OLDL CJS loss is not (used in REPORT.md, section 2)."""
    torch.set_num_threads(1)
    g = torch.Generator().manual_seed(0)
    masks = torch.tensor([[1, 1, 0, 0, 0], [0, 0, 1, 0, 1], [0, 1, 0, 1, 0]]).bool()
    labels = masks.double() / masks.sum(-1, keepdim=True)
    w = torch.tensor([0.5, 0.3, 0.2])
    t = (w[:, None] * labels).sum(0)
    for fn, head in [(L.soft_corn_loss, "corn"), (L.soft_coral_loss, "or"), (L.ce_soft_loss, "softmax")]:
        assert (_argmin_pmf(fn, head, labels, w, g) - t).abs().max() < 1e-4
    assert (_argmin_pmf(L.oldl_cjs_loss, "softmax", labels, w, g) - t).abs().max() > 1e-2
