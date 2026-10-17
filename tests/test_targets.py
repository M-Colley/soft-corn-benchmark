"""Training targets built from the m ratings (empirical, exact set likelihood, mixtures) and the extra
baselines (SORD, CDW-CE, unimodal heads, cumulative link models)."""
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ordbench import losses as L  # noqa: E402
from ordbench import synthetic as S  # noqa: E402
from ordbench.heads import EnsembleHead  # noqa: E402
from ordbench.metrics import evaluate, prepare  # noqa: E402

K = 5


@pytest.fixture(autouse=True)
def float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def rating_batch(N=48, m=3, seed=0, K=K):
    g = np.random.default_rng(seed)
    pi = g.dirichlet(np.ones(K), size=N)
    ratings = S.sample_ratings(pi, m, g)
    counts = np.eye(K)[ratings].sum(1)
    mask = counts > 0
    return pi, ratings, torch.as_tensor(counts), torch.as_tensor(mask)


@pytest.mark.parametrize("K_,m", [(4, 2), (5, 3), (5, 5), (3, 4)])
def test_exact_set_likelihood_matches_enumeration(K_, m):
    g = np.random.default_rng(1)
    pi = g.dirichlet(np.ones(K_), size=6)
    sets = [frozenset(c) for r in range(1, K_ + 1) for c in itertools.combinations(range(K_), r)]
    for row in pi:
        ref = L.brute_force_set_prob(row, m)
        mask = torch.zeros(len(sets), K_, dtype=torch.bool)
        for i, s in enumerate(sets):
            mask[i, list(s)] = True
        logp = torch.as_tensor(np.log(row))[None].expand(len(sets), K_)
        got = L.exact_set_logp(logp, mask, m).exp()
        for i, s in enumerate(sets):
            assert abs(float(got[i]) - ref.get(s, 0.0)) < 1e-10, (s, m)
        # the probabilities of the sets that can occur with m ratings sum to one
        feasible = [i for i, s in enumerate(sets) if len(s) <= m]
        assert abs(sum(float(got[i]) for i in feasible) - 1) < 1e-10


def test_exact_loss_gradient_and_singletons():
    pi, _, _, mask = rating_batch()
    z = torch.randn(2, len(mask), K - 1, requires_grad=True)
    lp = L.corn_log_pmf(z)
    loss = L.exact_set_loss(lp, mask, 3)
    loss.sum().backward()
    assert torch.isfinite(z.grad).all()
    # a singleton set {k} has probability pi_k ** m
    single = torch.zeros(K, K, dtype=torch.bool)
    single[torch.arange(K), torch.arange(K)] = True
    logp = torch.log_softmax(torch.randn(K, K), -1)
    assert torch.allclose(L.exact_set_logp(logp, single, 3), 3 * logp.diagonal())


def test_exact_likelihood_is_maximised_by_the_rater_distribution():
    """In expectation over sets drawn from pi, -log P(S | pi_hat) is minimised at pi_hat = pi (it is a
    likelihood); the uniform-over-set cross-entropy is not minimised there."""
    g = np.random.default_rng(2)
    pi = g.dirichlet(np.ones(K) * 0.7)
    m = 3
    sets = L.brute_force_set_prob(pi, m)
    keys = list(sets)
    mask = torch.zeros(len(keys), K, dtype=torch.bool)
    for i, s in enumerate(keys):
        mask[i, list(s)] = True
    w = torch.tensor([sets[k] for k in keys])

    def expected(pmf, exact):
        lp = torch.as_tensor(np.log(pmf))[None].expand(len(keys), K)
        if exact:
            return float((w * -L.exact_set_logp(lp, mask, m)).sum())
        u = mask.double() / mask.sum(-1, keepdim=True)
        return float((w * -(u * lp).sum(-1)).sum())

    base = expected(pi, True)
    for _ in range(20):
        other = g.dirichlet(np.ones(K) * 3)
        assert expected(other, True) > base
    t = sum(sets[k] * (mask[i].double() / mask[i].sum()) for i, k in enumerate(keys))
    assert np.abs(t.numpy() - pi).max() > 0.01          # E[u_S] differs from pi, so uniform targets are biased


def test_new_methods_run_and_backprop():
    pi, _, counts, mask = rating_batch(N=32)
    p = mask.double() / mask.sum(-1, keepdim=True)
    y = torch.zeros(len(mask), dtype=torch.long)
    t = L.Target(p, mask, y, counts, 3)
    for method in L.ALL_METHODS:
        dim = L.output_dim(method, K)
        head = EnsembleHead(3, 6, dim, output=L.OUTPUT_OF[L.HEAD_OF[method]], hidden=4, dropout=0.0).double()
        x = torch.randn(len(mask), 6)
        out = head(x)
        assert out.shape == (3, len(mask), dim), method
        loss = L.compute_loss(method, out, t)
        assert loss.shape == (3,), method
        loss.sum().backward()
        assert all(torch.isfinite(q.grad).all() for q in head.parameters() if q.grad is not None), method
        lp = L.log_pmf(L.HEAD_OF[method], out.detach(), K)
        assert lp.shape == (3, len(mask), K), method
        assert torch.allclose(lp.exp().sum(-1), torch.ones(3, len(mask))), method


def test_empirical_target_reduces_to_uniform_when_ratings_are_distinct():
    mask = torch.tensor([[1, 1, 1, 0, 0], [0, 1, 1, 1, 0]]).bool()
    counts = mask.double()                                # three distinct ratings per item
    p = mask.double() / 3
    z = torch.randn(2, 4)
    t = L.Target(p, mask, torch.zeros(2, dtype=torch.long), counts, 3)
    assert torch.allclose(L.compute_loss("soft_corn_emp", z, t), L.compute_loss("soft_corn", z, t))
    # a mix of identical targets is that target, for any logits
    z5 = torch.randn(2, 5)
    assert torch.allclose(L.compute_loss("ce_mix50", z5, t), L.compute_loss("ce_soft", z5, t))


def test_unimodal_pmfs_match_scipy():
    s = torch.tensor([[-1.3], [0.0], [0.8]])
    pb = L.log_pmf("binom", s, K).exp().numpy()
    q = 1 / (1 + np.exp(-s.numpy()[:, 0]))
    for i, qi in enumerate(q):
        assert np.allclose(pb[i], stats.binom.pmf(np.arange(K), K - 1, qi))
    pp = L.log_pmf("poisson", s, K).exp().numpy()
    for i, si in enumerate(s.numpy()[:, 0]):
        ref = stats.poisson.pmf(np.arange(K), np.exp(si))
        assert np.allclose(pp[i], ref / ref.sum())


def test_links_special_cases():
    z = torch.linspace(-4, 4, 9)
    assert torch.allclose(L.link_cdf("logit", z), torch.sigmoid(z))
    assert torch.allclose(L.link_cdf("probit", z), torch.as_tensor(stats.norm.cdf(z.numpy())))
    assert torch.allclose(L.link_cdf("cloglog", z), 1 - torch.exp(-torch.exp(z)))
    assert torch.allclose(L.link_cdf("ao", z, torch.tensor(1.0)), torch.sigmoid(z))
    assert torch.allclose(L.link_cdf("ao", z, torch.tensor(1e-4)), L.link_cdf("cloglog", z), atol=1e-3)


@pytest.mark.parametrize("output", ["clm", "clm_ao"])
def test_clm_thresholds_are_ordered_whatever_the_parameters(output):
    g = torch.Generator().manual_seed(0)
    out_dim = K - 1 if output == "clm" else K
    head = EnsembleHead(4, 7, out_dim, output=output)
    with torch.no_grad():
        for p in head.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 3)
    z = head(torch.randn(11, 7))
    z = z[..., :K - 1]
    assert (z[..., 1:] < z[..., :-1]).all()
    pmf = L.log_pmf("clm_logit" if output == "clm" else "clm_ao", head(torch.randn(11, 7)), K).exp()
    assert (pmf > 0).all() and torch.allclose(pmf.sum(-1), torch.ones(4, 11))


def test_sord_and_cdwce():
    Mx = L.sord_matrix(K, 1, dtype=torch.float64)
    assert torch.allclose(Mx.sum(-1), torch.ones(K))
    assert (Mx.argmax(-1) == torch.arange(K)).all()
    mask = torch.tensor([[0, 1, 0, 0, 0], [1, 0, 0, 0, 1]]).bool()
    p = mask.double() / mask.sum(-1, keepdim=True)
    logits = torch.randn(2, K)
    assert torch.allclose(L.sord_loss(logits, p), L.ce_soft_loss(logits, p @ Mx))
    # CDW-CE vanishes when all mass sits on the (single) level of the set, and is positive otherwise
    sharp = torch.full((1, K), -30.0)
    sharp[0, 1] = 30.0
    assert L.cdwce_loss(sharp, p[:1]) < 1e-6
    assert L.cdwce_loss(torch.zeros(1, K), p[:1]) > 0.1


def test_ratings_behind_the_sets_are_consistent():
    X, mask, pi, ratings = S.make("po", 200, 3, seed=1, return_ratings=True)
    counts = np.eye(5)[ratings].sum(1)
    assert (counts.sum(1) == 3).all() and ((counts > 0) == mask).all()
    X2, mask2, pi2 = S.make("po", 200, 3, seed=1)
    assert np.array_equal(mask, mask2) and np.allclose(pi, pi2)     # the extra output changes no draw


def test_xnll_metric_and_selection_criterion():
    pi, _, counts, mask = rating_batch(N=100, m=3, seed=3)
    p = mask.double() / mask.sum(-1, keepdim=True)
    logp = torch.log_softmax(torch.randn(2, 100, K), -1)
    out = evaluate(logp, prepare(p, mask, m=3))
    assert torch.allclose(out["xnll"], -L.exact_set_logp(logp, mask, 3).mean(-1))
    assert "xnll" not in evaluate(logp, prepare(p, mask))


def test_random_effects_recovers_a_common_effect_and_heterogeneity():
    from ordbench.analysis import random_effects
    rng = np.random.default_rng(0)
    v = np.full(40, 0.01 ** 2)
    out = random_effects(-0.02 + rng.normal(0, 0.003, 40), v)
    assert abs(out["mu"] + 0.02) < 0.003 and out["tau"] < 0.01 and out["p_below"] > 0.99
    het = random_effects(rng.normal(0, 0.05, 40), v)                       # effects differ across problems
    assert abs(het["tau"] - 0.05) < 0.02 and het["se"] > 0.005
    flat = random_effects(rng.normal(0, 0.001, 40), v, rope=0.01)
    assert flat["p_rope"] > 0.9


def test_pairwise_table_agrees_with_a_direct_computation():
    import pandas as pd
    from ordbench.analysis import corrected_ttest, pairwise_table
    rng = np.random.default_rng(1)
    rows = []
    for ds in ("a", "b", "c"):
        for seed in range(5):
            base = rng.normal(1.0, 0.05)
            for method, shift in (("soft_corn", -0.04), ("other", 0.0)):
                rows.append(dict(dataset=ds, n_train=100, n_test=50, seed=seed, method=method, select_by="nll",
                                 nll=base + shift + rng.normal(0, 0.005)))
    df = pd.DataFrame(rows)
    out = pairwise_table(df, ["dataset", "n_train"], "nll", "nll", ["other"], rope=0.005).iloc[0]
    diffs = []
    for ds in "abc":
        g = df[df.dataset == ds].pivot(index="seed", columns="method", values="nll")
        diffs.append(corrected_ttest((g.soft_corn - g.other).values, 100, 50)[0])
    assert abs(out.mean_diff - np.mean(diffs)) < 1e-12 and out.wins == 3 and out.p_better > 0.9
