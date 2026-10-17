"""The vectorised ensemble must train each member exactly as an independent model would."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ordbench.heads import EnsembleHead, MemberAdam  # noqa: E402


@pytest.mark.parametrize("decoupled", [False, True])
def test_member_adam_matches_torch_adam(decoupled):
    torch.manual_seed(0)
    lrs, wds = [1e-3, 3e-3, 1e-2], [0.0, 0.1, 1.0]
    head = EnsembleHead(3, 8, 4, body="mlp", output="full", hidden=6, dropout=0.0).double()
    refs = []
    for m in range(3):
        W1 = head.W1[m].detach().clone().requires_grad_(True)
        b1 = head.b1[m].detach().clone().requires_grad_(True)
        W = head.W[m].detach().clone().requires_grad_(True)
        b = head.b[m].detach().clone().requires_grad_(True)
        cls = torch.optim.AdamW if decoupled else torch.optim.Adam
        opt = cls([{"params": [W1, W], "weight_decay": wds[m]},
                                 {"params": [b1, b], "weight_decay": 0.0}], lr=lrs[m])
        refs.append((W1, b1, W, b, opt))
    opt = MemberAdam(list(head.named_parameters()), lrs, wds, head.decay_mask(), decoupled=decoupled)
    for _ in range(25):
        x = torch.randn(16, 8, dtype=torch.float64)
        y = torch.randn(16, 4, dtype=torch.float64)
        opt.zero_grad()
        ((head(x) - y) ** 2).sum((1, 2)).sum().backward()
        opt.step()
        for W1, b1, W, b, o in refs:
            o.zero_grad()
            out = torch.relu(x @ W1.T + b1) @ W.T + b
            ((out - y) ** 2).sum().backward()
            o.step()
    for m, (W1, b1, W, b, _) in enumerate(refs):
        for got, want in [(head.W1[m], W1), (head.b1[m], b1), (head.W[m], W), (head.b[m], b)]:
            assert torch.allclose(got, want, atol=1e-10)


def test_coral_head_shares_weight_across_thresholds():
    head = EnsembleHead(2, 5, 3, output="coral")
    out = head(torch.randn(7, 5))
    assert out.shape == (2, 7, 3)
    diffs = out - out[..., :1]
    assert torch.allclose(diffs, (head.b - head.b[:, :1])[:, None, :].expand_as(diffs), atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_graph_training_matches_eager():
    from ordbench.train import make_evalset, run_cell
    g = torch.Generator().manual_seed(0)
    X = torch.randn(300, 12, generator=g).numpy()
    mask = (torch.rand(300, 5, generator=g) < 0.4)
    mask[~mask.any(-1), 0] = True
    ev = {"val": make_evalset(X[:60], mask[:60].numpy(), device="cuda")}
    kw = dict(K=5, methods=["soft_corn", "coral_hard", "ce_soft"], body="linear", steps=120,
              eval_every=40, batch=64, configs=[(1e-3, 1e-2), (1e-2, 1.0)], device="cuda")
    a, _ = run_cell(X[60:], mask[60:].numpy(), ev, use_graph=True, **kw)
    b, _ = run_cell(X[60:], mask[60:].numpy(), ev, use_graph=False, **kw)
    for m in kw["methods"]:
        for k in ("nll", "set_mae_med", "rps"):
            assert np.allclose(a[m]["val"][k], b[m]["val"][k], atol=1e-5), (m, k)


@pytest.mark.parametrize("method", ["ce_soft", "soft_coral", "soft_corn", "soft_or"])
def test_prior_bias_init_reproduces_marginal(method):
    """Fine-tuning heads start at the training-label marginal for every parameterisation."""
    from ordbench.finetune import init_bias_to_prior
    from ordbench.losses import HEAD_OF, log_pmf, output_dim
    pbar = torch.tensor([0.05, 0.15, 0.4, 0.3, 0.1])
    head = EnsembleHead(1, 4, output_dim(method, 5), output="coral" if HEAD_OF[method] == "coral" else "full")
    with torch.no_grad():
        head.W.zero_()
    init_bias_to_prior(head, method, pbar)
    pmf = log_pmf(HEAD_OF[method], head(torch.zeros(3, 4)).detach()).exp()
    assert torch.allclose(pmf, pbar.expand_as(pmf), atol=1e-4)


@pytest.mark.parametrize("shape", [(3, 6), (1, 6)])        # dense head, and CORAL's single shared weight vector
def test_fold_standardisation_is_exact(shape):
    from ordbench.finetune import _fold_standardisation
    g = torch.Generator().manual_seed(0)
    W, b = torch.randn(*shape, generator=g, dtype=torch.float64), torch.randn(3, generator=g, dtype=torch.float64)
    x = torch.randn(20, 6, generator=g, dtype=torch.float64) * 3 + 2
    mu, sd = x.mean(0), x.std(0)
    W2, b2 = _fold_standardisation(W, b, mu, sd)
    assert torch.allclose(x @ W2.T + b2, ((x - mu) / sd) @ W.T + b)


@pytest.mark.parametrize("method", ["soft_corn", "soft_coral", "ce_soft"])
def test_linear_probe_init_fits_features_and_keeps_head_usable(method):
    """The LP stage of LP-FT must leave `head` predicting from raw (unstandardised) features at least as
    well as the training-label marginal, whichever parameterisation is probed."""
    from ordbench.finetune import init_bias_to_prior, linear_probe_init
    from ordbench.losses import HEAD_OF, log_pmf, output_dim
    from ordbench.metrics import evaluate
    from ordbench.train import make_evalset, target_from_mask

    class Toy:                                     # stands in for a backbone: features are the inputs
        def __init__(self, X):
            self.X, self.lengths = X, np.ones(len(X), int)

        def batch(self, idx, device):
            return (self.X[idx],)

    g = np.random.default_rng(0)
    N, d, K = 600, 5, 4
    X = torch.as_tensor(g.normal(size=(N, d)) * 5 + 3, dtype=torch.float32)
    mask = np.zeros((N, K), bool)
    mask[np.arange(N), np.clip((X[:, 0].numpy() / 5 + 0.6 + 0.3 * g.normal(size=N)) * 2 + 1, 0, K - 1).astype(int)] = True
    Ttr = target_from_mask(mask)
    tr, va = np.arange(400), np.arange(400, 600)
    val = make_evalset(np.zeros((len(va), 1)), mask[va])
    val["idx"] = va
    head = EnsembleHead(1, d, output_dim(method, K), body="linear",
                        output="coral" if HEAD_OF[method] == "coral" else "full")
    init_bias_to_prior(head, method, Ttr.p[tr].mean(0))
    enc = torch.nn.Identity()
    data = Toy(X)
    prior_nll = float(evaluate(log_pmf(HEAD_OF[method], head(X[va]).detach()), val)["nll"])
    wd, plr = linear_probe_init(head, method, enc, data, tr, Ttr, val, "cpu", steps=300)
    nll = float(evaluate(log_pmf(HEAD_OF[method], head(X[va]).detach()), val)["nll"])
    assert wd in (0.01, 0.1, 1.0, 10.0) and plr == 1e-2
    wd2, plr2 = linear_probe_init(head, method, enc, data, tr, Ttr, val, "cpu", wds=(0.001, 0.1), lrs=(3e-3, 3e-2), steps=200)
    assert wd2 in (0.001, 0.1) and plr2 in (3e-3, 3e-2)
    assert nll < prior_nll - 0.05, (prior_nll, nll)


def test_text_encoder_trains_in_fp32_even_if_the_checkpoint_is_fp16(tmp_path):
    """transformers 5 loads a checkpoint in its stored dtype, and DeBERTa-v3-base is stored in fp16: AdamW's eps = 1e-8
    underflows to zero there and the first step turns every weight into NaN. The encoder must hold fp32 weights."""
    transformers = pytest.importorskip("transformers")
    from ordbench.finetune import TextEncoder
    cfg = transformers.BertConfig(hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32,
                                  vocab_size=50)
    transformers.BertModel(cfg).half().save_pretrained(tmp_path)
    enc = TextEncoder(str(tmp_path))
    assert {p.dtype for p in enc.parameters()} == {torch.float32}
    ids, mask = torch.randint(0, 50, (2, 6)), torch.ones(2, 6, dtype=torch.long)
    opt = torch.optim.AdamW(enc.parameters(), lr=1e-3)
    enc(ids, mask).pow(2).mean().backward()
    opt.step()
    assert all(torch.isfinite(p).all() for p in enc.parameters())


@pytest.mark.parametrize("has_output_dim", [True, False])
def test_image_encoder_reads_feature_size_with_or_without_output_dim(monkeypatch, has_output_dim):
    """open_clip's SigLIP vision tower is a timm model without `output_dim`; the encoder must still find its feature size."""
    open_clip = pytest.importorskip("open_clip")
    from ordbench import finetune

    class Tower(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(3 * 8 * 8, 12)
            if has_output_dim:
                self.output_dim = 12

        def forward(self, x):
            return self.proj(x.flatten(1))

    stub = type("Clip", (), {"visual": Tower()})()
    monkeypatch.setattr(open_clip, "create_model_and_transforms", lambda arch, pretrained: (stub, None, None))
    monkeypatch.setattr(finetune, "image_preproc", lambda spec: (8, (0.5,) * 3, (0.5,) * 3))
    enc = finetune.ImageEncoder("Stub-Arch/none")
    assert enc.dim == 12
    assert enc.v.training
