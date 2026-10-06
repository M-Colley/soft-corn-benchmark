"""Shared training loop for the real-data and synthetic benchmarks.

`run_cell` trains every requested method on one training set. Each method is a vectorised
ensemble with one member per (learning rate, weight decay) configuration. All methods see the
same minibatch sequence and the same initial weights (common random numbers), so differences
between methods are not confounded by data order or initialisation. Metrics for every member
are logged on every evaluation set every `eval_every` steps. Model selection (which config
and which step) is done afterwards from the validation logs, so every selection rule is
applied to exactly the same runs.
"""
import itertools
from collections import defaultdict

import numpy as np
import torch

from .heads import EnsembleHead, MemberAdam
from .losses import HEAD_OF, Target, compute_loss, log_pmf, output_dim, rank_violation
from .metrics import evaluate, prepare

LRS = (3e-4, 1e-3, 3e-3, 1e-2, 3e-2)
WDS = (1e-3, 1e-2, 1e-1, 1.0, 10.0)     # L2 penalty (coupled weight decay)
CONFIGS = list(itertools.product(LRS, WDS))


def target_from_mask(mask):
    """Uniform-over-set soft label, and the lower median of the set as the hard label."""
    mask = torch.as_tensor(mask).bool()
    p = mask.float() / mask.sum(-1, keepdim=True)
    size = mask.sum(-1)
    order = torch.cumsum(mask.long(), -1)                   # rank of each accepted level
    y = ((order == ((size - 1) // 2 + 1)[:, None]) & mask).float().argmax(-1)
    return Target(p, mask, y)


def batch_schedule(n, batch, steps, seed):
    """Indices for `steps` minibatches drawn by reshuffling every epoch."""
    g = torch.Generator().manual_seed(seed)
    batch = min(batch, n)
    per_epoch = n // batch
    out = []
    while len(out) < steps:
        perm = torch.randperm(n, generator=g)
        out.extend(perm[i * batch:(i + 1) * batch] for i in range(per_epoch))
    return torch.stack(out[:steps])


def make_evalset(X, mask, p=None, ho=None, device="cpu"):
    """p defaults to the uniform-over-set label; pass the population target in synthetic runs."""
    t = target_from_mask(mask)
    p = t.p if p is None else torch.as_tensor(p, dtype=torch.float32)
    ho = None if ho is None else torch.as_tensor(ho, dtype=torch.float32, device=device)
    ev = prepare(p.to(device), t.mask.to(device), ho)
    ev["X"] = torch.as_tensor(X, dtype=torch.float32, device=device)
    return ev


def prior_metrics(mask_tr, evalsets):
    """Metrics of the feature-free predictor that outputs the mean training soft label."""
    prior = target_from_mask(mask_tr).p.mean(0).clamp_min(1e-12)
    out = {}
    for name, ev in evalsets.items():
        lp = prior.log().to(ev["p"].device).expand(1, len(ev["p"]), -1)
        out[name] = {k: float(v[0]) for k, v in evaluate(lp, ev).items()}
    return out


def run_cell(X_tr, mask_tr, evalsets, K, methods, body="linear", seed=0, steps=2000,
             eval_every=50, batch=256, configs=CONFIGS, device="cuda", use_graph=True):
    """Returns {method: {split: {metric: array (n_evals, M)}}} and the evaluated step numbers."""
    Xtr = torch.as_tensor(X_tr, dtype=torch.float32, device=device)
    Ttr = target_from_mask(mask_tr)
    Ttr = Target(Ttr.p.to(device), Ttr.mask.to(device), Ttr.y.to(device))
    sched = batch_schedule(len(Xtr), batch, steps, seed).to(device)
    lrs = [c[0] for c in configs]
    wds = [c[1] for c in configs]
    M = len(configs)
    eval_steps = list(range(eval_every, steps + 1, eval_every))
    results = {}
    for method in methods:
        torch.manual_seed(seed)            # dropout masks
        head = EnsembleHead(M, Xtr.shape[1], output_dim(method, K), body=body,
                            output="coral" if HEAD_OF[method] == "coral" else "full",
                            seed=seed).to(device)
        opt = MemberAdam(list(head.named_parameters()), lrs, wds, head.decay_mask())
        idx = sched[0].clone()

        def train_step():
            loss = compute_loss(method, head(Xtr[idx]), Ttr.index(idx))
            loss.sum().backward()
            opt.step()
            return loss

        graph = _capture(train_step, head, opt) if (use_graph and device == "cuda") else None
        log = defaultdict(lambda: defaultdict(list))
        for step in range(1, steps + 1):
            idx.copy_(sched[step - 1])
            if graph is not None:
                graph[0].replay()
                loss = graph[1]
            else:
                opt.zero_grad()
                loss = train_step()
            if step % eval_every == 0:
                head.eval()
                with torch.no_grad():
                    log["train"]["loss"].append(loss.detach().cpu().numpy())
                    for name, ev in evalsets.items():
                        out = head(ev["X"])
                        lp = log_pmf(HEAD_OF[method], out)
                        met = evaluate(lp, ev)
                        met["rank_viol"] = rank_violation(HEAD_OF[method], out).mean(-1)
                        vals = torch.stack([v.float() for v in met.values()]).cpu().numpy()
                        for k, v in zip(met, vals):
                            log[name][k].append(v)
                head.train()
        results[method] = {s: {k: np.stack(v) for k, v in d.items()} for s, d in log.items()}
    return results, np.array(eval_steps)


def _capture(train_step, head, opt, warmup=3):
    """Record one optimisation step as a CUDA graph (torch.cuda.CUDAGraph), so a step costs one
    launch instead of ~1000 when kernel launches dominate. Warm-up steps run on a side stream as
    the PyTorch docs require; parameters and optimiser state are then restored, so the graph
    replays start from the same initial point as an uncaptured run."""
    init = [p.detach().clone() for p in head.parameters()]
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup):
            opt.zero_grad()
            train_step()
    torch.cuda.current_stream().wait_stream(s)
    opt.zero_grad()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        loss = train_step()
    with torch.no_grad():
        for p, p0 in zip(head.parameters(), init):
            p.copy_(p0)
    opt.reset()
    return g, loss


def flatten(results):
    """{method: {split: {metric: arr}}} -> {"method/split/metric": arr} for np.savez."""
    return {f"{m}/{s}/{k}": v for m, d in results.items() for s, dd in d.items() for k, v in dd.items()}
