"""Output heads trained as a vectorised ensemble: M independent members (one per
hyperparameter configuration) share each minibatch but own separate parameters, stacked along
a leading member axis. `MemberAdam` applies torch.optim.Adam's (or AdamW's) update with a
per-member learning rate and weight decay; tests/test_heads.py checks it against the torch
optimiser run separately on each member.
"""
import math

import torch
import torch.nn as nn


def _uniform(shape, fan_in, gen):
    bound = 1.0 / math.sqrt(fan_in)
    return (torch.rand(shape, generator=gen) * 2 - 1) * bound


class EnsembleHead(nn.Module):
    """x (N, d) -> outputs (M, N, out).

    body:   "linear" (no hidden layer) or "mlp" (one ReLU hidden layer with dropout).
    output: "full"  -> a dense (h, out) layer (softmax / CORN / OR parameterisations);
            "coral" -> CORAL's shared (h, 1) weight plus K-1 free biases, initialised as in
                       coral_pytorch.layers.CoralLayer (preinit_bias=True). With out = 1 it is a
                       single-score head (unimodal binomial / Poisson outputs);
            "clm"   -> cumulative link model: one score s(x) and `out` strictly decreasing thresholds
                       b_k = b_0 - sum_{j<=k} softplus(d_j), output s(x) + b_k;
            "clm_ao" -> the same plus one learned link parameter, appended as the last output
                       (`out` counts the thresholds plus that parameter).
    Weights are stored as nn.Linear does, (M, out_features, in_features), with nn.Linear's
    uniform(-1/sqrt(fan_in), 1/sqrt(fan_in)) initialisation; the first layer of all members is
    then a single GEMM against the shared input with no weight copies.
    """

    def __init__(self, M, d_in, out, body="linear", output="full", hidden=128, dropout=0.2, seed=0, n_aux=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.M, self.body, self.output = M, body, output
        self.dropout = nn.Dropout(dropout)
        h = d_in
        if body == "mlp":
            self.W1 = nn.Parameter(_uniform((M, hidden, d_in), d_in, g))
            self.b1 = nn.Parameter(_uniform((M, hidden), d_in, g))
            h = hidden
        if n_aux:
            # per-annotator offsets of the output logits (rater bias); a zero-initialised table shrunk by the L2 penalty,
            # one extra row can stay untrained as the offset of annotators never seen
            self.Woff = nn.Parameter(torch.zeros(M, n_aux, out))
        if output in ("clm", "clm_ao"):
            n_thr = out - 1 if output == "clm_ao" else out
            self.W = nn.Parameter(_uniform((M, 1, h), h, g))
            self.b0 = nn.Parameter(torch.full((M, 1), (n_thr - 1) / 2))
            self.d = nn.Parameter(torch.full((M, max(n_thr - 1, 0)), 0.5413))      # softplus(0.5413) = 1
            if output == "clm_ao":
                self.lam = nn.Parameter(torch.full((M, 1), 0.5413))                # link parameter = 1: logit
        elif output == "coral":
            self.W = nn.Parameter(_uniform((M, 1, h), h, g))
            self.b = nn.Parameter((torch.arange(out, 0, -1).float() / out).repeat(M, 1))
        else:
            self.W = nn.Parameter(_uniform((M, out, h), h, g))
            self.b = nn.Parameter(_uniform((M, out), h, g))

    @staticmethod
    def _shared_input(x, W):
        """x (N, d), W (M, o, d) -> (M, N, o) as one (N, d) @ (d, M*o) GEMM."""
        M, o, d = W.shape
        return (x @ W.reshape(M * o, d).T).view(-1, M, o).transpose(0, 1)

    def forward(self, x, aux=None):
        """aux: optional (N,) annotator indices selecting the rows of the offset table (heads built with n_aux)."""
        if self.body == "mlp":
            hdn = self._shared_input(x, self.W1) + self.b1[:, None, :]
            hdn = self.dropout(torch.relu(hdn))
            out = torch.bmm(hdn, self.W.transpose(1, 2))
        else:
            out = self._shared_input(x, self.W)
        if self.output in ("clm", "clm_ao"):
            steps = torch.cat([torch.zeros_like(self.b0), torch.cumsum(torch.nn.functional.softplus(self.d), -1)], -1)
            z = out + (self.b0 - steps)[:, None, :]                      # (M, N, thresholds), decreasing in k
            if self.output == "clm_ao":
                z = torch.cat([z, self.lam[:, None, :].expand(-1, z.shape[1], -1)], -1)
            return z
        out = out + self.b[:, None, :]
        if aux is not None:
            out = out + self.Woff[:, aux, :]
        return out

    def decay_mask(self):
        """Weights are decayed, biases are not (the usual AdamW convention)."""
        return {n: n.startswith("W") for n, _ in self.named_parameters()}


class MemberAdam:
    """torch.optim.Adam (or AdamW) with one parameter group per ensemble member, so member m
    trains with learning rate lr[m] and weight decay wd[m]; biases are never decayed.

    decoupled=False (default): Adam with an L2 penalty wd/2 ||W||^2 added to the loss, so for a
        linear head the fixed point is the ridge-regularised minimiser (the usual linear-probe
        protocol). Decoupled decay (AdamW) regularises high-dimensional linear heads poorly:
        Adam's sign-like steps drive every weight with a consistent gradient sign to |w| ~ 1/wd,
        whatever the size of its gradient.
    decoupled=True: torch.optim.AdamW.

    The optimiser holds, for every stacked parameter, one leaf view per member that aliases the
    stacked storage; before each step those views receive the matching slices of the stacked
    gradient. On CUDA the fused, capturable kernel is used, so the whole update is one pass over
    memory and can be recorded in a CUDA graph."""

    def __init__(self, named_params, lr, wd, decay, decoupled=False, betas=(0.9, 0.999), eps=1e-8):
        self.stacked = [p for _, p in named_params]
        flags = [decay[n] for n, _ in named_params]
        M = self.stacked[0].shape[0]
        self.views = [[nn.Parameter(p.data[m]) for m in range(M)] for p in self.stacked]
        groups = []
        for m in range(M):
            dec = [v[m] for v, f in zip(self.views, flags) if f]
            nodec = [v[m] for v, f in zip(self.views, flags) if not f]
            groups.append({"params": dec, "lr": float(lr[m]), "weight_decay": float(wd[m])})
            groups.append({"params": nodec, "lr": float(lr[m]), "weight_decay": 0.0})
        cuda = self.stacked[0].is_cuda
        cls = torch.optim.AdamW if decoupled else torch.optim.Adam
        self.opt = cls(groups, betas=betas, eps=eps, fused=cuda, capturable=cuda)

    def zero_grad(self):
        for p in self.stacked:
            p.grad = None

    def step(self):
        for p, vs in zip(self.stacked, self.views):
            for m, v in enumerate(vs):
                v.grad = p.grad[m]
        self.opt.step()

    @torch.no_grad()
    def reset(self):
        """Zero the moment estimates and step counts in place (keeps CUDA-graph addresses)."""
        for st in self.opt.state.values():
            for t in st.values():
                t.zero_()
