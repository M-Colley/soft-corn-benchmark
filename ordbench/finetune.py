"""End-to-end fine-tuning of the encoders used for the frozen features.

Text: sentence-transformers/all-mpnet-base-v2 (Hugging Face transformers, mean pooling); any other
Hugging Face encoder can be named with `text_model`. Images: the OpenCLIP ViT-B/16 (LAION-2B) image
tower; `image_model` names another OpenCLIP model as "arch/pretrained" (for example
"ViT-B-16-SigLIP/webli") or a Hugging Face vision model as "hf:<name>" (for example "hf:facebook/dinov2-base").
Each loss trains its own copy of the backbone, so the loss can shape the representation. The
output heads, losses and metrics are the ones used for the frozen-feature benchmark.

Head initialisation (`init`): "prior" starts every head at the training-label marginal (default);
"lpft" first fits the head on the frozen pretrained features and then fine-tunes everything
(linear probing, then fine-tuning; Kumar et al., ICLR 2022).

Items are rebuilt from the raw releases in the exact order of the processed feature files
(their `item_id`), so splits and set labels are identical to the main benchmark.
"""
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from . import extra_data
from .heads import EnsembleHead
from .losses import HEAD_OF, OUTPUT_OF, Target, compute_loss, log_pmf, output_dim
from .metrics import evaluate
from .paths import RAW

TEXT_MODEL = "sentence-transformers/all-mpnet-base-v2"
IMAGE_MODEL = ("ViT-B-16", "laion2b_s34b_b88k")
DEFAULT_IMAGE = "/".join(IMAGE_MODEL)
CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
MAX_LEN = {"cb": 256}            # default 128 tokens


# --------------------------------------------------------------------------------------------
# Inputs in processed-file order
# --------------------------------------------------------------------------------------------

def load_inputs(name, item_id):
    """Text (str or (str, str) pair) or image path for every item, aligned with `item_id`."""
    if name == "mhs":
        df = pd.read_parquet(RAW / "mhs" / "measuring-hate-speech.parquet", columns=["comment_id", "text"])
        text = df.groupby("comment_id")["text"].first()
        return [text.loc[i] for i in item_id]
    if name == "wikitox":
        com = pd.read_csv(RAW / "wikitox" / "toxicity_annotated_comments.tsv", sep="\t", index_col=0)
        t = com.loc[item_id, "comment"].str.replace("NEWLINE_TOKEN", " ").str.replace("TAB_TOKEN", " ")
        return t.str.replace(r"\s+", " ", regex=True).str.strip().tolist()
    if name == "humicroedit":
        d = RAW / "humicroedit" / "subtask-1"
        df = pd.concat([pd.read_parquet(p) for p in sorted(d.glob("*.parquet"))], ignore_index=True)
        df = df.drop_duplicates("id").set_index("id")
        return [(df.loc[i, "headline"], df.loc[i, "edited"]) for i in item_id]
    if name == "cb":
        df = pd.read_csv(RAW / "cb" / "CommitmentBank-All.csv").groupby("uID").first()
        df[["Context", "Target", "Prompt"]] = df[["Context", "Target", "Prompt"]].fillna("")
        return [((df.loc[i, "Context"] + " " + df.loc[i, "Target"]).strip(), df.loc[i, "Prompt"]) for i in item_id]
    if name == "scut":
        root = RAW / "scut" / "SCUT-FBP5500_v2" / "Images"
        return [str(root / f) for f in item_id]
    if name in extra_data.VADER:
        ids, texts, _ = extra_data.vader_items(name)
        pos = {str(i): j for j, i in enumerate(ids)}
        return [texts[pos[str(i)]] for i in item_id]
    if name == "convabuse":
        _, texts, _ = extra_data.convabuse_items()
        return [texts[int(i)] for i in item_id]
    raise ValueError(name)


class TextData:
    def __init__(self, inputs, max_len, model=TEXT_MODEL):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model)
        pairs = isinstance(inputs[0], tuple)
        a = [x[0] for x in inputs] if pairs else list(inputs)
        b = [x[1] for x in inputs] if pairs else None
        enc = self.tok(a, b, truncation=True, max_length=max_len)
        self.ids = enc["input_ids"]
        self.lengths = np.array([len(r) for r in self.ids])

    def batch(self, idx, device):
        rows = [self.ids[i] for i in idx]
        L = max(len(r) for r in rows)
        ids = torch.full((len(rows), L), self.tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(rows), L), dtype=torch.long)
        for j, r in enumerate(rows):
            ids[j, :len(r)] = torch.tensor(r)
            mask[j, :len(r)] = 1
        return ids.to(device, non_blocking=True), mask.to(device, non_blocking=True)


def image_preproc(spec=DEFAULT_IMAGE):
    """(input size, mean, std) of an image backbone given as "arch/pretrained" (OpenCLIP) or "hf:<name>"."""
    if spec == DEFAULT_IMAGE:
        return 224, CLIP_MEAN, CLIP_STD
    if spec.startswith("hf:"):
        from transformers import AutoImageProcessor
        proc = AutoImageProcessor.from_pretrained(spec[3:])
        size = proc.crop_size["height"] if getattr(proc, "crop_size", None) else proc.size["height"]
        return size, tuple(proc.image_mean), tuple(proc.image_std)
    import open_clip
    arch, tag = spec.split("/", 1)
    cfg = open_clip.get_pretrained_cfg(arch, tag) or {}
    size = open_clip.get_model_config(arch)["vision_cfg"]["image_size"]
    return size, tuple(cfg.get("mean") or CLIP_MEAN), tuple(cfg.get("std") or CLIP_STD)


class ImageData:
    """Images resized and centre-cropped once (CLIP preprocessing without normalisation), kept as
    uint8; normalisation happens on the GPU."""

    def __init__(self, paths, size=224, mean=CLIP_MEAN, std=CLIP_STD):
        from PIL import Image
        from torchvision.transforms import functional as TF
        arr = np.empty((len(paths), 3, size, size), np.uint8)
        for i, p in enumerate(paths):
            im = Image.open(p).convert("RGB")
            im = TF.center_crop(TF.resize(im, size, interpolation=TF.InterpolationMode.BICUBIC), size)
            arr[i] = np.asarray(im).transpose(2, 0, 1)
        self.x = torch.from_numpy(arr)
        self.lengths = np.ones(len(paths), int)
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

    def batch(self, idx, device):
        x = self.x[idx].to(device, non_blocking=True).float().div_(255)
        return ((x - self.mean.to(device)) / self.std.to(device),)


# --------------------------------------------------------------------------------------------
# Encoders
# --------------------------------------------------------------------------------------------

class TextEncoder(nn.Module):
    def __init__(self, model=TEXT_MODEL, grad_ckpt=False):
        super().__init__()
        from transformers import AutoModel
        self.m = AutoModel.from_pretrained(model, dtype=torch.float32)   # not the stored dtype (DeBERTa-v3 is fp16)
        if grad_ckpt:                       # trade compute for memory on the large encoders
            self.m.gradient_checkpointing_enable()
        self.dim = self.m.config.hidden_size

    def forward(self, ids, mask):
        h = self.m(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).to(h.dtype)
        return (h * m).sum(1) / m.sum(1).clamp_min(1)          # mean pooling, as sentence-transformers


class ImageEncoder(nn.Module):
    def __init__(self, spec=DEFAULT_IMAGE):
        super().__init__()
        self.hf = spec.startswith("hf:")
        if self.hf:
            from transformers import AutoModel
            self.v = AutoModel.from_pretrained(spec[3:], dtype=torch.float32)
            self.dim = self.v.config.hidden_size
        else:
            import open_clip
            arch, tag = spec.split("/", 1)
            model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=tag)
            self.v = model.visual
            # SigLIP's vision tower is a timm model without output_dim: read the feature size off a dummy forward pass
            self.dim = getattr(self.v, "output_dim", None)
            if self.dim is None:
                size = image_preproc(spec)[0]
                with torch.no_grad():
                    self.dim = self.v.eval()(torch.zeros(1, 3, size, size)).shape[-1]
                self.v.train()

    def forward(self, x):
        return self.v(pixel_values=x).pooler_output if self.hf else self.v(x)


def make(name, text_model=TEXT_MODEL, image_model=DEFAULT_IMAGE, grad_ckpt=False):
    return ImageEncoder(image_model) if name == "scut" else TextEncoder(text_model, grad_ckpt)


def make_data(name, inputs, text_model=TEXT_MODEL, image_model=DEFAULT_IMAGE):
    if name == "scut":
        return ImageData(inputs, *image_preproc(image_model))
    return TextData(inputs, MAX_LEN.get(name, 128), text_model)


# --------------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------------

@torch.no_grad()
def init_bias_to_prior(head, method, pbar, eps=1e-4):
    """Set the head's biases so that, with zero input signal, every parameterisation predicts the
    training-label marginal pbar: log pbar (softmax), logit P(y > k) (CORAL / OR) and
    logit P(y > k | y >= k) (CORN). All methods then start from the same prior predictive and only
    have to learn the input dependence; without this, CORAL's thresholds stay near their generic
    initial spacing for the few hundred steps of fine-tuning."""
    pbar = pbar.double().clamp_min(eps)
    pbar = pbar / pbar.sum()
    q = (1 - torch.cumsum(pbar, 0))[:-1].clamp(eps, 1 - eps)           # P(y > k)
    head_type = HEAD_OF[method]
    if head_type == "softmax":
        b = pbar.log()
    elif head_type in ("coral", "or"):
        b = torch.logit(q)
    elif head_type == "corn":
        q_prev = torch.cat([torch.ones(1, dtype=q.dtype), q[:-1]])
        b = torch.logit((q / q_prev).clamp(eps, 1 - eps))
    else:
        raise ValueError(head_type)
    head.b.copy_(b.to(head.b).expand_as(head.b))


def length_grouped_batches(idx, lengths, batch, gen, mega=50):
    """Shuffle, then sort by length within mega-batches of `mega` x `batch` items and shuffle the
    resulting batches (the `group_by_length` sampler of Hugging Face's Trainer): far less padding,
    same data per epoch."""
    perm = idx[torch.randperm(len(idx), generator=gen).numpy()]
    out = []
    for s in range(0, len(perm), mega * batch):
        chunk = perm[s:s + mega * batch]
        chunk = chunk[np.argsort(-lengths[chunk], kind="stable")]
        out.extend(chunk[i:i + batch] for i in range(0, len(chunk), batch))
    order = torch.randperm(len(out), generator=gen).numpy()
    return [out[i] for i in order]


def _param_groups(modules, wd):
    decay, no_decay = [], []
    for mod in modules:
        for n, p in mod.named_parameters():
            if not p.requires_grad:
                continue
            leaf = n.split(".")[-1]
            is_bias = "bias" in leaf or leaf in ("b", "b1")       # EnsembleHead biases are named b / b1
            (no_decay if p.ndim < 2 or is_bias or "norm" in n.lower() else decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


@torch.no_grad()
def encode(enc, data, idx, device, batch=128):
    """Backbone features of the items `idx`, in the order of `idx` (length-sorted batches)."""
    enc.eval()
    order = np.argsort(-data.lengths[idx], kind="stable")
    outs = []
    for s in range(0, len(idx), batch):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outs.append(enc(*data.batch(idx[order[s:s + batch]], device)).float())
    enc.train()
    z = torch.cat(outs)
    out = torch.empty_like(z)
    out[torch.as_tensor(order, device=z.device)] = z
    return out


@torch.no_grad()
def predict(enc, head, method, data, idx, device, batch=128):
    head.eval()
    out = head(encode(enc, data, idx, device, batch))
    head.train()
    return log_pmf(HEAD_OF[method], out)


@torch.no_grad()
def _fold_standardisation(W, b, mu, sd):
    """Weights and bias of x -> W x + b acting on raw features, equal to (W, b) acting on (x - mu) / sd."""
    W = W / sd
    return W, b - W @ mu


def linear_probe_init(head, method, enc, data, tr, Ttr, val, device, wds=(0.01, 0.1, 1.0, 10.0), lrs=(1e-2,),
                      steps=500, batch=256, seed=0):
    """Fit `head` on the frozen features of the pretrained encoder (the first stage of LP-FT).
    An ensemble over the grid of coupled-L2 strengths `wds` x learning rates `lrs` is trained exactly like the
    frozen-feature benchmark (standardised features, Adam, biases at the training-label marginal) and the member
    with the best validation NLL is copied into `head`, with the standardisation folded into its weights.
    Returns the chosen (L2, learning rate)."""
    from .heads import MemberAdam
    from .train import batch_schedule
    Z, Zv = encode(enc, data, tr, device), encode(enc, data, val["idx"], device)
    mu, sd = Z.mean(0), Z.std(0) + 1e-6
    Z, Zv = (Z - mu) / sd, (Zv - mu) / sd
    grid = [(w, lr) for lr in lrs for w in wds]
    M = len(grid)
    probe = EnsembleHead(M, Z.shape[1], head.b.shape[-1], body="linear", output=head.output, seed=seed).to(device)
    init_bias_to_prior(probe, method, Ttr.p[torch.as_tensor(tr)].mean(0))
    opt = MemberAdam(list(probe.named_parameters()), [g[1] for g in grid], [g[0] for g in grid], probe.decay_mask())
    T = Target(*(v[torch.as_tensor(tr)].to(device) for v in (Ttr.p, Ttr.mask, Ttr.y)))
    for idx in batch_schedule(len(Z), batch, steps, seed).to(device):
        opt.zero_grad()
        compute_loss(method, probe(Z[idx]), T.index(idx)).sum().backward()
        opt.step()
    with torch.no_grad():
        j = int(evaluate(log_pmf(HEAD_OF[method], probe(Zv)), val)["nll"].argmin())
        W, b = _fold_standardisation(probe.W[j], probe.b[j], mu, sd)
        head.W.copy_(W[None])
        head.b.copy_(b[None])
    return grid[j]


def train_run(name, method, K, data, tr, evalsets, lr, epochs, seed, batch=32, wd=0.01, warmup=0.1,
              head_lr=1e-3, device="cuda", log=print, init="prior", text_model=TEXT_MODEL, image_model=DEFAULT_IMAGE,
              grad_ckpt=False, probe=None):
    """Fine-tune one backbone with one loss and backbone learning rate `lr`. The freshly initialised
    output head trains at `head_lr` (the same for every method): at backbone learning rates
    CORAL's threshold biases, and to a lesser degree CORN's, barely move from their initial values
    within a few hundred steps, which handicaps those heads. Returns {split: {metric: (epochs,)}}."""
    from transformers import get_linear_schedule_with_warmup
    torch.manual_seed(seed)
    enc = make(name, text_model, image_model, grad_ckpt).to(device)
    Ttr = evalsets["_train_target"]
    head = EnsembleHead(1, enc.dim, output_dim(method, K), body="linear",
                        output=OUTPUT_OF[HEAD_OF[method]], seed=seed).to(device)
    init_bias_to_prior(head, method, Ttr.p[torch.as_tensor(tr)].mean(0))
    if init == "lpft":
        w, plr = linear_probe_init(head, method, enc, data, tr, Ttr, evalsets["val"], device, seed=seed, **(probe or {}))
        log(f"    linear probe: L2 {w:g}, lr {plr:g}")
    elif init != "prior":
        raise ValueError(init)
    groups = ([dict(g, lr=lr) for g in _param_groups([enc], wd)]
              + [dict(g, lr=head_lr) for g in _param_groups([head], wd)])
    opt = torch.optim.AdamW(groups, lr=lr, fused=True)
    steps_per_epoch = math.ceil(len(tr) / batch)
    sched = get_linear_schedule_with_warmup(opt, int(warmup * epochs * steps_per_epoch), epochs * steps_per_epoch)
    g = torch.Generator().manual_seed(seed)
    hist = {s: {} for s in ("val", "test")}
    for ep in range(epochs):
        for idx in length_grouped_batches(tr, data.lengths, batch, g):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = enc(*data.batch(idx, device))
            out = head(z.float())
            t = Target(*(v[torch.as_tensor(idx)].to(device) for v in (Ttr.p, Ttr.mask, Ttr.y)))
            loss = compute_loss(method, out, t).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(head.parameters()), 1.0)
            opt.step()
            sched.step()
        for split in ("val", "test"):
            ev = evalsets[split]
            met = evaluate(predict(enc, head, method, data, ev["idx"], device), ev)
            for k, v in met.items():
                hist[split].setdefault(k, []).append(float(v[0]))
        log(f"    epoch {ep + 1}/{epochs}  val nll {hist['val']['nll'][-1]:.4f}  "
            f"set-MAE {hist['val']['set_mae_med'][-1]:.4f}")
    del enc, head, opt
    torch.cuda.empty_cache()
    return {s: {k: np.array(v) for k, v in d.items()} for s, d in hist.items()}
