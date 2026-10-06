"""Fine-tuned-backbone experiment.

For each (dataset, training size, method) the backbone is fine-tuned with every learning rate on
seed 0; the learning rate with the best validation NLL is then used for the other seeds (run seed 0
first, or a seed without a seed-0 file falls back to trying every learning rate). Validation and
test metrics are logged after every epoch. The file stores the learning rates as the
configuration axis, so ordbench.analysis.select picks the learning rate and epoch on validation
data exactly as for the frozen-feature runs. Splits, set labels (m = 3) and held-out raters are
those of scripts/03_run.py.

    python scripts/06_finetune.py --datasets cb mhs --seeds 0 1 2
Output: $SCB_DATA/results/finetune/<dataset>_m3_s<seed>_n<n>_finetune__<method>.npz (finished files are skipped)
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ordbench.paths import RUNS  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ordbench import finetune as FT  # noqa: E402
from ordbench.train import make_evalset, prior_metrics, target_from_mask  # noqa: E402

_spec = importlib.util.spec_from_file_location("run03", Path(__file__).with_name("03_run.py"))
run03 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run03)

EPOCHS = {"cb": 8}                 # default 5; CommitmentBank (840 items) gets more passes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["cb", "scut", "humicroedit", "mhs", "wikitox"])
    ap.add_argument("--methods", nargs="+", default=["soft_corn", "soft_coral", "ce_soft"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--lrs", nargs="+", type=float, default=[1e-5, 3e-5])
    ap.add_argument("--max_train", type=int, default=2048,
                    help="training items; 2048 matches a frozen-feature cell (CommitmentBank uses its 840)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--out", default=str(RUNS / "finetune"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = "cuda"
    m = 3
    for name in a.datasets:
        z = np.load(FT.RAW.parent / "processed" / f"{name}.npz", allow_pickle=True)
        valid = z[f"valid_m{m}"]
        mask, ho, K = z[f"mask_m{m}"][valid], z[f"ho_m{m}"][valid], int(z["K"])
        inputs = FT.load_inputs(name, z["item_id"][valid])
        data = None
        epochs = a.epochs or EPOCHS.get(name, 5)
        for seed in a.seeds:
            te, va, pool = run03.split(len(mask), seed)
            tr = pool[:min(len(pool), a.max_train)]
            todo = [mt for mt in a.methods
                    if not (out / f"{name}_m{m}_s{seed}_n{len(tr)}_finetune__{mt}.npz").exists()]
            if not todo:
                continue
            if data is None:
                t0 = time.time()
                data = FT.make_data(name, inputs)
                print(f"[{name}] inputs ready in {time.time() - t0:.0f}s", flush=True)
            evalsets = {"val": make_evalset(np.zeros((len(va), 1)), mask[va], device=dev),
                        "test": make_evalset(np.zeros((len(te), 1)), mask[te], ho=ho[te], device=dev)}
            evalsets["val"]["idx"], evalsets["test"]["idx"] = va, te
            evalsets["_train_target"] = target_from_mask(mask)
            prior = prior_metrics(mask[tr], {k: evalsets[k] for k in ("val", "test")})
            for method in todo:
                f = out / f"{name}_m{m}_s{seed}_n{len(tr)}_finetune__{method}.npz"
                lrs = a.lrs
                # learning rate chosen once per (dataset, size, method) on seed 0's validation NLL and
                # reused for every other seed, also when those seeds run in a later call
                f0 = out / f"{name}_m{m}_s0_n{len(tr)}_finetune__{method}.npz"
                if seed != 0 and f0.exists():
                    z0 = np.load(f0, allow_pickle=False)
                    j = int(np.nanargmin(np.nanmin(z0[f"{method}/val/nll"], axis=0)))
                    lrs = [json.loads(str(z0["meta"]))["configs"][j][0]]
                res = {}
                for lr in lrs:
                    t0 = time.time()
                    print(f"{f.stem}  lr={lr:g}", flush=True)
                    res[lr] = FT.train_run(name, method, K, data, tr, evalsets, lr, epochs, seed, head_lr=a.head_lr, device=dev,
                                           log=lambda s: print(s, flush=True))
                    print(f"    done in {time.time() - t0:.0f}s", flush=True)
                arrays = {}
                for split in ("val", "test"):
                    for k in res[lrs[0]][split]:
                        arrays[f"{method}/{split}/{k}"] = np.stack([res[lr][split][k] for lr in lrs], 1)
                meta = dict(dataset=name, m=m, seed=seed, n_train=len(tr), pool=len(pool), body="finetune",
                            K=K, n_val=len(va), n_test=len(te), configs=[(lr, 0.01) for lr in lrs],
                            methods=[method], prior=prior, epochs=epochs, head_lr=a.head_lr)
                np.savez_compressed(f, steps=np.arange(1, epochs + 1), meta=json.dumps(meta), **arrays)


if __name__ == "__main__":
    main()
