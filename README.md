# soft-CORN benchmark

[![tests](https://github.com/M-Colley/soft-corn-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/M-Colley/soft-corn-benchmark/actions/workflows/ci.yml)

**Results and verdict: [REPORT.md](REPORT.md)** (tables in `results/`, figures in `figures/`).
**Next experiments: [FUTURE_WORK.md](FUTURE_WORK.md).** Notes for Claude Code: [CLAUDE.md](CLAUDE.md).

Benchmarks soft-CORN (CORN with expected subset membership under a soft label) against
soft-CORAL / soft-OR (Matton et al., 2026), ordinal label-distribution learning (Wen et al.,
ICCV 2023), partial-label likelihood, soft and hard cross-entropy, and hard CORN / CORAL,
on set-valued ordinal labels built from public multi-annotator datasets and on synthetic data
with a known population target.

Storage: raw data, features, run logs and model caches live outside the repository, under the data
root given by the `SCB_DATA` environment variable. If it is unset, the root is
`D:/soft-corn-benchmark-data` when that exists (the original machine), otherwise `./data`
(git-ignored); see `ordbench/paths.py`. Only code, small tables (`results/`) and figures (`figures/`)
are committed.

## Setting up on another machine

```bash
git clone <your-repo-url> soft-corn-benchmark && cd soft-corn-benchmark
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126   # match your CUDA
pip install -r requirements.txt          # or requirements-lock.txt for the exact versions of the last full run
export SCB_DATA=/path/with/space/soft-corn-benchmark-data  # Windows: setx SCB_DATA D:\...
python -m pytest -q                                         # should pass (the CUDA-only test needs a GPU)
```

RTX 50-series (Blackwell) cards need a CUDA 12.8+ build of PyTorch, e.g. `--index-url https://download.pytorch.org/whl/cu130`.

Then either rebuild the data:

```bash
python scripts/01_download.py && python scripts/02_build.py
```

or copy `processed/` (about 0.3 GB) from the old data root, which is needed to merge results across
machines exactly. To continue earlier work, also copy `results/` (about 0.4 GB of run logs): the
runners skip finished cells and `05_analyze.py` analyses everything present.

Continuous integration (`.github/workflows/ci.yml`) runs ruff and the test suite on CPU, on Ubuntu
(Python 3.11, 3.13, 3.14, 3.15) and Windows (Python 3.13, 3.14, 3.15), using `requirements-ci.txt`.

## Pipeline

```bash
python scripts/01_download.py        # official releases / canonical HF copies
python scripts/02_build.py           # frozen-encoder features + set-valued labels
python scripts/03_run.py             # real-data benchmark (resumable; skips finished cells)
python scripts/04_synthetic.py       # synthetic benchmark (resumable)
python scripts/06_finetune.py        # fine-tuned backbones (resumable)
python scripts/07_checks.py          # grid-edge checks, hyperparameter sensitivity, fine-tuning variants -> results/checks.md
python scripts/05_analyze.py         # selection, tables, tests, figures -> results/, figures/
python -m pytest -q                  # 114 tests incl. a CPU end-to-end pipeline smoke test

# extension studies (FUTURE_WORK.md)
python scripts/08_replay.py          # predictions at the selected checkpoints of the real linear-head cells
python scripts/09_setpred.py         # item-level bootstrap, conformal sets, reliability, temperature scaling -> results/setpred.md
python scripts/10_extensions.py      # targets, baselines, synthetic variants, extra datasets, shift, personalisation -> results/extensions.md
python scripts/11_cost.py            # compute cost per method -> results/cost.md
python scripts/12_shift.py           # leave-one-group-out distribution shift
python scripts/13_personal.py        # per-rater offsets on SCUT
python scripts/01_download.py vader convabuse && python scripts/02_build.py vader_tweets vader_nyt vader_movie vader_amazon convabuse
python scripts/01_download.py provoice && python scripts/02_build.py provoice provoice_task provoice_centred provoice_known   # in-vehicle study, 0.55 GB
bash scripts/run_provoice.sh python  # its runs (driver-level splits, --m 0) and scripts/14_provoice.py -> results/provoice.md
bash scripts/run_provoice.sh python kcurve kanalyze && bash scripts/run_kcontrol.sh python   # K-curve personalisation (~3 h) -> results/kcurve.md
```

`scripts/run_frozen.sh`, `scripts/run_finetune.sh` and `scripts/run_extras.sh` run every stage of the frozen-feature,
fine-tuning and extension experiments one after another (resumable; stages are listed in the script headers).
Run them sequentially: two jobs on one GPU time-slice it and finish no sooner than one after the other.

## Design

**Datasets** (`02_build.py`). For each item, m = 3 annotators are drawn once (fixed seed); the
set-valued label is the set of levels they chose, and the remaining annotators are kept as
held-out raters that never enter training or model selection.

| dataset | K | items (≥3 ratings) | features |
|---|---|---|---|
| Measuring Hate Speech, `sentiment` (Kennedy et al., 2020) | 5 | 17,352 | all-mpnet-base-v2 |
| Wikipedia Toxicity (Wulczyn et al., 2017), 40k-comment sample | 5 | 40,000 | all-mpnet-base-v2 |
| Humicroedit funniness (Hossain et al., 2020) | 4 | 15,095 | mpnet(edited), mpnet(edited) − mpnet(original) |
| CommitmentBank (de Marneffe et al., 2019) | 7 | 1,200 | mpnet(context+target), mpnet(prompt) |
| SCUT-FBP5500 (Liang et al., 2018), 60 raters | 5 | 5,500 | OpenCLIP ViT-B/16 (LAION-2B) |
| *extension:* VADER sentiment (Hutto & Gilbert, 2014), four corpora, ≥ 15 raters | 9 | 4,200 / 5,190 / 10,605 / 3,708 | all-mpnet-base-v2 |
| *extension:* ConvAbuse (Cercas Curry et al., 2021), 2–7 annotators | 5 | 4,185 (3,091 with ≥ 3) | mpnet(user utterance), mpnet(exchange) |
| *extension:* ProVoice study 1 (in-vehicle assistant; CC-BY-NC-4.0), 12 drivers, **native sets** (6.8% of answers) | 5 | 1,446 | 45 window summaries of driver state and vehicle, task code |

**Synthetic** (`ordbench/synthetic.py`). Rater distributions π(x) from five families with logits
linear in x: proportional odds (CORAL's model), continuation ratio (CORN's), multinomial
logit (softmax's), location-scale (x-dependent spread), and a polarised mixture (gapped sets).
Sets come from m = 3 raters, as on real data. The population target of every soft loss,
t(x) = E[u_S | x], is computed exactly by enumerating all K^m rating tuples, so the
excess risk KL(t ‖ p̂) can be measured without noise.

**Extension methods** (`TARGET_METHODS`, `BASELINE_METHODS`; run into their own result folders): training targets built
from the ratings (empirical distribution, 50/50 mixture, exact set likelihood) for the CORN, softmax and CORAL
parameterisations; SORD (Laplace, Gaussian), CDW-CE, binomial and Poisson heads, cumulative link models (logit, probit,
cloglog, learned Aranda–Ordaz link). **Extension data:** synthetic variants (K, d, signal, nonlinear generators, noise,
rater heterogeneity, interval / threshold / mirror set mechanisms), leave-one-group-out shift, SCUT personalisation,
natively set-valued tables (`ordbench/native.py`).

**Methods** (`ordbench/losses.py`, 12 core). soft-CORN; soft-CORAL and soft-OR; soft CE;
OLDL-CAD / QFD / CJS (ported from github.com/downdric/OLDL and checked against it);
PLL-softmax / PLL-CORN (−log P(y ∈ S)); hard CE / CORN / CORAL on the lower median of the set.
CORN and CORAL use `coral-pytorch`.

**Training protocol** (`ordbench/train.py`). Heads on frozen features: linear (all training sizes),
and as a robustness check a 1-hidden-layer MLP (128 ReLU units, dropout 0.2; real data at
n = 512 and the full pool, synthetic at n = 1,024 and 16,384). Adam with an L2 penalty on weights (biases unpenalised), grid
lr ∈ {3e-4, 1e-3, 3e-3, 1e-2, 3e-2} × L2 ∈ {1e-3, 1e-2, 1e-1, 1, 10}. All 25 configurations
train as one vectorised ensemble with torch's fused Adam per member, captured in a CUDA graph.
Batch 256, 2,000 steps, evaluated every 100 steps; TF32 matmuls. All methods share data order and
initialisation. The step and configuration are chosen offline on the validation split
(by set-MAE as in the thesis, by NLL, or by RPS), and every rule is applied to the same runs.
Splits: 20% test, 10% validation; training sizes {128, 512, 2048, 8192, full}, nested; 5 seeds.

**Fine-tuned backbones** (`ordbench/finetune.py`, `scripts/06_finetune.py`).
- Backbones: all-mpnet-base-v2 for text (mean pooling; sentence pairs for Humicroedit and
  CommitmentBank) and the OpenCLIP ViT-B/16 image tower for SCUT, each fine-tuned end to end
  with each loss.
- Setup: soft-CORN, soft-CORAL and soft CE; m = 3 labels; 2,048 training items (CommitmentBank's
  840), so the results match a frozen-feature cell.
- Optimiser: AdamW (weight decay 0.01), bf16, linear warm-up and decay, 5 epochs (8 for
  CommitmentBank), batch 32 with length-grouped batching. The backbone learning rate is
  {1e-5, 3e-5}, chosen per (dataset, method) on seed 0's validation NLL; the head uses 1e-3.
- Head initialisation: biases start at the training-label marginal (log p̄, logit P(y>k), or
  logit P(y>k | y≥k)), so every parameterisation starts from the same prior.
- Epoch: selected on validation. Seeds: 3.

**Metrics** (`ordbench/metrics.py`). Set-MAE / set-accuracy (median and mode decoding), NLL
against the soft target, partial-label NLL, RPS, EMD, spread correlation, held-out-rater
NLL / RPS / EMD / MAE, and per set type (singleton / contiguous / gapped). Calibration and
set-valued predictions: ordinal ECE against held-out raters (`ho_ece`) and, for the narrowest
contiguous interval holding 50 / 80 / 90 % of the predicted mass, its width, the share of held-out
ratings it covers, its Jaccard distance to the observed set and an interval score
(`int{50,80,90}_{width,hocov,jdist,iscore}`).

**Statistics** (`scripts/05_analyze.py`). Per problem: Nadeau–Bengio corrected resampled t-test
over seeds. Across problems: Friedman + Nemenyi (autorank) and the Bayesian signed-rank test
with a region of practical equivalence (baycomp).

## Status
Last full run: 2026-10-10 / 11 on a second machine (RTX 5080, data root `D:/soft-corn-benchmark-data`),
everything re-run from the raw data. Complete:
- linear heads: all real datasets (10 seeds × all sizes) and all synthetic families (10 seeds × 4 sizes);
- MLP heads: real n ∈ {512, full} and synthetic n ∈ {1,024, 16,384}, 10 seeds each;
- robustness checks, analysed by `scripts/07_checks.py` into `results/checks.md`: extended L2 grid on
  CommitmentBank / Humicroedit (`results/wd_check`), a low-learning-rate MLP grid
  (`results/mlp_lr_check`), hyperparameter sensitivity, and the fine-tuning learning-rate and
  head-initialisation checks;
- annotators-per-set ablation, m ∈ {2, 5}, linear heads, n ∈ {512, 2048, full}, 5 seeds
  (`results/m_ablation`; m = 5 excludes MHS, which has too few items with five ratings);
- fine-tuned backbones (5 datasets, 5 seeds): soft-CORN, soft-CORAL and soft CE at n = 2,048
  (CommitmentBank 840) and n = 512, main protocol (`results/finetune`) and LP-FT (`results/finetune_lpft`); reference
  methods at n = 2,048; backbone learning rate 1e-4 as a check (`results/finetune_lr_check`);
- extension studies (REPORT.md sections 4.1 and 5.1–5.7): richer synthetic variants (`results/synth_*`), training
  targets (`targets_*`), further baselines (`baselines_*`), VADER and ConvAbuse (`real_extra`), distribution shift
  (`shift`), personalisation (`personal`), replay and conformal sets (`replay`), compute cost.

Also complete: fine-tuning with a tuned LP-FT probe, and with DeBERTa-v3-base, RoBERTa-large, DINOv2 and SigLIP (3 seeds,
`scripts/run_finetune.sh`); the in-vehicle study with natively set-valued labels (`results/provoice*`, REPORT.md section 5.8:
four feature variants, 20 driver-level or window-level splits) and its K-curve (REPORT.md section 5.9: K = 0 ... 40 calibration
answers of a held-out driver, 12 drivers, 3 seeds, plus a control that removes the set information). Open: fine-tuning seeds 5–9
for the LP-FT variants; fine-tuning at 8,192 items and the full pools; the thesis's L2-SP and ANIL arms; the rest of `FUTURE_WORK.md`.

Background tool jobs may be killed after 2 h, so long runs were split by stage and `--seeds`. The
runners skip finished cells. `03_run.py` accepts `--lrs` / `--wds` to override the grid. Run GPU
stages one at a time (`scripts/run_frozen.sh`, `scripts/run_finetune.sh`, `scripts/run_extras.sh`).

## Decisions taken along the way
- The C: drive was full, so all data moved to D:.
- Pilot runs showed that AdamW's decoupled weight decay does not regularise high-dimensional
  linear heads. Adam's sign-like steps push weights to about ±1/wd, and every method overfitted
  CommitmentBank to below the no-feature baseline. The benchmark therefore uses a coupled L2
  penalty (ridge-type fixed point, the usual linear-probe protocol). The grid was moved until
  most selections were interior.

## Positioning note
soft-CORN with uniform-over-set targets is the cross-entropy of the CORN-decoded PMF, i.e. the
grouped-data (fractional-count) likelihood of the continuation-ratio / sequential logit model
(Fienberg; Tutz). The derivation is correct, but the loss itself is not new; the question is
empirical.
