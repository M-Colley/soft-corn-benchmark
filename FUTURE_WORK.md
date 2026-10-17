# Future work: status and what is left

The original list had 15 items. This file records what was done in the runs of 2026-10-11 and 12, where the result is, and what
is still open. Results and conclusions are in REPORT.md (sections in brackets); every claim there traces to a table in
`results/`. Costs are for the RTX 5080 box (a 12-method frozen-feature cell takes ~12 s alone; a fine-tuning run of
2,048 items ~40 s; the machine is shared, so wall-clock times vary by 2–3×).

**What the evidence says now** (REPORT.md):
- soft-CORN is a correct but known likelihood (the cross-entropy of the CORN-decoded PMF).
- It is essentially tied with soft cross-entropy, including on calibration, conformal sets and shift.
- It ties soft-CORAL on likelihood with frozen features but beats it on set-MAE, at large n, on gapped sets and on larger K;
  soft-CORAL is better with little data and, with per-rater thresholds, for personalisation from few ratings.
- The training target (uniform-over-set against the ratings' empirical distribution or the exact set likelihood) matters
  little; rating-based targets help a little at m = 5.
- Under fine-tuning its lead over soft-CORAL is −0.051 nats with the prior-bias head initialisation, −0.009 with LP-FT and
  −0.029 when each method uses its best variant (also with a tuned LP-FT probe).
- On synthetic data each parameterisation wins on its own generating model, across K, d, signal, noise and set mechanisms.

## Status of the 15 items

| # | item | status | where |
|---|---|---|---|
| 1 | fine-tuning at full strength | **partly done**: 5 seeds, n = 512 and 2,048, soft-OR / OLDL-QFD / PLL-CORN / hard CORN references, LP-FT, a tuned LP-FT probe, learning rate 1e-4; seeds 5–9 are done for the three core methods at the main size (main protocol and LP-FT) only. **Open:** n = 8,192 and the full pools (`run_finetune.sh sizes full`; several GPU-hours per dataset), per-seed learning-rate selection (`--lr_per_seed` exists) | 5.5 |
| 2 | other backbones | **done** (3 seeds, n = 2,048, both head initialisations): DeBERTa-v3-base and RoBERTa-large on the four text datasets, DINOv2-base and SigLIP-B/16 on SCUT (`run_finetune.sh deberta roberta dinov2 siglip`; `results/finetune_<backbone>[_lpft]`, `checks.md`). Best-variant soft-CORN − soft-CORAL: −0.025, −0.020 (text), +0.002, +0.004 (SCUT). The first DeBERTa run was invalid (fp16 weights) and was repeated | 5.5 |
| 3 | natively set-valued labels | **done on the ProVoice in-vehicle study** (Hugging Face, 12 drivers, 1,446 answers, 98 sets): `ordbench/provoice.py`, `scripts/14_provoice.py`, `run_provoice.sh`; four feature variants (all features, task only, driver-centred, known driver), 20 splits each. Result: too few sets (6.8%) to separate the set-aware losses; across drivers no loss beats the label marginal; with a driver's own answers soft-CORN has the best NLL with MLP heads and soft-CORAL is worst. The K-curve (K = 0 ... 40 calibration answers of a held-out driver, 12 drivers) is also done, with one adaptation arm (a driver code with L2 shrinkage): 1-2 answers buy 0.00-0.09 nats, 5 buy 0.32, 40 buy 0.82; soft-CORN beats soft-CORAL by 0.11-0.21 from K = 10 and ties soft CE (MLP); training on a random member of each set matches training on the sets (`15_kcurve.py`, `16_kcurve.py`, `run_kcontrol.sh`). **Open:** the thesis's L2-SP and ANIL arms, the sequence features of the thesis, assistant-chosen calibration answers, a dataset with more native sets | 5.8, 5.9 |
| 4 | what should the soft target be | **done**: empirical, 50/50 and exact set likelihood for CORN / softmax / CORAL, on synthetic and real data; the uniform-over-set target is not the weak point | 5.1 |
| 5 | statistical power | **done**: 10 seeds for the frozen benchmark; item-level paired bootstrap (replay); random-effects meta-analysis as a stand-in for the Stan hierarchical test (PyStan does not install on this setup) | 5, 9 |
| 6 | more datasets, larger K | **partly done**: VADER sentiment (K = 9, four corpora) and ConvAbuse (K = 5). **Open:** AVA and ChaLearn apparent age (very large or gated), DICES (not ordinal), LeWiDi 2025 CSC / Par (download location unconfirmed) | 5.3 |
| 7 | more baselines | **done**: SORD (Laplace, Gaussian), CDW-CE, binomial and Poisson heads, cumulative link models (logit, probit, cloglog, learned link), temperature scaling. **Not done:** POE (needs a stochastic embedding trained with the backbone) | 5.2, 5.4 |
| 8 | set-valued predictions | **done**: contiguous prediction intervals (metrics in every run), split-conformal sets (ordinal APS) from replayed predictions | 5.4 |
| 9 | richer synthetic study | **done**: K 3–10, d, signal, label noise, rater heterogeneity, interval / threshold / mirror set mechanisms, nonlinear generator | 4.1 |
| 10 | calibration and selection protocol | **done**: ordinal ECE, reliability, sharpness, temperature scaling, cost of every selection rule across all cells | 5.4, 5 |
| 11 | distribution shift | **done**: MHS platforms, Wikipedia namespace and year, SCUT subsets | 5.6 |
| 12 | personalisation | **done for SCUT** (per-rater offsets; own rating against the population consensus; known and held-out raters) | 5.7 |
| 13 | hyperparameter sensitivity | **done** | 7 |
| 14 | compute cost | **done** (timed on an idle GPU after a warm-up pass) | 8 |
| 15 | cross-machine reproducibility | **done**; environment pinned in `requirements-lock.txt` | 6 |

## What is left, in priority order

1. **The thesis's other personalisation arms on the in-vehicle study.** The K-curve of section 5.9 uses a driver code with L2
   shrinkage as the only arm; the thesis compares L2-SP (fine-tuning with an L2 penalty towards the population weights) and ANIL
   (meta-learned initialisation). Implement them as extra arms on the same splits (`ordbench/kcurve.py`), with the population models
   trained once per held-out driver. Also: the sequence features of the thesis instead of window summaries, calibration answers chosen
   by an assistant instead of at random, and a native set-valued dataset in which sets are common (6.8% here); the control of
   section 5.9 says a random member of each set is as good as the set, so a dataset with wider or more frequent sets is the test.
2. **Fine-tuning at scale (item 1).** Run `scripts/run_finetune.sh <python> sizes full` (n = 8,192 and the full pools), then
   `lpft` variants at those sizes, and per-seed learning rates with `--lr_per_seed --lrs 1e-5 3e-5 1e-4`. Report with
   every method's best variant (`scripts/07_checks.py`), not the main protocol alone: the soft-CORAL gap depends on the
   head initialisation.
3. **Fine-tuning seeds 5–9 for the LP-FT, tuned-probe and reference variants** (`SEEDS5="5 6 7 8 9" bash scripts/run_finetune.sh
   <python> lpft refs`; the main protocol already has 10 seeds), and more seeds for the new backbones; then re-run
   `07_checks.py`, which intersects the seeds of all variants.
4. **More datasets (item 6).** Candidates whose access is still unconfirmed: LeWiDi 2025 CSC (sarcasm, 1–6, 4–6 observers) and
   Par (paraphrase, −5…+5, 4 annotators); check licences and download locations before adding builders to
   `02_build.py` and `ordbench/extra_data.py`.
5. **Personalisation beyond rater bias (item 12).** Annotator embeddings that interact with the features, a prior over
   offsets for unseen users, and per-user sets (the thesis's setting) once native data exist.
6. **A hierarchical Bayesian model** (Corani et al., 2017) if a Stan toolchain is available; `baycomp.HierarchicalTest`
   needs PyStan.
7. **Larger or other fine-tuning backbones** (DeBERTa-v3-large, DINOv2-large) and POE with a stochastic embedding.

## Suggested order on a new machine

| step | what | command |
|---|---|---|
| 0 | set up, copy `processed/` (and `results/`), run tests | see README "Setting up on another machine" |
| 1 | frozen-feature and synthetic benchmark | `bash scripts/run_frozen.sh` (`SEEDS="5 6 7 8 9"` for more seeds) |
| 2 | extension experiments | `bash scripts/run_extras.sh` |
| 3 | extra datasets | `python scripts/01_download.py vader convabuse`, `python scripts/02_build.py vader_tweets vader_nyt vader_movie vader_amazon convabuse`, `python scripts/03_run.py --datasets <those> --out $SCB_DATA/results/real_extra` |
| 4 | fine-tuning | `bash scripts/run_finetune.sh main seeds lpft refs lrcheck deberta dinov2 siglip roberta` |
| 5 | analysis | `05_analyze.py`, `07_checks.py`, `09_setpred.py`, `10_extensions.py`, `11_cost.py` (idle GPU) |

After each step: re-run the analysis, update `REPORT.md`, and republish the page if it is being shared.
