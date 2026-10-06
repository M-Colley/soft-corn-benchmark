# Future work: towards a comprehensive evaluation

This lists what is still worth testing, in priority order. It is written for the next person (or the
next Claude Code session) who runs the benchmark on a new machine.

**What the evidence says so far** (REPORT.md):
- soft-CORN is a correct but known likelihood.
- It ties soft cross-entropy everywhere.
- It ties soft-CORAL on likelihood with frozen features, but beats it on set-MAE, at large n and
  on gapped sets.
- It beats soft-CORAL clearly once the backbone is fine-tuned.
- On synthetic data, each parameterisation wins on its own generating model.

**What a comprehensive evaluation still has to show:**
1. that the fine-tuning result holds with enough seeds, sizes, baselines and backbones to be a
   headline;
2. that the conclusions hold on natively set-valued labels and under other set mechanisms;
3. that uniform-over-set is the right target, or what is;
4. how the losses compare on the set-valued predictions an application actually needs.

Each item below gives the question, the design, whether the current code can run it (**runs now**)
or **needs code**, and a rough cost. Costs are for the original laptop GPU (RTX 4050, 6 GB, about
30 W). A desktop card (RTX 4090 / A100 class) is typically 5–10× faster for fine-tuning.

---

## Tier 1: make the main claims airtight

### 1. Fine-tuning at full strength (runs now, partly needs code)
*Question:* is "soft-CORN ≫ soft-CORAL, soft-CORN ≈ soft CE under fine-tuning" robust?

Current state: 3 methods, n = 2,048, 3 seeds; only 1–2 per-dataset differences are confirmed
individually.

- **Seeds:** 5 per cell (the corrected t-test needs it), ideally 10 for the headline table.
  *Runs now:* `python scripts/06_finetune.py --datasets mhs --seeds 3 4`. Later seeds reuse seed 0's
  learning rate automatically.
- **Training sizes:** 512 and the full pool, next to the existing frozen-probe cells.
  *Runs now:* `--max_train 512`, and `--max_train 100000` for the full pool. `05_analyze.py` keeps
  sizes apart.
- **Methods:** add soft-OR, OLDL-QFD (best OLDL), PLL-CORN and CORN (hard) as references.
  *Runs now:* `--methods soft_corn soft_coral ce_soft soft_or oldl_qfd pll_corn corn_hard`.
- **Learning-rate grid:** three backbone rates {1e-5, 2e-5, 5e-5}, selected per seed rather than
  once on seed 0. *Needs code:* a flag to switch off the seed-0 reuse in `06_finetune.py`.
- **Head initialisation:** add LP-FT as a variant (fit the head on frozen features, then fine-tune
  everything; Kumar et al., ICLR 2022). This checks that the conclusions do not depend on the
  prior-bias initialisation. *Needs code:* about 30 lines in `ordbench/finetune.py`.
- *Cost:* per seed, about 1 GPU-hour for 3 methods on 5 datasets at n = 2,048 on the laptop; full
  pools are 3–10× more.

### 2. Other backbones (needs code, small)
*Question:* does soft-CORAL's handicap under fine-tuning hold across encoders?

- **Text:** DeBERTa-v3-base/large, RoBERTa-large.
- **Images:** DINOv2 ViT-B/14, SigLIP.
- *Code:* make `TEXT_MODEL` / `IMAGE_MODEL` in `ordbench/finetune.py` command-line options, and
  write the backbone name into the output file name and metadata. Large models need gradient
  checkpointing or a GPU with ≥ 16 GB.

### 3. Natively set-valued labels (needs data, most important for the application)
*Question:* do the conclusions hold when people genuinely accept several levels, rather than for
sets built from independent raters?

- **The thesis's in-vehicle user study** (accepted proactivity levels per situation) is the
  natural target. Add a builder to `02_build.py` that writes `X`, `K`, `item_id`, `mask_m*` and
  `ho_m*` (an empty `ho_*` array is fine), and a loader to `finetune.load_inputs`.
- **Other sources of interval- or range-valued ordinal labels:** check their licences.
- For the application, evaluate per-user models as well (see item 12).

### 4. What should the soft target be? (needs code)
Every soft loss here trains on the uniform distribution over the observed set, so its population
target is E[u_S | x]. But the held-out raters follow π(x), not E[u_S | x], and the m-rater mechanism
is known for the real-data sets. Compare four ways of turning the same observations into a training
signal:
- uniform over the set (current);
- the empirical distribution of the m ratings;
- the **exact set likelihood** P(S | π̂) = Σ_{T⊆S} (−1)^{|S|−|T|} π̂(T)^m, which is the correct
  coarse-data likelihood when sets come from m independent raters;
- a learned mechanism, for example a mixture of an interval model and a rater model.

*Code:*
- `02_build.py` must also store the m sampled ratings per item; today only the set and the
  held-out counts are kept.
- The new losses go in `ordbench/losses.py`.
- The synthetic study already has exact targets, which makes it the cleanest testbed.

*Expected value:* high. It decides whether uniform-over-set (shared by soft-CORN and soft-CORAL) is
the right object in the first place.

### 5. More statistical power on the frozen benchmark (runs now, cheap)
- **More seeds:** 10 instead of 5 for linear heads. Run
  `python scripts/03_run.py --bodies linear --seeds 5 6 7 8 9`, which takes about 1–2 GPU-hours.
- **Hierarchical Bayesian test across datasets:** `baycomp.HierarchicalTest` uses per-seed scores
  and models dataset heterogeneity properly. It needs `pystan` / CmdStan; add it to `05_analyze.py`.
- **Per-item paired tests,** for example a bootstrap over test items within each seed, to support
  per-dataset claims.

---

## Tier 2: broaden the evidence

### 6. More multi-annotator ordinal datasets, including larger K (needs builders)
Larger K stresses the parameterisations differently: CORN chains K−1 conditionals, while CORAL
shares one score across K−1 thresholds. Candidates (verify access and licence):
- **AVA** (image aesthetics, 10-point scores, about 200 votes per image);
- **ChaLearn LAP apparent age** (many annotators per face; bin into K = 10–20);
- **DICES** (conversational safety, many raters per item);
- **ConvAbuse** (ordinal abuse severity, multiple annotators);
- other LeWiDi (Learning with Disagreements) tasks that have an ordinal scale.

Each needs a builder in `02_build.py` and an input loader in `finetune.load_inputs`. AVA and age
data also test image backbones beyond faces.

### 7. More baselines (needs code)
- **Unimodal heads:** binomial and Poisson outputs (Beckham & Pal, 2017). Strong inductive bias.
- **Soft ordinal targets:** SORD (Díaz & Marathe, 2019), Gaussian label distributions, ordinal
  label smoothing.
- **Probabilistic ordinal embeddings** (POE; Li et al., 2021), and class-distance-weighted CE
  (CDW-CE).
- **Cumulative link models** with a learned link (logit / probit / cloglog), the proper
  generalisation of CORAL.
- **Post-hoc temperature scaling per method,** to separate discrimination from calibration.

Add each to `losses.py` with a test (`HEAD_OF`, `compute_loss`), and to `LABEL` in `05_analyze.py`.

### 8. Evaluate set-valued predictions (needs code, important for the thesis)
The application has to *predict an acceptance set*, not just a distribution.
- From each method's PMF, build highest-probability sets (contiguous for ordinal, or unrestricted)
  and **conformal prediction sets** (split conformal with an ordinal score).
- Report coverage against size, and set precision, recall and Jaccard against the observed set.
- Report coverage of held-out raters' choices.
- All of this is computable from the logged test PMFs, which are not stored today. *Code:* save the
  test log-PMF at the selected checkpoint (small), then compute these metrics in `05_analyze.py`.

### 9. Richer synthetic study (runs now for some items, needs code for others)
*Runs now:* `04_synthetic.py --m 2` / `--m 5`, and more seeds (cheap, minutes).

*Needs code:* `ordbench/synthetic.py` arguments or new families for:
- **K:** 3–10, to see whether the gap to the home model grows with K;
- **dimension and signal strength,** to see where the bias–variance crossover sits;
- **set mechanisms beyond m raters:** interval sets around a latent value with x-dependent width;
  users who accept every level above a personal threshold; adversarial gapped sets;
- **nonlinear logits** (MLP generating models), so MLP heads have something to find;
- **label noise and rater heterogeneity** (raters with their own biases).

### 10. Calibration and selection protocol (needs code, small)
- Ordinal expected calibration error on the cumulative probabilities P(y > k), reliability
  diagrams, and sharpness.
- Effect of the selection rule (set-MAE vs NLL vs RPS) for every method. The thesis protocol
  (set-MAE) damages soft-CORN's calibration more than its rivals'; quantify this across all cells.

---

## Tier 3: robustness and practicality

11. **Distribution shift.** MHS records the source platform: train on some platforms, test on
    others. Do the soft losses differ in robustness?
12. **Personalisation.** This is the thesis theme ("personalised proactivity"). Condition on
    annotator ID (annotator embeddings, or per-annotator heads) and compare against population-level
    soft labels. Evaluate on held-out annotators and on held-out items separately.
13. **Hyperparameter sensitivity and stability.** Per method: the area under the lr × L2 grid, the
    fraction of configurations within 0.01 nats of the best, and variance across seeds. All of this
    can be computed from the existing logs.
14. **Compute cost** per method (time per step, memory), which matters for the in-vehicle setting.
15. **Cross-machine reproducibility.** On the new machine, re-run a handful of existing cells with
    the copied `processed/` features. Check that the selected test metrics match within tolerance,
    then pin the environment with a lock file (`pip freeze > requirements-lock.txt`).

---

## Suggested order on a new machine

| step | what | command / work | rough cost (laptop GPU) |
|---|---|---|---|
| 0 | set up, copy `processed/` (and `results/`), run tests | see README "Setting up on another machine" | 30 min |
| 1 | reproducibility spot check (item 15) | `03_run.py --datasets cb --seeds 0 --n_train full --bodies linear --out $SCB_DATA/results/repro` | 5 min |
| 2 | fine-tuning seeds 3–4 (item 1) | `06_finetune.py --datasets <d> --seeds 3 4`, per dataset | 2 h |
| 3 | fine-tuning extra methods and sizes (item 1) | `--methods ... --max_train 512` / full pool | 6–12 h |
| 4 | target-construction study (item 4) | new losses + store ratings in `02_build.py` | 1–2 days of code, 2 h of runs |
| 5 | set-prediction evaluation (item 8) | store test PMFs, add metrics | 1 day of code |
| 6 | other backbones (item 2), larger-K datasets (item 6), richer synthetic (item 9) | builders and flags | days |

After each step: `python scripts/05_analyze.py`, update `REPORT.md`, and republish the page if it is
being shared.
