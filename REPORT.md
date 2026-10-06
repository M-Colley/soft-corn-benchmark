# soft-CORN: is it a stand-alone ML contribution?

Benchmark report, 2026-10-05. All numbers come from `results/summary.md` and `results/*.csv`,
which `scripts/05_analyze.py` regenerates from the run logs.

## Verdict (short)

* **Correct, but not new as a method.** soft-CORN equals cross-entropy between the soft label
  and the CORN-decoded PMF. That is the fractional-count likelihood of a continuation-ratio
  (sequential-logit) model with level-specific effects. Reviewers at an ML venue will see the
  loss itself as known.
* **Empirically it is a strong, safe choice, not a clear winner.** Linear probes on 5 datasets ×
  up to 5 training sizes (22 problems, 5 seeds) give the picture below. Ties use a region of
  practical equivalence (ROPE) of 0.005 nats.
  * vs **soft-CORAL**, the published competitor: practically tied on likelihood (NLL). soft-CORN
    is better on set-MAE (8 of 22 problems significant, 0 losses; P(better) = 0.998), on
    gapped sets (linear heads only), and at larger training sizes. soft-CORAL is better at ≤ 512 training items.
  * vs **soft cross-entropy**: indistinguishable overall (NLL: 3 wins / 15 ties / 4 losses;
    P(ROPE) = 0.85). soft CE is slightly ahead on the largest datasets.
  * Clearly better than every hard-label loss, the OLDL losses (Wen et al., ICCV 2023) and
    partial-label likelihood on NLL.
  * **With MLP heads** (10 problems, 5 seeds), the four proper soft losses (soft-CORN, soft CE,
    soft-OR, soft-CORAL) are statistically tied on NLL. soft-CORN is ahead of soft-CORAL on the
    full pool (−0.019 nats, P(better) = 0.95), but the gapped-set advantage disappears.
  * **Varying the annotators per set** (m = 2, 3, 5) leaves these conclusions unchanged. With linear
    heads, soft-CORN's lead over soft-CORAL on gapped sets grows with m, from 0.031 to 0.051 nats.
  * **With fine-tuned backbones** (5 datasets, 2,048 training items, 3 seeds), the gap to soft-CORAL
    widens. soft-CORN is ahead on all five datasets, by 0.04–0.13 nats on the four with signal
    (P(better) = 0.985). soft-CORAL gains little or nothing from fine-tuning. soft-CORN and soft CE
    remain tied.
* **The synthetic study explains the pattern.** Each output parameterisation wins on data generated
  by its own model. soft-CORN reaches about 10× lower excess risk than soft CE on continuation-ratio
  data, while soft-CORAL is about 10× better on proportional-odds data. On the two families that
  match no model, soft-CORN comes second both times. soft-CORAL's weakness is structural:
  its single shared score cannot follow item-level changes in spread, even with an MLP backbone.
* **Recommendation.** A pure "new loss" paper would be hard to place at a main ML venue. Two
  framings could work: (i) a benchmark/analysis paper on learning from set-valued ordinal labels,
  with soft-CORN as the rank-consistent CORN counterpart of soft-CORAL (venues: TMLR or a
  datasets-and-benchmarks track); or (ii) keep soft-CORN inside the application paper, with this
  benchmark as supporting evidence.

## 1. Question

The thesis proposes **soft-CORN**: CORN (Shi et al., 2023) trained on a set-valued ordinal label S.
It replaces CORN's hard subset membership 1[y ≥ k] and indicator 1[y > k] by their expectations
under the uniform distribution u_S over the accepted set. Could this be split off as an ML paper?
That would need (a) a contribution reviewers accept as new and (b) evidence that it beats the
closest published alternative, **soft-CORAL / soft-OR** (Matton et al., 2026), and the standard
label-distribution baselines.

## 2. What soft-CORN is, formally (all checked numerically in `tests/`)

Let p be the soft label, q_k = P_p(y > k), and z_k the CORN conditional logits.

1. **It is a categorical likelihood.** Per example,
   L = −Σ_k [q_k log σ(z_k) + p_k log(1 − σ(z_k))] = −Σ_k p_k log p̂_k,
   where p̂ is the PMF that CORN's chain rule decodes. soft-CORN is therefore cross-entropy between
   the soft label and the continuation-ratio (sequential-logit) PMF. Equivalently, it is the
   grouped-data / fractional-count likelihood of the continuation-ratio model with level-specific
   effects (Fienberg, 1980; Tutz, 1991). `test_soft_corn_is_cross_entropy_of_decoded_pmf`.
2. **It reduces to CORN on one-hot labels.** This is exact with CORN's normalisation by the
   conditional-set size, and holds up to a constant per batch with per-example normalisation.
   `test_soft_corn_equals_corn_on_one_hot`.
3. **Gradient and curvature match the thesis appendix:** ∂L/∂z_k = q_{k−1} σ(z_k) − q_k and
   ∂²L/∂z_k² = q_{k−1} σ(1−σ). `test_soft_corn_gradient_and_hessian`.

Consequences for positioning:

* soft-CORN, soft-CORAL, soft-OR and soft cross-entropy are all **strictly proper for the same
  target**. In a rich enough model class, their population minimiser is t(x) = E[u_S | x]. They
  differ only in the output parameterisation:
  * continuation ratio with level-specific weights (CORN);
  * proportional odds with one shared weight vector (CORAL);
  * independent cumulative logits without rank consistency (OR);
  * multinomial logit (softmax).

  A linear soft-CORN head and a linear softmax head have the same effective number of parameters,
  (K−1)·(d+1), because softmax is invariant to a common shift of its logits.
* The three OLDL losses are **not** proper for t:
  * CAD (L1 between CDFs) is minimised by the coordinatewise *median* CDF of the labels.
  * QFD as published puts an absolute value inside the quadratic form.
  * CJS is a Jensen–Shannon term on cumulatives.

  Over 8 random three-label problems, the minimiser of the expected loss misses t by up to 0.14
  (QFD) and 0.17 (CJS) in total variation. For CAD, the median CDF beats the mean CDF by up to 0.21
  in expected loss. soft-CORN, soft-OR/CORAL and soft CE recover t exactly
  (`test_properness_for_the_mean_soft_label`).
* Neither the partial-label likelihood −log P̂(S) (whose target is the coarsened distribution of
  sets) nor a hard-label loss on the set's median targets t either.

So the *method* is a known model fitted with a known likelihood. An ML paper would have to rest on
an empirical finding, or on an application.

## 3. Benchmark design (details in README.md)

* **Real data.** Five public multi-annotator ordinal datasets with K = 4–7:
  * text: MHS sentiment, Wikipedia Toxicity, Humicroedit, CommitmentBank;
  * images: SCUT-FBP5500.

  Each set label is the set of levels chosen by 3 sampled annotators. The remaining annotators
  are held out and used only to evaluate (*held-out-rater NLL*). Features come from frozen
  encoders (all-mpnet-base-v2, OpenCLIP ViT-B/16).
* **Synthetic data.** Five rater-distribution families with logits linear in x. Three are the home
  models of CORAL (proportional odds), CORN (continuation ratio) and softmax (multinomial logit);
  two match none of them: location-scale, and a polarised mixture that produces gapped sets. Labels
  again come from 3 raters. The exact population target gives the excess risk KL(t ‖ p̂) with no
  annotator noise.
* **Twelve methods, one protocol.**
  * Heads: linear at every training size (128, 512, 2,048, 8,192, full); a one-hidden-layer MLP
    as a robustness check.
  * Optimiser: Adam with an L2 penalty, 5 × 5 grid of learning rate and penalty, 2,000 steps.
  * Selection: the checkpoint and configuration for each test metric are chosen by the matching
    validation metric (set-MAE for set-MAE, NLL for NLL/held-out NLL, RPS for RPS).
  * Shared across methods: data order, initialisation, splits, 5 seeds.
* **Statistics.**
  * Per problem: Nadeau–Bengio corrected resampled t-test → win / tie / loss.
  * Across problems: Friedman + Nemenyi test, and the Bayesian signed-rank test with a ROPE of
    0.005 nats for NLL, 0.005 for set-MAE and 0.001 for RPS.

## 4. Results: synthetic (linear heads, 5 seeds; `figures/synthetic_klt_linear.png`)

Excess risk KL(t ‖ p̂) at n = 16,384 (lower is better; the best in each row is bold):

| family (home model) | soft-CORN | soft-CORAL | soft-OR | soft CE | OLDL-QFD |
|---|---|---|---|---|---|
| proportional odds (CORAL) | 0.0063 | **0.0006** | 0.0014 | 0.0056 | 0.0064 |
| continuation ratio (CORN) | **0.0012** | 0.2104 | 0.0628 | 0.0135 | 0.0166 |
| multinomial logit (softmax) | 0.0422 | 0.4617 | 0.1221 | **0.0015** | 0.0039 |
| location-scale (none) | 0.1107 | 0.1538 | **0.0977** | 0.1277 | 0.1316 |
| polarised mixture (none) | 0.0153 | 0.0479 | 0.0235 | **0.0122** | 0.0193 |

* **Each parameterisation wins on its home model**, with gaps of 10× or more that persist as n grows.
  So these are bias effects, not variance.
* **soft-CORAL fails off its home model.** Its shared score cannot follow item-level changes in
  spread. The correlation between predicted and true per-item spread is 0.33–0.53 on the
  non-proportional-odds families, against 0.77–1.00 for soft-CORN.
* **soft-CORN is second on both families that match no model.**
* **Hard-label losses, PLL and OLDL plateau well above the proper soft losses**, as their
  improperness predicts (section 2).
* **MLP heads (5 seeds, n ∈ {1,024, 16,384}).** The flexible parameterisations converge:
  * soft-CORN and soft CE are practically equivalent (P(ROPE) = 0.83).
  * soft-OR is slightly behind (P(soft-CORN better) = 0.73).
  * soft-CORAL stays far behind on every family except proportional odds (soft-CORN wins 8 of
    10 problems): at n = 16,384 its excess risk is 0.19 on continuation-ratio data and 0.39 on
    multinomial-logit data, against ≤ 0.006 for soft-CORN.
  * On proportional-odds data all four land within 0.0019–0.0026.

  With enough capacity, the choice among soft-CORN, soft CE and soft-OR stops mattering, but
  CORAL's single-score restriction does not go away.

## 5. Results: real data (linear heads; 22 problems, 5 seeds; `figures/real_*_vs_softcoral_linear.png`)

**soft-CORN minus baseline** (< 0 favours soft-CORN). Cells give the mean difference followed by
win / tie / loss counts across the 22 problems:

| metric | vs soft-CORAL | vs soft CE | vs soft-OR | vs best OLDL (QFD) | vs PLL-CORN | vs CORN (hard) |
|---|---|---|---|---|---|---|
| test NLL | 0.000 (5/17/0) | −0.001 (3/15/4) | −0.014 (8/14/0) | −0.011 (6/14/2) | −0.061 (15/7/0) | −0.155 (16/6/0) |
| held-out-rater NLL | −0.002 (2/19/1) | −0.000 (4/14/4) | −0.013 (5/17/0) | −0.008 (5/14/3) | −0.049 (12/10/0) | −0.161 (15/7/0) |
| set-MAE | −0.014 (8/14/0) | −0.004 (2/19/1) | −0.004 (3/19/0) | −0.009 (1/21/0) | 0.000 (0/21/1) | −0.038 (8/14/0) |

Bayesian signed-rank probabilities, written as P(soft-CORN better / practically equivalent / worse):

* **NLL:** vs soft-CORAL 0.16 / 0.65 / 0.19; vs soft CE 0.08 / 0.85 / 0.07.
* **set-MAE:** vs soft-CORAL 0.998 / 0.000 / 0.002; vs soft CE 0.64 / 0.34 / 0.02.

Mean ranks over the 12 methods (Nemenyi critical difference 3.55):

* **NLL:** soft-CORN 2.27, soft-CORAL 2.64, soft CE 2.82, soft-OR 4.91 … CORN (hard) 11.50.
* **set-MAE:** PLL-CORN 3.70, soft-CORN 4.45, soft-OR 4.57, soft CE 5.14 … soft-CORAL 8.73 (last).

The top three on NLL are not separated statistically.

**Sample-size regimes** (soft-CORN minus baseline; win / tie / loss; Bayesian P in brackets):

| | n ≤ 512 | n ≥ 2,048 |
|---|---|---|
| NLL vs soft-CORAL | +0.008, 0/10/0 (P(worse) = 0.91) | −0.007, 5/6/0 (P(better) = 0.59) |
| NLL vs soft CE | −0.007, 1/9/0 (P(better) = 0.63) | +0.005, 2/5/4 (P(worse) = 0.58) |
| set-MAE vs soft-CORAL | −0.011, 0/10/0 (P(better) = 0.96) | −0.023, 8/3/0 (P(better) = 1.00) |

* **vs soft-CORAL.** Proportional odds is a useful restriction with little data. It becomes a bias
  once data is plentiful.
* **vs soft CE.** The pattern partly reverses: softmax overtakes soft-CORN on the two largest text
  datasets, MHS and Wiki Toxicity (`figures/real_nll_vs_softcoral_linear.png`). So soft-CORN does
  not dominate soft CE in any regime.

**By set type** (full pool, mean over datasets):

* **Gapped sets:** soft-CORN beats soft-CORAL by 0.062 nats and 0.029 set-MAE, and soft CE by
  0.030 and 0.012. This is the regime the thesis argues for.
* **Singleton sets:** soft CE is slightly better (0.015 nats, 0.031 set-MAE).

**Other observations:**

* **Selecting on set-MAE, as the thesis does, damages calibration, and soft-CORN is hit more than
  its closest rivals.** Compared with NLL-based selection, the set-MAE-selected checkpoint raises
  test NLL (full pool, worst dataset; `results/real_full_linear_thesis_protocol.csv`) by:
  * 0.03 for soft-CORAL and 0.06 for soft CE;
  * 0.38 for soft-CORN;
  * 1.4 for soft-OR;
  * 0.7 for PLL;
  * 9–13 for the OLDL losses.

  Probabilistic claims in the thesis should therefore use NLL-based selection.
* **soft-OR is not rank-consistent.** Its implied P(y > k) is non-monotone for 2% of test items
  (NLL selection) up to 40% (set-MAE selection). CORAL and CORN never are.
* **All methods are close to the feature-free baseline on CommitmentBank and Humicroedit.** On
  CommitmentBank the best NLL is 1.840 against a prior of 1.923, so differences there are small
  in absolute terms.

**MLP heads** (5 datasets × n ∈ {512, full} = 10 problems, 5 seeds; 128 hidden units, dropout 0.2):

| metric | vs soft-CORAL | vs soft CE | vs soft-OR |
|---|---|---|---|
| test NLL | −0.006 (2/6/2) | 0.000 (1/9/0) | 0.000 (2/7/1) |
| held-out-rater NLL | −0.009 (2/8/0) | +0.003 (1/9/0) | +0.001 (1/9/0) |
| set-MAE | −0.007 (0/10/0) | −0.010 (0/10/0) | −0.002 (0/10/0) |

* **Bayesian NLL probabilities** (P(soft-CORN better / practically equivalent / worse)):
  vs soft-CORAL 0.84 / 0.00 / 0.16; vs soft CE 0.03 / 0.96 / 0.02.
* **Mean NLL ranks (critical difference 5.27):** soft CE 2.6, soft-CORN 2.9, soft-OR 3.1,
  soft-CORAL 3.4. The four proper soft losses are statistically tied. soft-CORN has the best mean
  rank on RPS (2.5).
* **On the full pool,** soft-CORN beats soft-CORAL by 0.019 nats (2/2/0; P(better) = 0.95).
* **The gapped-set advantage seen with linear heads does not carry over to MLP heads.** On gapped
  sets soft-CORN is ahead of soft-CORAL by only 0.008 nats and behind it by 0.026 in set-MAE.
* Earlier, with only one MLP seed, soft-OR appeared to lead; that did not survive five seeds.

**Annotators per set** (m ∈ {2, 3, 5}; linear heads, n ∈ {512, 2,048, full}, 5 seeds;
`figures/m_ablation_linear.png`). The comparison uses the four datasets with enough raters at every
m (CommitmentBank, Humicroedit, SCUT, Wiki Toxicity; MHS has only 281 items with five ratings),
giving 11 problems per m. Larger m means larger sets: singletons fall from 32–48% of labels at
m = 2 to 6–10% at m = 5.

soft-CORN minus baseline (win / tie / loss across the 11 problems):

| comparison | m = 2 | m = 3 | m = 5 |
|---|---|---|---|
| NLL vs soft-CORAL | −0.006 (2/9/0) | −0.002 (2/9/0) | −0.007 (4/7/0) |
| NLL vs soft CE | −0.003 (2/8/1) | −0.004 (3/7/1) | −0.002 (1/9/1) |
| set-MAE vs soft-CORAL | −0.016 (3/8/0) | −0.015 (3/8/0) | −0.019 (3/8/0) |
| gapped-set NLL vs soft-CORAL | −0.031 (1/10/0) | −0.042 (2/9/0) | −0.051 (2/9/0) |
| gapped-set set-MAE vs soft-CORAL | −0.033 (1/10/0) | −0.051 (3/8/0) | −0.070 (3/8/0) |
| gapped-set NLL vs soft CE | +0.004 (1/10/0) | −0.009 (1/10/0) | −0.020 (2/9/0) |
| NLL vs PLL-CORN | −0.034 (7/4/0) | −0.070 (9/2/0) | −0.124 (9/2/0) |

* **The m = 3 conclusions hold at m = 2 and m = 5.** Overall NLL differences to soft-CORAL and
  soft CE stay within 0.007 nats. The set-MAE advantage over soft-CORAL is present at every m
  (Bayesian P(better) ≥ 0.98).
* **The gapped-set advantage grows with m.** Against soft-CORAL it rises from 0.031 to 0.051 nats
  and from 0.033 to 0.070 set-MAE. Against soft CE it appears only at m = 5 (P(better) = 0.88).
  This is the strongest evidence for the thesis's motivation, but it applies to linear heads.
* **The partial-label likelihood falls further behind as sets grow**, as expected for a loss that
  only rewards mass inside the set.

**Fine-tuned backbones** (`figures/finetune_vs_frozen_nll.png`). Each loss fine-tunes its own copy of
the encoder used for the frozen features, end to end: all-mpnet-base-v2 for text, the OpenCLIP
ViT-B/16 image tower for SCUT.
* Data: 2,048 training items (CommitmentBank: its 840), so each result sits next to a frozen-probe
  cell on the same items.
* Optimiser: AdamW, bf16, 5 epochs (8 for CommitmentBank).
* Learning rates: backbone {1e-5, 3e-5}, chosen per method on seed 0's validation NLL; head 1e-3.
* Head initialisation: biases start at the training-label marginal.
* Epoch: chosen on validation. Seeds: 3. The frozen probe uses 5.

Test NLL, fine-tuned (frozen linear probe in brackets):

| dataset | soft-CORN | soft-CORAL | soft CE |
|---|---|---|---|
| CommitmentBank | 1.711 (1.840) | 1.842 (1.842) | **1.683** (1.845) |
| Humicroedit | **1.251** (1.253) | 1.253 (1.255) | 1.256 (1.259) |
| MHS | 1.068 (1.133) | 1.148 (1.135) | **1.063** (1.133) |
| SCUT | **1.090** (1.090) | 1.132 (1.085) | 1.101 (1.092) |
| Wiki Toxicity | 0.983 (1.038) | 1.024 (1.051) | **0.980** (1.039) |

* **Fine-tuning helps the flexible heads, not CORAL.** On CommitmentBank, MHS and Wiki Toxicity,
  soft-CORN and soft CE improve by 0.05–0.16 nats over the frozen probe. soft-CORAL improves only on
  Wiki Toxicity (0.03); it is unchanged on CommitmentBank and gets worse on MHS and SCUT. Neither
  Humicroedit (little signal) nor SCUT (2,048 face images) benefits much for any method.
* **Against soft-CORAL, soft-CORN now wins clearly.**
  * NLL: −0.059 on average, ahead on all five datasets (−0.131, −0.002, −0.080, −0.042, −0.041;
    P(better) = 0.985).
  * Held-out-rater NLL: −0.068. Set-MAE: −0.033.
  * With only 3 seeds, the corrected per-dataset t-test confirms 1–2 of these differences
    individually.
* **Against soft CE, still a tie.** NLL differs by +0.004 (0/5/0; P(practically equivalent) = 0.60),
  and set-MAE by +0.013. Both are driven by CommitmentBank, where soft CE is ahead.
* **This is the synthetic result again, with learned features.** Once the representation can adapt,
  the proportional-odds restriction of CORAL's single shared score becomes the binding constraint.
  Continuation-ratio and softmax heads, which have no such restriction, do equally well.

## 6. Robustness checks run

* **Grid edges.** 29% of linear selections took the strongest L2 penalty, mostly on CommitmentBank
  and Humicroedit, and more often for soft-CORAL (32%) than soft-CORN (16%). Re-running those two
  datasets with the penalty grid extended to 1,000 changed NLL by ≤ 0.004 and never selected the
  new maximum. The soft-CORN – soft-CORAL difference moved from +0.002 to +0.003 nats, so the
  ranking is unchanged and the grid did not disadvantage soft-CORAL.
* **MLP learning-rate edge.** With MLP heads, the lowest learning rate (3e-4) was selected 36% of
  the time on real data, often at the first checkpoint. This happened for soft-CORN and soft CE but
  never for soft-CORAL, so it could have biased the comparison. I re-ran those three methods with
  learning rates {3e-5, 1e-4, 3e-4, 1e-3} and checkpoints every 50 steps.
  * soft-CORN and soft CE changed by ≤ 0.001 nats.
  * The new lowest rate was chosen 5% of the time and never at the first checkpoint.
  * soft-CORAL selected the top rate (1e-3) in all 50 cells, i.e. it needs the higher rates
    that only the main grid covers.

  Every method's optimum therefore lies inside the main grid, and the MLP results stand.
* **Fine-tuning head initialisation.** In the first fine-tuning attempt, the output head trained at the
  backbone's learning rate from its default initialisation. soft-CORAL then stayed worse than the
  feature-free prior (validation NLL about 2.55 on CommitmentBank), because its K−1 threshold biases
  barely move from their generic spacing in a few hundred steps. Giving the head its own learning rate
  was not enough on its own. Initialising every head's biases at the training-label marginal fixed it
  for all methods. Those results were discarded and everything was re-run with this protocol
  (`test_prior_bias_init_reproduces_marginal`). CORAL-style heads are the most sensitive to this
  choice, which practitioners fine-tuning them should know.
* **Optimiser.** Pilots showed that AdamW's decoupled weight decay does not regularise
  768–1,536-d linear heads: every method overfitted CommitmentBank to below the feature-free
  baseline. The benchmark therefore uses a coupled L2 penalty, the standard linear-probe protocol.
* **Implementation checks.**
  * The vectorised ensemble trains each configuration exactly as an independent
    torch.optim.Adam run.
  * The CUDA-graph training path matches eager training.
  * The OLDL ports match the reference implementation.
  * soft-CORN matches `coral_pytorch.corn_loss` on one-hot labels.

## 7. Limitations

* **Fine-tuning was run at one training size** (2,048 items) with 3 seeds and two backbone learning
  rates. Per-dataset tests are therefore weak, and conclusions rest on the consistent direction
  across datasets.
* **Simulated set labels.** Sets come from sampled annotators (3 in the main runs, 2 and 5 in the
  ablation) rather than native set-valued responses like those in the thesis's in-vehicle study.
  The ablation varies set size but keeps the same mechanism.
* **Held-out raters are thin on two datasets.** On MHS, most items have few annotators, so
  held-out-rater metrics rest on a subset of items. Humicroedit has only 2 held-out ratings per item.

## 8. What a submission would need

1. **A framing that does not claim the loss is new.** Cite the continuation-ratio grouped
   likelihood, and position soft-CORN as the CORN counterpart of soft-CORAL that drops the
   proportional-odds restriction.
2. **This benchmark, leading with three results:** the synthetic home-model result; the fine-tuning
   result, where soft-CORN beats soft-CORAL on every dataset while soft-CORAL barely benefits from
   fine-tuning; and the gapped-set advantage under linear heads, which grows with the number of
   annotators per set. A larger fine-tuning study (more seeds and training sizes) would make the
   fine-tuning result the headline.
3. **The properness analysis of the OLDL losses.** This is the most novel-looking finding here.
4. **Ideally, one dataset with native set-valued labels,** such as the thesis's own user study.
