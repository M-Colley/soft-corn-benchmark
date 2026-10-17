#!/usr/bin/env bash
# Experiments beyond the core benchmark (FUTURE_WORK.md), one GPU job at a time. Every runner skips finished cells.
#
#   SCB_DATA=/path bash scripts/run_extras.sh [python] [stage ...]
#
# Stages (default: replay targets_syn targets_real baselines_syn baselines_real synth_variants shift personal):
#   replay          predictions of the selected checkpoints of the real linear-head cells (08_replay.py), for the
#                   per-item tests, conformal sets, reliability diagrams and temperature scaling (09_setpred.py)
#   targets_syn     training-target study on synthetic data, m = 2, 3, 5: uniform-over-set (soft-CORN / soft CE /
#                   soft-CORAL), empirical distribution of the ratings, 50/50 mixture, exact set likelihood
#   targets_real    the same on the real data (m = 3; m = 2 and 5 at two sizes)
#   baselines_syn   SORD, Gaussian label distribution, CDW-CE, binomial / Poisson heads, cumulative link models
#   baselines_real  the same on the real data
#   synth_variants  richer synthetic study: K, input dimension, signal strength, set mechanisms, nonlinear generators,
#                   label noise and rater heterogeneity (3 seeds each)
#   shift           distribution shift: leave one platform / namespace / year / demographic subset out (12_shift.py)
#   personal        personalisation on SCUT: per-rater offsets, own rating vs population consensus (13_personal.py)
#   cost            compute cost per method (11_cost.py); run it on an otherwise idle GPU
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${1:-python}"
shift || true
STAGES="${*:-replay targets_syn targets_real baselines_syn baselines_real synth_variants shift personal}"
OUT="${SCB_DATA:-./data}/results"
SEEDS="${SEEDS:-0 1 2 3 4}"
TARGETS="soft_corn soft_corn_emp soft_corn_mix50 exact_corn ce_soft ce_emp ce_mix50 exact_ce soft_coral soft_coral_emp soft_coral_mix50 exact_coral"
BASELINES="sord sord_sq cdwce binom poisson clm_logit clm_probit clm_cloglog clm_ao"
export PYTHONIOENCODING=utf-8

variant() {   # variant <name> <04_synthetic args...>: 3 seeds, n = 1,024 and 16,384, linear heads (MLP too if nonlinear)
  local name=$1; shift
  local bodies="linear"
  case "$*" in *--nonlinear*) bodies="linear mlp";; esac
  $PY scripts/04_synthetic.py --seeds 0 1 2 --n_train 1024 16384 --bodies $bodies --out "$OUT/synth_$name" "$@"
}

for stage in $STAGES; do
  echo "== $stage"
  case $stage in
    replay)         $PY scripts/08_replay.py --bodies linear ;;
    targets_syn)    for m in 3 2 5; do
                      $PY scripts/04_synthetic.py --m $m --seeds $SEEDS --n_train 1024 16384 --bodies linear \
                          --methods $TARGETS --out "$OUT/targets_synthetic"
                    done ;;
    targets_real)   $PY scripts/03_run.py --seeds $SEEDS --n_train 512 2048 full --bodies linear \
                        --methods $TARGETS --out "$OUT/targets_real"
                    $PY scripts/03_run.py --m 2 --seeds $SEEDS --n_train 512 full --bodies linear \
                        --methods $TARGETS --out "$OUT/targets_real"
                    $PY scripts/03_run.py --m 5 --datasets cb scut humicroedit wikitox --seeds $SEEDS \
                        --n_train 512 full --bodies linear --methods $TARGETS --out "$OUT/targets_real" ;;
    baselines_syn)  $PY scripts/04_synthetic.py --seeds $SEEDS --bodies linear --methods $BASELINES \
                        --out "$OUT/baselines_synthetic"
                    $PY scripts/04_synthetic.py --seeds $SEEDS --n_train 1024 16384 --bodies mlp --methods $BASELINES \
                        --out "$OUT/baselines_synthetic" ;;
    baselines_real) $PY scripts/03_run.py --seeds $SEEDS --bodies linear --methods $BASELINES --out "$OUT/baselines_real"
                    $PY scripts/03_run.py --seeds $SEEDS --n_train 512 full --bodies mlp --methods $BASELINES \
                        --out "$OUT/baselines_real" ;;
    shift)          $PY scripts/12_shift.py --seeds $SEEDS ;;
    personal)       $PY scripts/13_personal.py --seeds $SEEDS ;;
    cost)           $PY scripts/11_cost.py ;;
    synth_variants) variant K3 --K 3
                    variant K8 --K 8
                    variant K10 --K 10
                    variant d3 --d 3
                    variant d50 --d 50
                    variant d200 --d 200
                    variant sig05 --signal 0.5
                    variant sig2 --signal 2
                    variant interval --mechanism interval
                    variant threshold --mechanism threshold
                    variant gapped --mechanism gapped
                    variant nonlinear --nonlinear
                    variant noise --eps 0.2
                    variant hetero --shift 0.2 0.6 0.2
                    variant m2 --m 2
                    variant m5 --m 5 ;;
    *) echo "unknown stage $stage"; exit 1 ;;
  esac
done
echo "== extras finished"
