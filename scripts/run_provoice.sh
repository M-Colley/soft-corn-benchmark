#!/usr/bin/env bash
# The ProVoice study (natively set-valued labels), one GPU job after the other (all stages resume).
#   core      the twelve core losses, linear and MLP heads, 512 answers and the full training pool
#   task      the same with the task code as the only feature
#   centred   the same with driver-state features centred on each driver's own mean (label-free)
#   known     the personalised setting: a driver code as a feature, split by 20 s window instead of by driver
#   baselines SORD, CDW-CE, binomial / Poisson heads and cumulative link models
#   analyze   results/provoice.md, results/provoice_*.csv, figures/provoice_pairwise*.png
#   kcurve    the thesis's personalisation protocol: leave one driver out, K calibration answers of that driver
#             (scripts/15_kcurve.py; not in the default stages, ~3 h) and its analysis (scripts/16_kcurve.py)
# Usage: SCB_DATA=... bash scripts/run_provoice.sh <python> [stages]
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${1:-python}"
shift || true
STAGES="${*:-core task centred known baselines analyze}"
OUT="${SCB_DATA:-./data}/results"
SEEDS="${SEEDS:-$(seq 0 19)}"
COMMON="--m 0 --seeds $SEEDS --n_train 512 full --bodies linear mlp --test_frac 0.25 --val_frac 0.17"

# A job that shares the GPU with another heavy process has aborted with a device-side assert (it never produced a wrong
# cell: stored cells are bit-identical to re-runs on an idle card). Cells are resumable, so retry the stage.
retry() {
  for attempt in 1 2 3 4 5 6 7 8 9 10; do
    "$@" && return 0
    echo "attempt $attempt failed, resuming"
  done
  return 1
}
for stage in $STAGES; do
  echo "== $stage"
  case "$stage" in
    core)      retry "$PY" scripts/03_run.py --datasets provoice $COMMON --out "$OUT/provoice" ;;
    task)      retry "$PY" scripts/03_run.py --datasets provoice_task $COMMON --out "$OUT/provoice_task" ;;
    centred)   retry "$PY" scripts/03_run.py --datasets provoice_centred $COMMON --out "$OUT/provoice_centred" ;;
    known)     retry "$PY" scripts/03_run.py --datasets provoice_known $COMMON --out "$OUT/provoice_known" ;;
    baselines) retry "$PY" scripts/03_run.py --datasets provoice $COMMON --out "$OUT/provoice_baselines" \
                 --methods sord sord_sq cdwce binom poisson clm_logit clm_probit clm_cloglog clm_ao ;;
    analyze)   "$PY" scripts/14_provoice.py ;;
    kcurve)    retry "$PY" scripts/15_kcurve.py --features task all --seeds 0 1 2 ;;
    kanalyze)  "$PY" scripts/16_kcurve.py ;;
    *) echo "unknown stage $stage"; exit 1 ;;
  esac
done
echo "== ProVoice stages finished"
