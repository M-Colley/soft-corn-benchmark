#!/usr/bin/env bash
# Frozen-feature part of the benchmark, one stage after another on a single GPU (concurrent runs
# time-slice the GPU and are slower in total). Every runner skips finished cells, so this script can
# be interrupted and started again. Usage:  SCB_DATA=/path bash scripts/run_frozen.sh [python]
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${1:-python}"
OUT="${SCB_DATA:-./data}/results"
SEEDS="${SEEDS:-0 1 2 3 4}"          # e.g. SEEDS="5 6 7 8 9" adds five more seeds to the main benchmark
export PYTHONIOENCODING=utf-8

echo "== synthetic, linear (5 families x 5 seeds x 4 sizes)"
$PY scripts/04_synthetic.py --bodies linear --seeds $SEEDS
echo "== synthetic, MLP (n = 1,024 and 16,384)"
$PY scripts/04_synthetic.py --bodies mlp --n_train 1024 16384 --seeds $SEEDS
echo "== real data, linear heads (5 datasets x 5 sizes x 5 seeds)"
$PY scripts/03_run.py --bodies linear --seeds $SEEDS
echo "== real data, MLP heads (n = 512 and the full pool)"
$PY scripts/03_run.py --bodies mlp --n_train 512 full --seeds $SEEDS
echo "== annotators per set: m = 2 and m = 5 (MHS has too few items with five ratings)"
$PY scripts/03_run.py --m 2 --bodies linear --n_train 512 2048 full --out "$OUT/m_ablation"
$PY scripts/03_run.py --m 5 --datasets cb scut humicroedit wikitox --bodies linear --n_train 512 2048 full \
    --out "$OUT/m_ablation"
echo "== grid-edge checks"
$PY scripts/03_run.py --datasets cb humicroedit --bodies linear --wds 0.001 0.01 0.1 1 10 100 1000 \
    --out "$OUT/wd_check"
$PY scripts/03_run.py --bodies mlp --n_train 512 full --methods soft_corn ce_soft soft_coral \
    --lrs 3e-5 1e-4 3e-4 1e-3 --eval_every 50 --out "$OUT/mlp_lr_check"
echo "== frozen-feature stages finished"
