#!/usr/bin/env bash
# Fine-tuning part of the benchmark, one GPU job at a time (a 6 GB card fits one CommitmentBank run; a
# 16 GB card fits one run comfortably and is slower when two share it). One dataset per call keeps each
# call short, and every call skips finished runs, so the script can be interrupted and started again.
#
#   SCB_DATA=/path bash scripts/run_finetune.sh [python] [stage ...]
#
# Stages (default: main seeds lpft refs):
#   main     3 methods (soft-CORN, soft-CORAL, soft CE), n = 2,048, seeds 0-2, backbone lr {1e-5, 3e-5} on seed 0
#   seeds    seeds 3 and 4 (reuse seed 0's learning rate)
#   lpft     same, with the head first fitted on frozen features (LP-FT), seeds 0-4   -> results/finetune_lpft
#   refs     soft-OR, OLDL-QFD, PLL-CORN and hard CORN as references, seeds 0-4
#   sizes    n = 512 and n = 8,192, seeds 0-4
#   full     the full training pool, seeds 0-2
#   deberta  text datasets with microsoft/deberta-v3-base, seeds 0-4                  -> results/finetune_deberta
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${1:-python}"
shift || true
STAGES="${*:-main seeds lpft refs}"
OUT="${SCB_DATA:-./data}/results"
ALL="cb scut humicroedit mhs wikitox"
TEXT="cb humicroedit mhs wikitox"
SEEDS5="0 1 2 3 4"
export PYTHONIOENCODING=utf-8

for stage in $STAGES; do
  echo "== $stage"
  case $stage in
    main)    for d in $ALL; do $PY scripts/06_finetune.py --datasets "$d"; done ;;
    seeds)   for d in $ALL; do $PY scripts/06_finetune.py --datasets "$d" --seeds 3 4; done ;;
    lpft)    for d in $ALL; do $PY scripts/06_finetune.py --datasets "$d" --seeds $SEEDS5 --init lpft \
                 --out "$OUT/finetune_lpft"; done ;;
    refs)    for d in $ALL; do $PY scripts/06_finetune.py --datasets "$d" --seeds $SEEDS5 \
                 --methods soft_or oldl_qfd pll_corn corn_hard; done ;;
    sizes)   for n in 512 8192; do for d in $ALL; do $PY scripts/06_finetune.py --datasets "$d" --seeds $SEEDS5 \
                 --max_train "$n"; done; done ;;
    full)    for d in $ALL; do $PY scripts/06_finetune.py --datasets "$d" --seeds 0 1 2 --max_train 100000; done ;;
    deberta) for d in $TEXT; do $PY scripts/06_finetune.py --datasets "$d" --seeds $SEEDS5 \
                 --text_model microsoft/deberta-v3-base --out "$OUT/finetune_deberta"; done ;;
    *) echo "unknown stage $stage"; exit 1 ;;
  esac
done
echo "== fine-tuning stages finished"
