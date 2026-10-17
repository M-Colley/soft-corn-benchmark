#!/usr/bin/env bash
# Fine-tuning part of the benchmark, one GPU job at a time (a 6 GB card fits one CommitmentBank run; a
# 16 GB card fits one run comfortably and is slower when two share it). One dataset per call keeps each
# call short, and every call skips finished runs, so the script can be interrupted and started again.
#
#   SCB_DATA=/path bash scripts/run_finetune.sh [python] [stage ...]
#
# Stages (default: main seeds lpft refs):
#   main     3 methods (soft-CORN, soft-CORAL, soft CE), n = 2,048, seeds 0-2, backbone lr {1e-5, 3e-5} on seed 0
#   seeds    seeds 3 and 4, or $EXTRA_SEEDS (reuse seed 0's learning rate)
#   lpft     same, with the head first fitted on frozen features (LP-FT), seeds 0-4   -> results/finetune_lpft
#   refs     soft-OR, OLDL-QFD, PLL-CORN and hard CORN as references, seeds 0-4
#   sizes    n = 512 and n = 8,192, seeds 0-4
#   lpft512  LP-FT variant at n = 512, seeds 0-4
#   full     the full training pool, seeds 0-2
#   lrcheck  backbone lr 1e-4 (above the main grid), 3 methods, n = 2,048, seeds 0-4  -> results/finetune_lr_check
#   deberta  text datasets with microsoft/deberta-v3-base, seeds 0-2, prior + LP-FT   -> results/finetune_deberta[_lpft]
#   roberta  text datasets with roberta-large (lr 5e-6 .. 2e-5), seeds 0-2            -> results/finetune_roberta_large[_lpft]
#   dinov2   SCUT with the DINOv2 ViT-B/14 (hf:facebook/dinov2-base), seeds 0-2       -> results/finetune_dinov2[_lpft]
#   siglip   SCUT with the OpenCLIP SigLIP ViT-B/16 (webli), seeds 0-2                -> results/finetune_siglip[_lpft]
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${1:-python}"
shift || true
STAGES="${*:-main seeds lpft refs}"
OUT="${SCB_DATA:-./data}/results"
ALL="cb scut humicroedit mhs wikitox"
TEXT="cb humicroedit mhs wikitox"
SEEDS5="${SEEDS5:-0 1 2 3 4}"          # seeds of the lpft / refs / sizes stages
EXTRA_SEEDS="${EXTRA_SEEDS:-3 4}"      # seeds of the `seeds` stage (main protocol, reuses seed 0 learning rate)
export PYTHONIOENCODING=utf-8

backbone() {   # backbone <tag> "<datasets>" <06_finetune.py args>: 3 methods, n = 2,048, seeds 0-2, prior and LP-FT head init
  local tag=$1 datasets=$2; shift 2
  for d in $datasets; do
    $PY scripts/06_finetune.py --datasets "$d" --out "$OUT/finetune_$tag" "$@"
    $PY scripts/06_finetune.py --datasets "$d" --init lpft --out "$OUT/finetune_${tag}_lpft" "$@"
  done
}

for stage in $STAGES; do
  echo "== $stage"
  case $stage in
    main)    for d in $ALL; do $PY scripts/06_finetune.py --datasets "$d"; done ;;
    seeds)   for d in $ALL; do $PY scripts/06_finetune.py --datasets "$d" --seeds $EXTRA_SEEDS; done ;;
    lpft)    for d in $ALL; do $PY scripts/06_finetune.py --datasets "$d" --seeds $SEEDS5 --init lpft \
                 --out "$OUT/finetune_lpft"; done ;;
    refs)    for d in $ALL; do $PY scripts/06_finetune.py --datasets "$d" --seeds $SEEDS5 \
                 --methods soft_or oldl_qfd pll_corn corn_hard; done ;;
    sizes)   for n in ${SIZES:-512 8192}; do for d in $ALL; do $PY scripts/06_finetune.py --datasets "$d" --seeds $SEEDS5 \
                 --max_train "$n"; done; done ;;
    lpft512) for d in $ALL; do $PY scripts/06_finetune.py --datasets "$d" --seeds $SEEDS5 --max_train 512 --init lpft \
                 --out "$OUT/finetune_lpft"; done ;;
    full)    for d in $ALL; do $PY scripts/06_finetune.py --datasets "$d" --seeds 0 1 2 --max_train 100000; done ;;
    lrcheck) for d in $ALL; do $PY scripts/06_finetune.py --datasets "$d" --seeds $SEEDS5 --lrs 1e-4 \
                 --out "$OUT/finetune_lr_check"; done ;;
    deberta) backbone deberta "$TEXT" --text_model microsoft/deberta-v3-base ;;
    roberta) backbone roberta_large "$TEXT" --text_model roberta-large --lrs 5e-6 1e-5 2e-5 ;;
    dinov2)  backbone dinov2 scut --image_model hf:facebook/dinov2-base ;;
    siglip)  backbone siglip scut --image_model ViT-B-16-SigLIP/webli ;;
    *) echo "unknown stage $stage"; exit 1 ;;
  esac
done
echo "== fine-tuning stages finished"
