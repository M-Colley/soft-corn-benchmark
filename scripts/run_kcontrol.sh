#!/usr/bin/env bash
# Control for the K-curve: train on one level of every set (the upper median, a random member) instead of the whole set,
# for the two parameterisations that use the set (soft-CORN, soft CE); evaluation keeps the sets.
# Usage: SCB_DATA=... bash scripts/run_kcontrol.sh <python>
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${1:-python}"
for rule in upper random; do
  echo "== collapse $rule"
  for attempt in 1 2 3 4 5 6 7 8 9 10; do
    "$PY" scripts/15_kcurve.py --features task --seeds 0 1 2 --collapse "$rule" --methods soft_corn ce_soft && break
    echo "attempt $attempt failed, resuming"
  done
done
echo "== controls finished"
