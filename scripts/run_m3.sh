#!/usr/bin/env bash
# Waits for the Stage 0 cache, then trains both predictor seeds and evaluates.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=./.conda/bin/python
until grep -q "stage0 cache complete" data/logs/stage0_cache.out; do sleep 120; done
for seed in 0 1; do
    [ -f checkpoints/stage0/predictor_seed${seed}.pt ] || $PY -u scripts/stage0_train.py train --seed $seed
done
$PY -u scripts/stage0_train.py eval
