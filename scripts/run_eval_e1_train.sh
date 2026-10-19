#!/usr/bin/env bash
# Full-train E1: low host RAM (stream videos, batch 2). Cap 18G on a 32G machine.
set -euo pipefail
cd "$(dirname "$0")/.."
exec scripts/run_capped.sh 18G data/logs/eval_e1_step13000_train.out \
  ./.conda/bin/python -u scripts/eval_e1.py \
  --ckpt checkpoints/stage_a/step13000.pt \
  --split train \
  --batch 2 \
  --log data/logs/stage_a.jsonl \
  --out checkpoints/gate1/e1_step13000_train.json
