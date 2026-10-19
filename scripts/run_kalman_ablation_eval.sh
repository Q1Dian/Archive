#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=./.conda/bin/python
CKPT="${1:-checkpoints/stage_a_kalman/latest.pt}"
$PY -u scripts/eval_e1.py --ckpt "$CKPT" --split val --out checkpoints/gate1/e1_kalman_val.json
$PY -u scripts/eval_e4.py --ckpt "$CKPT" --split train --out checkpoints/gate2/e4_kalman_train.json --tb-run e4_kalman
$PY -u scripts/eval_e2.py --ckpt "$CKPT" --out checkpoints/gate1/e2_kalman.json
