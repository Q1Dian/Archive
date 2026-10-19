#!/usr/bin/env bash
# Full D2 val E1/E2 (all videos with length>=512; optional multi-window via WINDOW_STRIDE).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=./.conda/bin/python
CKPT="${1:-checkpoints/stage_a_kalman/step8000.pt}"
STRIDE="${WINDOW_STRIDE:-128}"
echo "ckpt=$CKPT window_stride=$STRIDE"
$PY -u scripts/eval_e1.py --ckpt "$CKPT" --split val --window-stride "$STRIDE" \
  --out "checkpoints/gate1/e1_$(basename "$CKPT" .pt)_val_full.json"
$PY -u scripts/eval_e2.py --ckpt "$CKPT" --window-stride "$STRIDE" \
  --out "checkpoints/gate1/e2_$(basename "$CKPT" .pt)_val_full.json"
