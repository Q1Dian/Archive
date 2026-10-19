#!/usr/bin/env bash
# Gate 2 after Stage B: E4 (EK100 train), E1 val (no E1 regression), E3b/E3c when cached.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=./.conda/bin/python
ANCHOR=checkpoints/stage_a_decoder_tune/step4000.pt
CKPT="${1:-checkpoints/stage_b_joint_readout/step1000.pt}"
if [[ ! -f "$CKPT" ]]; then
  FALLBACK="$(dirname "$CKPT")/latest.pt"
  if [[ -f "$FALLBACK" ]]; then
    echo "checkpoint missing: $CKPT — using $FALLBACK" >&2
    CKPT="$FALLBACK"
  else
    echo "error: no checkpoint at $CKPT (and no $FALLBACK)" >&2
    exit 1
  fi
fi
SIGMA=0.05
LOG=data/logs/stage_a_decoder_tune.jsonl

$PY -u scripts/eval_e4.py --ckpt "$CKPT" --split train --sigma-obs "$SIGMA" --log-lam -4.6 \
  --out checkpoints/gate2/e4_stage_b_train.json
$PY -u scripts/eval_e1.py --ckpt "$CKPT" --split val --window-stride 128 --sigma-obs "$SIGMA" \
  --out checkpoints/gate1/e1_stage_b_joint_val.json
$PY -u scripts/eval_e1.py --ckpt "$ANCHOR" --split val --window-stride 128 --sigma-obs "$SIGMA" \
  --out checkpoints/gate1/e1_anchor_val.json
$PY -u scripts/eval_e3.py --ckpt "$CKPT" --dataset goalstep --sigma-obs "$SIGMA" || true
$PY -u scripts/eval_e3.py --ckpt "$CKPT" --dataset assembly101 --sigma-obs "$SIGMA" || true
