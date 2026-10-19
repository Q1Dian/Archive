#!/usr/bin/env bash
# Extract -> precompute -> Stage B L on Assembly101 (reuses EK100 PCA/prior).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=./.conda/bin/python
MF=data/manifests/assembly101.txt
LOG=data/logs/assembly101_pipeline.out
exec >>"$LOG" 2>&1
echo "=== assembly101 pipeline $(date) ==="

pkill -f "extract.py assembly101" 2>/dev/null || true
sleep 1
$PY -u scripts/extract.py assembly101 --manifest "$MF" &
EXT=$!
echo "extract pid $EXT"

while kill -0 "$EXT" 2>/dev/null; do
  sleep 60
done
echo "backfill tcells..."
$PY -u scripts/extract.py assembly101 --manifest "$MF" --add-token-cells
$PY -u scripts/precompute.py assembly101 --manifest "$MF"

READY=$($PY -c "
from pathlib import Path
from wcem.data.assembly101 import manifest_cache_ids
from wcem.paths import CACHE
feat = CACHE / 'features' / 'assembly101'
print(sum(1 for v in manifest_cache_ids() if (feat / f'{v}_dupsim.npy').exists()))
")
echo "precomputed videos: $READY"
if [[ "$READY" -lt 10 ]]; then
  echo "too few precomputed videos for training ($READY)"; exit 1
fi

$PY -u scripts/train.py --config configs/stage_b_assembly101.yaml
echo "=== pipeline complete $(date) ==="
