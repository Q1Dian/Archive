"""Block until every extracted video in a manifest has per-token cells.

    python scripts/wait_tcells.py data/manifests/ek100_d2_train.txt && python scripts/precompute.py ...
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wcem.paths import CACHE  # noqa: E402

feat = CACHE / "features" / "ek100"
vids = Path(sys.argv[1]).read_text().split()
while True:
    metas = [feat / f"{v}.json" for v in vids]
    missing = [m for m in metas if not m.exists() or "tcells" not in json.loads(m.read_text()).get("arrays", [])]
    if not missing:
        break
    print(f"{len(missing)} videos without tcells", flush=True)
    time.sleep(120)
