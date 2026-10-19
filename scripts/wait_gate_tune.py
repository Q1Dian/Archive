"""Wait for frozen Stage A step5000.pt, then launch gate tuning (single-GPU queue).

    scripts/run_capped.sh 20G data/logs/wait_gate_tune.out \\
        ./.conda/bin/python -u scripts/wait_gate_tune.py
"""
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wcem.paths import CHECKPOINTS, REPO  # noqa: E402

FROZEN = CHECKPOINTS / "stage_a_frozen" / "step5000.pt"
ALT = CHECKPOINTS / "stage_a" / "step5000.pt"
RESUME = FROZEN if FROZEN.exists() else ALT
LOG = REPO / "data/logs/stage_a_gate_tune.out"


def main():
    while not RESUME.exists():
        print(f"waiting for {RESUME}", flush=True)
        time.sleep(120)
    if subprocess.run(["pgrep", "-f", "scripts/train.py"], capture_output=True).returncode == 0:
        print("waiting for other train.py to exit", flush=True)
        while subprocess.run(["pgrep", "-f", "scripts/train.py"], capture_output=True).returncode == 0:
            time.sleep(60)
    cmd = ["scripts/run_capped.sh", "20G", str(LOG),
           "./.conda/bin/python", "-u", "scripts/train.py",
           "--config", "configs/stage_a_gate_tune.yaml", "--resume", str(RESUME)]
    print("launching gate tune:", " ".join(cmd), flush=True)
    subprocess.Popen(cmd, cwd=REPO)


if __name__ == "__main__":
    main()
