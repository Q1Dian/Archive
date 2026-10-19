"""Wait for Stage B step 6000, then run E1 + E4 on latest checkpoint."""
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wcem.paths import CHECKPOINTS, REPO  # noqa: E402

PY = REPO / ".conda/bin/python"
CKPT = CHECKPOINTS / "stage_b"
TARGET = CKPT / "step6000.pt"
LATEST = CKPT / "latest.pt"
LOG = REPO / "data/logs/stage_b.jsonl"


def max_step():
    if not LOG.exists():
        return 0
    m = 0
    for line in LOG.read_text().splitlines():
        if not line.strip():
            continue
        try:
            import json
            m = max(m, int(json.loads(line)["step"]))
        except Exception:
            pass
    return m


def run(*cmd):
    subprocess.run([str(PY), "-u", *cmd], cwd=REPO, check=True)


def main():
    while not TARGET.exists() and max_step() < 6000:
        print(f"stage_b: step {max_step()}, waiting for 6000", flush=True)
        time.sleep(30)
    while subprocess.run(["pgrep", "-f", "scripts/train.py.*stage_b"], capture_output=True).returncode == 0:
        print("waiting for stage_b train.py to exit", flush=True)
        time.sleep(20)
    ckpt = TARGET if TARGET.exists() else LATEST
    print(f"evaluating {ckpt}", flush=True)
    run("scripts/eval_e1.py", "--ckpt", str(ckpt), "--log", "data/logs/stage_b.jsonl")
    run("scripts/eval_e4.py", "--ckpt", str(ckpt))
    print("finish_stage_b done", flush=True)


if __name__ == "__main__":
    main()
