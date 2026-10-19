"""Resume after reboot: gate tune -> E1/E2 -> Stage B -> E4.

    scripts/run_capped.sh 24G data/logs/pipeline_gate2.out \\
        ./.conda/bin/python -u scripts/pipeline_gate2.py
"""
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wcem.paths import CHECKPOINTS, REPO  # noqa: E402

PY = REPO / ".conda/bin/python"
CAP = "24G"


def run(*cmd):
    full = [str(PY), "-u", *cmd]
    print("run:", " ".join(full), flush=True)
    subprocess.run(full, cwd=REPO, check=True)


def wait_ckpt(path, every=60):
    while not path.exists():
        print(f"waiting for {path}", flush=True)
        time.sleep(every)


def main():
    repr_ckpt = CHECKPOINTS / "stage_a" / "step13000.pt"
    if not repr_ckpt.exists():
        repr_ckpt = CHECKPOINTS / "stage_a" / "latest.pt"
    gate_latest = CHECKPOINTS / "stage_a_gate_tune" / "latest.pt"
    if not gate_latest.exists():
        run("scripts/train.py", "--config", "configs/stage_a_gate_tune.yaml", "--resume", str(repr_ckpt))
        wait_ckpt(gate_latest)
    run("scripts/eval_e1.py", "--ckpt", str(gate_latest), "--log", "data/logs/stage_a_gate_tune.jsonl")
    run("scripts/eval_e2.py", "--ckpt", str(gate_latest))
    stage_latest = CHECKPOINTS / "stage_b" / "latest.pt"
    if not stage_latest.exists():
        cfg = REPO / "configs/stage_b.yaml"
        import yaml
        y = yaml.safe_load(cfg.read_text())
        y["init_ckpt"] = str(gate_latest)
        tmp = REPO / "configs/stage_b_active.yaml"
        tmp.write_text(yaml.dump(y))
        run("scripts/train.py", "--config", str(tmp))
        wait_ckpt(stage_latest)
    run("scripts/eval_e4.py", "--ckpt", str(stage_latest))
    print("pipeline_gate2 finished", flush=True)


if __name__ == "__main__":
    main()
