"""Re-run Gate 1/2 evals on key checkpoints (after Stage B completes).

    ./.conda/bin/python -u scripts/gate_eval_sweep.py
"""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = REPO / ".conda/bin/python"
CKPTS = [
    ("gate_tune", REPO / "checkpoints/stage_a_gate_tune/step8000.pt", "data/logs/stage_a_gate_tune.jsonl"),
    ("stage_b_2000", REPO / "checkpoints/stage_b/step2000.pt", "data/logs/stage_b.jsonl"),
    ("stage_b_4500", REPO / "checkpoints/stage_b/step4500.pt", "data/logs/stage_b.jsonl"),
    ("stage_b_6000", REPO / "checkpoints/stage_b/step6000.pt", "data/logs/stage_b.jsonl"),
]


def run_e1(name, ckpt, log):
    if not ckpt.exists():
        print(f"skip {name}: missing {ckpt}", flush=True)
        return
    out = REPO / "checkpoints/gate1" / f"e1_{name}.json"
    subprocess.run([str(PY), "-u", "scripts/eval_e1.py", "--ckpt", str(ckpt), "--log", log],
                   cwd=REPO, check=True)
    src = REPO / "checkpoints/gate1/e1.json"
    if src.exists():
        out.write_text(src.read_text())
    data = json.loads(out.read_text())
    gain = data.get("trained", {}).get("gain", {})
    print(f"{name} gain_32-64={gain.get('32-64')} writes_rate={data.get('rate')}", flush=True)


def main():
    for name, ckpt, log in CKPTS:
        run_e1(name, ckpt, log)
    subprocess.run([str(PY), "-u", "scripts/eval_e2.py",
                    "--ckpt", str(REPO / "checkpoints/stage_a_gate_tune/step8000.pt")], cwd=REPO, check=True)
    subprocess.run([str(PY), "-u", "scripts/eval_e4.py",
                    "--ckpt", str(REPO / "checkpoints/stage_b/step6000.pt")], cwd=REPO, check=True)
    print("gate_eval_sweep done", flush=True)


if __name__ == "__main__":
    main()
