"""Phase A baseline: E4 and E2 log_lam sweeps on PairGate checkpoints.

    ./.conda/bin/python -u scripts/gate_lam_sweep.py
"""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = REPO / ".conda/bin/python"

SWEEPS = [
    ("step13000", REPO / "checkpoints/stage_a/step13000.pt", "train"),
    ("step8000_gate_tune", REPO / "checkpoints/stage_a_gate_tune/step8000.pt", "train"),
]

# Artifacts written by this script (summary only; sweeps may be run standalone).


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO, check=True)


def main():
    for name, ckpt, split in SWEEPS:
        if not ckpt.exists():
            print(f"skip {name}: missing {ckpt}", flush=True)
            continue
        run([str(PY), "-u", "scripts/eval_e4_sweep.py", "--ckpt", str(ckpt), "--split", split,
             "--out", str(REPO / f"checkpoints/gate2/e4_sweep_{name}_{split}.json"),
             "--tb-run", f"e4_sweep_{name}"])
        run([str(PY), "-u", "scripts/eval_e2.py", "--ckpt", str(ckpt),
             "--out", str(REPO / f"checkpoints/gate1/e2_sweep_{name}.json"),
             "--log-lams", "-5.0", "-4.5", "-4.0", "-3.5", "-3.0", "-2.5", "-2.0"])
    summary = REPO / "checkpoints/gate1/lam_sweep_summary.json"
    rows = []
    for name, ckpt, _ in SWEEPS:
        row = dict(checkpoint=name)
        e4p = REPO / f"checkpoints/gate2/e4_sweep_{name}_train.json"
        e2p = REPO / f"checkpoints/gate1/e2_sweep_{name}.json"
        if e4p.exists():
            e4 = json.loads(e4p.read_text())
            row["e4_best_repeat_minus_first"] = e4.get("best_repeat_minus_first")
        if e2p.exists():
            e2 = json.loads(e2p.read_text())
            row["e2_beats_uniform_count"] = e2.get("beats_uniform_count")
            row["e2_beats_uniform_all"] = e2.get("beats_uniform_all")
        if len(row) > 1:
            rows.append(row)
    summary.write_text(json.dumps(rows, indent=2))
    print(f"wrote {summary}", flush=True)


if __name__ == "__main__":
    main()
