"""Mirror training JSONL logs into TensorBoard event files.

    ./.conda/bin/python -u scripts/tb_watch.py
    tensorboard --logdir data/logs/tb --port 6006
"""
import json
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

REPO = Path(__file__).resolve().parent.parent
LOGS = REPO / "data/logs"
TB = LOGS / "tb"
RUNS = ("stage_a", "stage_a_gate_tune", "stage_b", "stage_b_assembly101", "stage_b_joint",
        "stage_b_joint_readout", "stage_a_kalman", "stage_a_decoder_tune")

EVAL_LOG = LOGS / "eval_metrics.jsonl"

GROUPS = {
    "seq_nll_per_dim": "loss",
    "comp_nll_per_dim": "loss",
    "slot_mse": "loss",
    "total": "loss",
    "writes_per_min": "memory",
    "writes_per_step": "memory",
    "size_per_step": "memory",
    "rate": "memory",
    "rate_weight": "memory",
    "key_cosine": "memory",
    "key_align": "memory",
    "sigma_clamp": "memory",
    "slow_residual": "slow",
    "slow_spread": "slow",
    "slow_repeat_frac": "slow",
}

EVAL_GROUPS = {
    "p_open_mean": "gate2",
    "p_open_std": "gate2",
    "p_write_first": "gate2",
    "p_write_repeat": "gate2",
    "repeat_minus_first": "gate2",
    "repeat_over_first": "gate2",
    "repeat_lt_first": "gate2",
    "frac_videos_repeat_lt_first": "gate2",
    "e4_videos": "gate2",
    "e4_n_repeat_steps": "gate2",
    "beats_uniform_count": "gate1",
    "beats_uniform_all": "gate1",
}


def emit(writer, rec):
    step = int(rec["step"])
    for key, value in rec.items():
        if not isinstance(value, (int, float)) or key == "step":
            continue
        group = GROUPS.get(key, "gain" if key.startswith("gain_") else "other")
        writer.add_scalar(f"{group}/{key}", float(value), step)


def emit_eval(writer, rec):
    step = int(rec.get("step", 0))
    for key, value in rec.items():
        if key in ("run", "step") or not isinstance(value, (int, float)):
            continue
        group = EVAL_GROUPS.get(key, "eval")
        writer.add_scalar(f"{group}/{key}", float(value), step)


def sync_run(name, writer, seen):
    src = LOGS / f"{name}.jsonl"
    if not src.exists():
        return seen
    with src.open() as f:
        lines = f.readlines()
    for line in lines[seen:]:
        if not line.strip():
            continue
        emit(writer, json.loads(line))
    if len(lines) != seen:
        writer.flush()
    return len(lines)


def sync_eval(writers, seen):
    if not EVAL_LOG.exists():
        return seen
    with EVAL_LOG.open() as f:
        lines = f.readlines()
    for line in lines[seen:]:
        if not line.strip():
            continue
        rec = json.loads(line)
        name = rec.get("run", "eval")
        if name not in writers:
            writers[name] = SummaryWriter(str(TB / name))
        emit_eval(writers[name], rec)
    if len(lines) != seen:
        for w in writers.values():
            w.flush()
    return len(lines)


def main():
    TB.mkdir(parents=True, exist_ok=True)
    writers = {n: SummaryWriter(str(TB / n)) for n in RUNS}
    seen = {n: 0 for n in RUNS}
    eval_seen = 0
    for name in RUNS:
        seen[name] = sync_run(name, writers[name], 0)
    eval_seen = sync_eval(writers, 0)
    while True:
        for name in RUNS:
            seen[name] = sync_run(name, writers[name], seen[name])
        eval_seen = sync_eval(writers, eval_seen)
        time.sleep(5)


if __name__ == "__main__":
    main()
