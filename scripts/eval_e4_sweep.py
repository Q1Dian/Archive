"""E4 λ sweep: repeat vs first write prob vs log_lam (Gate 2 baseline).

    python scripts/eval_e4_sweep.py --ckpt checkpoints/stage_a_gate_tune/step8000.pt
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wcem.data.cache import train_val_ids, u16_slice  # noqa: E402
from wcem.eval.e4 import SIM_THRESH, repeat_write_stats  # noqa: E402
from wcem.models.wcem import Config, WCEM  # noqa: E402
from wcem.paths import CACHE, CHECKPOINTS, LOGS  # noqa: E402
from wcem.train_util import model_config  # noqa: E402


def log_tensorboard(run, step, metrics):
    LOGS.mkdir(parents=True, exist_ok=True)
    rec = dict(run=run, step=int(step), **metrics)
    with (LOGS / "eval_metrics.jsonl").open("a") as f:
        f.write(json.dumps(rec) + "\n")


def iter_videos(root, ids, length):
    for vid in sorted(ids):
        meta = root / f"{vid}.json"
        dup = root / f"{vid}_dupsim.npy"
        if not meta.exists() or not dup.exists():
            continue
        n = int(json.loads(meta.read_text())["steps"])
        if n < length:
            continue
        yield vid


def load_window(root, vid, length, surprise_key="surprise"):
    tokens = u16_slice(root / f"{vid}_slot.npy", 0, length)
    z = torch.from_numpy(np.array(np.load(root / f"{vid}_z.npy", mmap_mode="r")[:length], copy=True))
    surprise = torch.from_numpy(np.load(root / f"{vid}_{surprise_key}.npy")[:length].copy())
    dupsim = np.load(root / f"{vid}_dupsim.npy", mmap_mode="r")[:length]
    return tokens, z, surprise, dupsim


def run_one(model, root, vids, length, sk, log_lam, sim_thresh, device):
    rows = []
    for vid in vids:
        tokens, z, surprise, dupsim = load_window(root, vid, length, sk)
        rows.append(repeat_write_stats(
            model, tokens, z.to(device), surprise.to(device), dupsim, log_lam, sim_thresh=sim_thresh))
    pf = float(np.nanmean([r["p_write_first"] for r in rows]))
    pr = float(np.nanmean([r["p_write_repeat"] for r in rows]))
    return dict(
        log_lam=log_lam,
        lam=float(np.exp(log_lam)),
        p_open_mean=float(np.nanmean([r["p_open_mean"] for r in rows])),
        p_open_std=float(np.nanmean([r["p_open_std"] for r in rows])),
        p_write_first=pf,
        p_write_repeat=pr,
        repeat_minus_first=pr - pf,
        repeat_over_first=(pr / pf) if pf > 0 else float("nan"),
        repeat_lt_first=pr < pf,
        frac_videos_repeat_lt_first=float(np.mean([
            1.0 if r["repeat_lt_first"] else 0.0 for r in rows if r["repeat_lt_first"] is not None
        ])) if rows else 0.0,
        videos=len(rows),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--split", choices=("train", "val"), default="train")
    ap.add_argument("--length", type=int, default=512)
    ap.add_argument("--max-videos", type=int, default=0)
    ap.add_argument("--log-lams", type=float, nargs="+",
                    default=[-6.0, -5.5, -5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0])
    ap.add_argument("--sim-thresh", type=float, default=SIM_THRESH)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--tb-run", type=str)
    args = ap.parse_args()
    device = "cuda"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt.get("cfg") or {}
    sk = cfg.get("surprise_key", "surprise")
    model = WCEM(model_config(cfg) if cfg else Config()).to(device).eval()
    model.load_state_dict(ckpt["model"], strict=False)
    train_ids, val_ids = train_val_ids()
    ids = train_ids if args.split == "train" else val_ids
    root = CACHE / "features" / "ek100"
    vids = list(iter_videos(root, ids, args.length))
    if args.max_videos:
        vids = vids[: args.max_videos]
    stem = args.ckpt.stem
    tb_run = args.tb_run or f"e4_sweep_{stem}"
    sweep = []
    for ll in args.log_lams:
        row = run_one(model, root, vids, args.length, sk, ll, args.sim_thresh, device)
        sweep.append(row)
        step = int(stem.replace("step", "")) if stem.startswith("step") else 0
        log_tensorboard(f"{tb_run}", step * 1000 + int(-ll * 100), {
            "p_write_first": row["p_write_first"],
            "p_write_repeat": row["p_write_repeat"],
            "repeat_minus_first": row["repeat_minus_first"],
            "repeat_lt_first": 1.0 if row["repeat_lt_first"] else 0.0,
            "p_open_mean": row["p_open_mean"],
            "log_lam_eval": ll,
        })
        print(json.dumps(row), flush=True)
    best = min(sweep, key=lambda r: r["repeat_minus_first"])
    out = dict(ckpt=str(args.ckpt), split=args.split, videos=len(vids), sweep=sweep,
               best_repeat_minus_first=best)
    path = args.out or (CHECKPOINTS / "gate2" / f"e4_sweep_{stem}_{args.split}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
