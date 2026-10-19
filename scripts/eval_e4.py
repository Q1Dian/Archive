"""E4 on EK100: lower write prob on near-duplicate steps (Gate 2).

    python scripts/eval_e4.py --ckpt checkpoints/stage_a/step13000.pt --split train
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wcem.data.cache import u16_slice, train_val_ids  # noqa: E402
from wcem.eval.e4 import SIM_THRESH, repeat_write_stats  # noqa: E402
from wcem.models.wcem import Config, WCEM  # noqa: E402
from wcem.paths import CACHE, CHECKPOINTS, LOGS  # noqa: E402
from wcem.train_util import model_config  # noqa: E402


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


def log_tensorboard(run, step, metrics):
    """Append one summary row for tb_watch → TensorBoard."""
    LOGS.mkdir(parents=True, exist_ok=True)
    rec = dict(run=run, step=int(step), **metrics)
    with (LOGS / "eval_metrics.jsonl").open("a") as f:
        f.write(json.dumps(rec) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--split", choices=("train", "val", "all"), default="train")
    ap.add_argument("--length", type=int, default=512)
    ap.add_argument("--max-videos", type=int, default=0, help="0 = all eligible videos")
    ap.add_argument("--log-lam", type=float, default=-4.6)
    ap.add_argument("--sim-thresh", type=float, default=SIM_THRESH)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--tb-run", type=str, help="TensorBoard subdir under data/logs/tb/ (default: e4_<ckpt stem>)")
    ap.add_argument("--set-sigma-q", type=float)
    ap.add_argument("--sigma-obs", type=float)
    args = ap.parse_args()
    device = "cuda"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt.get("cfg") or {}
    sk = cfg.get("surprise_key", "surprise")
    model = WCEM(model_config(cfg) if cfg else Config()).to(device).eval()
    model.load_state_dict(ckpt["model"], strict=False)
    if args.set_sigma_q is not None:
        model.memory.sigma_q = args.set_sigma_q
    if args.sigma_obs is not None:
        model.cfg.sigma_obs = args.sigma_obs
    train_ids, val_ids = train_val_ids()
    ids = train_ids if args.split == "train" else val_ids if args.split == "val" else train_ids | val_ids
    root = CACHE / "features" / "ek100"
    vids = list(iter_videos(root, ids, args.length))
    if args.max_videos:
        vids = vids[: args.max_videos]
    print(f"E4: {len(vids)} videos ({args.split}), log_lam={args.log_lam}, sim>={args.sim_thresh}", flush=True)

    rows = []
    for i, vid in enumerate(vids):
        tokens, z, surprise, dupsim = load_window(root, vid, args.length, sk)
        rows.append(repeat_write_stats(
            model, tokens, z.to(device), surprise.to(device), dupsim,
            args.log_lam, sim_thresh=args.sim_thresh))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(vids)}", flush=True)

    pf = float(np.nanmean([r["p_write_first"] for r in rows]))
    pr = float(np.nanmean([r["p_write_repeat"] for r in rows]))
    agg = dict(
        p_open_mean=float(np.nanmean([r["p_open_mean"] for r in rows])),
        p_open_std=float(np.nanmean([r["p_open_std"] for r in rows])),
        p_write_first=pf,
        p_write_repeat=pr,
        repeat_minus_first=pr - pf,
        repeat_over_first=(pr / pf) if pf > 0 else float("nan"),
        repeat_lt_first=pr < pf,
        videos=len(rows),
        n_first_total=int(sum(r["n_first"] for r in rows)),
        n_repeat_total=int(sum(r["n_repeat"] for r in rows)),
        frac_videos_repeat_lt_first=float(np.mean([
            1.0 if r["repeat_lt_first"] else 0.0 for r in rows if r["repeat_lt_first"] is not None
        ])) if rows else 0.0,
        log_lam=args.log_lam,
        sim_thresh=args.sim_thresh,
        sigma_q=float(model.memory.sigma_q),
    )
    out = dict(
        ckpt=str(args.ckpt),
        split=args.split,
        log_lam=args.log_lam,
        sim_thresh=args.sim_thresh,
        per_video=rows,
        summary=agg,
    )
    out_path = args.out or (CHECKPOINTS / "gate2" / f"e4_{args.ckpt.stem}_{args.split}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    tb_run = args.tb_run or f"e4_{args.ckpt.stem}"
    ckpt_step = int(cfg.get("step") or 0)
    if not ckpt_step:
        stem = args.ckpt.stem
        if stem.startswith("step"):
            ckpt_step = int(stem.replace("step", ""))
    log_tensorboard(tb_run, ckpt_step, {
        "p_open_mean": agg["p_open_mean"],
        "p_open_std": agg["p_open_std"],
        "p_write_first": agg["p_write_first"],
        "p_write_repeat": agg["p_write_repeat"],
        "repeat_minus_first": agg["repeat_minus_first"],
        "repeat_over_first": agg["repeat_over_first"],
        "repeat_lt_first": 1.0 if agg["repeat_lt_first"] else 0.0,
        "frac_videos_repeat_lt_first": agg["frac_videos_repeat_lt_first"],
        "e4_videos": float(agg["videos"]),
        "e4_n_repeat_steps": float(agg["n_repeat_total"]),
    })
    print(json.dumps(agg, indent=2), flush=True)
    print(f"wrote {out_path}", flush=True)
    print(f"tb_watch → data/logs/tb/{tb_run}/", flush=True)


if __name__ == "__main__":
    main()
