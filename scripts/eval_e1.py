"""E1: recall NLL and memory gain vs baselines (Gate 1).

    python scripts/eval_e1.py --ckpt checkpoints/stage_a/step13000.pt --split train --batch 2
"""
import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wcem.data.cache import TRIG, u16_slice, train_val_ids  # noqa: E402
from wcem.eval.e1 import (  # noqa: E402
    LAGS,
    accumulate_nll_gain,
    fifo_byte_cap_g,
    gain_vs_swap,
    nll_by_lag,
    query_taus,
    seq_nll,
    uniform_writes,
)
from wcem.eval.triggers import writes_at_rate  # noqa: E402
from wcem.models.wcem import Config, WCEM  # noqa: E402
from wcem.paths import CACHE, CHECKPOINTS  # noqa: E402
from wcem.train_util import model_config  # noqa: E402

TRIGGERS = ("featdiff", "err_raw", "err_norm", "fd_raw", "fd_norm")


def load_rate(log):
    p = Path(log)
    if not p.exists():
        return 0.05
    for line in reversed(p.read_text().splitlines()):
        if line.strip().startswith("{"):
            return json.loads(line).get("writes_per_step", 0.05)
    return 0.05


def iter_videos(root, ids, length, surprise_key="surprise", window_stride=0):
    """Yield (vid, start). stride 0 means a single window at start=0."""
    for vid in sorted(ids):
        meta = root / f"{vid}.json"
        if not meta.exists():
            continue
        n = int(json.loads(meta.read_text())["steps"])
        if n < length:
            continue
        if window_stride <= 0:
            yield vid, 0
        else:
            for start in range(0, n - length + 1, window_stride):
                yield vid, start


def load_window(root, vid, length, surprise_key="surprise", start=0):
    b = start + length
    tokens = u16_slice(root / f"{vid}_slot.npy", start, length)
    z = torch.from_numpy(np.array(np.load(root / f"{vid}_z.npy", mmap_mode="r")[start:b], copy=True))
    surprise = torch.from_numpy(np.load(root / f"{vid}_{surprise_key}.npy")[start:b].copy())
    t = np.load(root / f"{vid}_trig.npy", mmap_mode="r")[start:b]
    trigs = {name: torch.from_numpy(np.array(t[:, i], copy=True)) for i, name in enumerate(TRIG)}
    sur_err = torch.from_numpy(np.load(root / f"{vid}_surprise_err.npy", mmap_mode="r")[start:b].copy())
    return tokens, z, surprise, trigs, sur_err


def to_device(batch, device):
    tokens = torch.stack([b[0] for b in batch]).to(device, non_blocking=True)
    z = torch.stack([b[1] for b in batch]).to(device, non_blocking=True)
    surprise = torch.stack([b[2] for b in batch]).to(device, non_blocking=True)
    return tokens.float(), z.float(), surprise.float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--split", choices=("val", "train", "all"), default="val")
    ap.add_argument("--length", type=int, default=512)
    ap.add_argument("--rate", type=float)
    ap.add_argument("--log", type=Path, default=Path("data/logs/stage_a.jsonl"))
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--log-lam", type=float, default=-4.6)
    ap.add_argument("--set-sigma-q", type=float)
    ap.add_argument("--sigma-obs", type=float, help="Kalman observation noise override")
    ap.add_argument("--disable-slow", action="store_true", help="L off (fast-path E1 regression check)")
    ap.add_argument("--max-videos", type=int, default=0)
    ap.add_argument("--window-stride", type=int, default=0, help=">0: extra windows per video (e.g. 256)")
    args = ap.parse_args()
    if args.batch is None:
        args.batch = 2 if args.split == "train" else 4
    device = "cuda"
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg") or {}
    model = WCEM(model_config(cfg) if cfg else Config()).to(device).eval()
    model.load_state_dict(ckpt["model"], strict=False)
    del ckpt
    if args.set_sigma_q is not None:
        model.memory.sigma_q = args.set_sigma_q
    if args.sigma_obs is not None:
        model.cfg.sigma_obs = args.sigma_obs
    if args.disable_slow:
        model.cfg.slow_enabled = False
        model.slow.enabled = False
    rate = args.rate if args.rate is not None else load_rate(args.log)
    train_ids, val_ids = train_val_ids()
    ids = train_ids if args.split == "train" else val_ids if args.split == "val" else train_ids | val_ids
    root = CACHE / "features" / "ek100"
    sk = cfg.get("surprise_key", "surprise")
    windows = list(iter_videos(root, ids, args.length, sk, args.window_stride))
    if args.max_videos:
        windows = windows[: args.max_videos]
    n_vid = len({v for v, _ in windows})
    print(f"{len(windows)} windows / {n_vid} videos ({args.split}), batch {args.batch}, "
          f"stride {args.window_stride}, rate {rate:.4f}", flush=True)

    taus, used = query_taus(args.length, model.cfg.window, LAGS)
    lag_t = torch.tensor(used)
    lam_val = args.log_lam

    own_parts, swap_parts = [], []
    cue_parts, fifo_parts = [], []
    trig_acc = {name: [] for name in TRIGGERS}
    trig_acc["uniform"] = []
    err_parts = []

    batch = []
    for i, (vid, start) in enumerate(windows):
        batch.append((*load_window(root, vid, args.length, sk, start), vid))
        if len(batch) < args.batch and i < len(windows) - 1:
            continue
        tokens, z, surprise = to_device(batch, device)
        trigs = [b[3] for b in batch]
        sur_err = torch.stack([b[4] for b in batch]).to(device).float()
        B, N = z.shape[:2]
        lam = torch.full((B,), lam_val, device=device)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(tokens, z, surprise, lam)
            _, own, swap = gain_vs_swap(model, out, z, taus, used)
            own_parts.append(own.mean(0).float())
            if swap is not None:
                swap_parts.append(swap.mean(0).float())
            cue_parts.append(seq_nll(model, out, z, torch.zeros_like(out["S_chunks"][-1]), taus).mean(0).float())
            o_err = model(tokens, z, sur_err, lam)
            err_parts.append(seq_nll(model, o_err, z, o_err["S_chunks"][-1], taus).mean(0).float())

            fifo_nll = []
            for b, tr in enumerate(trigs):
                g = fifo_byte_cap_g(N, rate, model.cfg.dk, model.cfg.dk).to(device)
                o = model(tokens[b:b + 1], z[b:b + 1], surprise[b:b + 1], lam[b:b + 1], g_force=g[None])
                fifo_nll.append(seq_nll(model, o, z[b:b + 1], o["S_chunks"][-1], taus).mean(0))
            fifo_parts.append(torch.stack(fifo_nll).mean(0).float())

            for name in TRIGGERS + ("uniform",):
                masks = []
                for tr in trigs:
                    if name == "uniform":
                        w = uniform_writes(N, rate / 2.0)
                    else:
                        w = writes_at_rate(tr[name][:N], rate / 2.0)
                    masks.append(w[:N])
                g = torch.stack(masks).to(device).float()[..., None].expand(-1, -1, 2)
                o = model(tokens, z, surprise, lam, g_force=g)
                trig_acc[name].append(seq_nll(model, o, z, o["S_chunks"][-1], taus).mean(0).float())

        batch.clear()
        del tokens, z, surprise, sur_err, out
        gc.collect()
        torch.cuda.empty_cache()
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(windows)} windows", flush=True)

    gain_all, _ = accumulate_nll_gain(own_parts, swap_parts, used, device) if swap_parts else ({}, len(windows))
    results = dict(
        ckpt=str(args.ckpt),
        split=args.split,
        videos=n_vid,
        windows=len(windows),
        window_stride=args.window_stride,
        rate=rate,
        log_lam=lam_val,
        sigma_q=model.memory.sigma_q,
        steps=args.length,
        trained=dict(
            nll=nll_by_lag(torch.stack(own_parts).mean(0), lag_t),
            gain=gain_all,
        ),
        cue=nll_by_lag(torch.stack(cue_parts).mean(0), lag_t),
        surprise_err=nll_by_lag(torch.stack(err_parts).mean(0), lag_t),
        fifo=nll_by_lag(torch.stack(fifo_parts).mean(0), lag_t),
    )
    for name, parts in trig_acc.items():
        results[name] = nll_by_lag(torch.stack(parts).mean(0), lag_t)

    print(json.dumps(results["trained"], indent=2), flush=True)
    out_path = args.out or (CHECKPOINTS / "gate1" / f"e1_{args.ckpt.stem}_{args.split}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
