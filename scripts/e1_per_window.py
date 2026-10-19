"""Per-window E1 NLL vs MSE gain (64–128 diagnostic on small val sets).

    python scripts/e1_per_window.py --ckpt checkpoints/stage_a_decoder_tune/step4000.pt --sigma-obs 0.05
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.eval_e1 import iter_videos, load_window, to_device  # noqa: E402
from wcem.data.cache import train_val_ids  # noqa: E402
from wcem.eval.e1 import LAGS, gain_vs_swap, query_taus  # noqa: E402
from wcem.models.wcem import Config, WCEM  # noqa: E402
from wcem.paths import CACHE  # noqa: E402
from wcem.train_util import model_config  # noqa: E402


def mse_per_query(mu, z_q):
    return (z_q - mu).float().square().sum(dim=(-1, -2))


@torch.no_grad()
def per_window_64_128(model, tokens, z, surprise, lam_val, length, device, vid, start):
    tokens = tokens.to(device).float()
    z = z.to(device).float()
    surprise = surprise.to(device).float()
    B = z.shape[0]
    lam = torch.full((B,), lam_val, device=device)
    out = model(tokens, z, surprise, lam)
    taus, used = query_taus(length, model.cfg.window, LAGS)
    _, own, swap = gain_vs_swap(model, out, z, taus, used)
    if own.numel() == 0 or B < 2:
        return None
    lag = torch.tensor(used, device=device)
    m64 = (lag >= 64) & (lag < 128)
    if not m64.any():
        return None
    S, S_swap = out["S_chunks"][-1], out["S_chunks"][-1].roll(1, 0)
    tau = torch.tensor(taus, device=device)
    idx = tau[None].expand(B, -1)
    x_q = out["x_lag"].gather(1, idx[..., None].expand(-1, -1, out["x_lag"].shape[-1])).float()
    z_q = z.gather(1, idx[..., None, None].expand(-1, -1, *z.shape[2:]))
    mu_o, _ = model.read_seq(S.float(), x_q)
    mu_s, _ = model.read_seq(S_swap.float(), x_q)
    mse = mse_per_query(mu_s, z_q) - mse_per_query(mu_o, z_q)
    nll_g = swap - own
    rows = []
    for b in range(B):
        rows.append(
            dict(
                video=vid[b] if isinstance(vid, list) else vid,
                start=start[b] if isinstance(start, list) else start,
                nll_gain_64_128=nll_g[b, m64].mean().item(),
                mse_gain_64_128=mse[b, m64].mean().item(),
            )
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--window-stride", type=int, default=128)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--log-lam", type=float, default=-4.6)
    ap.add_argument("--sigma-obs", type=float)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    device = "cuda"
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg") or {}
    model = WCEM(model_config(cfg) if cfg else Config()).to(device).eval()
    model.load_state_dict(ckpt["model"], strict=False)
    if args.sigma_obs is not None:
        model.cfg.sigma_obs = args.sigma_obs
    _, val = train_val_ids()
    root = CACHE / "features" / "ek100"
    sk = cfg.get("surprise_key", "surprise")
    windows = list(iter_videos(root, val, 512, sk, args.window_stride))
    all_rows = []
    batch = []
    for vid, start in windows:
        batch.append((*load_window(root, vid, 512, sk, start), vid, start))
        if len(batch) < args.batch and len(all_rows) + len(batch) < len(windows):
            continue
        tokens, z, surprise = to_device([(b[0], b[1], b[2]) for b in batch], device)
        vids = [b[5] for b in batch]
        starts = [b[6] for b in batch]
        part = per_window_64_128(
            model, tokens, z, surprise, args.log_lam, 512, device, vids, starts
        )
        if part:
            all_rows.extend(part)
            for r in part:
                print(json.dumps(r), flush=True)
        batch.clear()

    nlls = [r["nll_gain_64_128"] for r in all_rows]
    mses = [r["mse_gain_64_128"] for r in all_rows]
    nlls_sorted = sorted(nlls)
    summary = dict(
        ckpt=str(args.ckpt),
        sigma_obs=model.cfg.sigma_obs,
        windows=len(all_rows),
        nll_gain_64_128_mean=sum(nlls) / len(nlls) if nlls else None,
        nll_gain_64_128_median=nlls_sorted[len(nlls) // 2] if nlls else None,
        mse_gain_64_128_mean=sum(mses) / len(mses) if mses else None,
        n_positive_nll_64_128=sum(1 for x in nlls if x > 0),
        n_positive_mse_64_128=sum(1 for x in mses if x > 0),
        per_window=all_rows,
    )
    out = args.out or Path(f"checkpoints/gate1/e1_per_window_{args.ckpt.stem}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in summary if k != "per_window"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
