"""Gate-tune readout check: Ballé noise only affects *training* forwards (model.train()).

At eval, sigma_q is unused; compare train-mode write stats on a val batch instead.

    ./.conda/bin/python -u scripts/investigate_sigma_q.py --ckpt checkpoints/stage_a_gate_tune/step8000.pt
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wcem.data.cache import WindowDataset, train_val_ids  # noqa: E402
from wcem.models.wcem import WCEM  # noqa: E402
from wcem.paths import CACHE, CHECKPOINTS  # noqa: E402
from wcem.train_util import model_config  # noqa: E402


@torch.no_grad()
def write_stats(model, tokens, z, surprise, lam, train):
    model.train(train)
    out = model(tokens, z, surprise, lam)
    delta = out["delta"]
    return dict(
        p_open=float(out["p_open"].mean()),
        delta_rms=float(delta.float().square().mean().sqrt()),
        delta_given_open=float(delta[out["p_open"] > 0.5].float().square().mean().sqrt())
        if (out["p_open"] > 0.5).any() else 0.0,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    args = ap.parse_args()
    device = "cuda"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt.get("cfg") or {}
    model = WCEM(model_config(cfg)).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    _, val = train_val_ids()
    root = CACHE / "features" / "ek100"
    ds = WindowDataset(root, sorted(val), 512)
    for tokens, z, surprise, vid, start in ds:
        if start == 0:
            break
    tokens = tokens[None].to(device).float()
    z = z[None].to(device).float()
    surprise = surprise[None].to(device).float()
    lam = torch.full((1,), -3.0, device=device)
    rows = {}
    for sq in (0.0, 0.01, 0.05):
        model.memory.sigma_q = sq
        rows[f"sigma_q={sq}"] = {
            "eval": write_stats(model, tokens, z, surprise, lam, False),
            "train": write_stats(model, tokens, z, surprise, lam, True),
        }
    note = ("Ballé noise is injected only when model.training and sigma_q>0. "
            "E1 eval runs in eval() so sigma_q does not change held-out NLL; "
            "train-mode delta_rms shows write corruption scale.")
    out = dict(ckpt=str(args.ckpt), note=note, stats=rows)
    path = CHECKPOINTS / "gate1" / "sigma_q_investigation.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
