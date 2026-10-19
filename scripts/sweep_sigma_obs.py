"""Sweep Kalman sigma_obs: E1 on full D2 val windows + E4 on train.

    ./.conda/bin/python -u scripts/sweep_sigma_obs.py --ckpt checkpoints/stage_a_kalman/step8000.pt
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = REPO / ".conda/bin/python"


def run(cmd):
    subprocess.run(cmd, cwd=REPO, check=True)


def read_e1_gain(path):
    d = json.loads(path.read_text())
    g = d["trained"]["gain"]
    return {k: g[k] for k in ("32-64", "64-128", "128-256")}, d.get("windows", d.get("videos"))


def read_e4(path):
    s = json.loads(path.read_text())["summary"]
    return dict(
        repeat_lt_first=s["repeat_lt_first"],
        repeat_minus_first=s["repeat_minus_first"],
        frac_videos=s["frac_videos_repeat_lt_first"],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=Path("checkpoints/stage_a_kalman/step8000.pt"))
    ap.add_argument("--sigmas", type=float, nargs="+",
                    default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0])
    ap.add_argument("--window-stride", type=int, default=128)
    args = ap.parse_args()
    out_dir = REPO / "checkpoints/gate1/sigma_obs_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for sig in args.sigmas:
        tag = f"sigma_{sig:g}".replace(".", "p")
        e1_out = out_dir / f"e1_{tag}.json"
        e4_out = REPO / "checkpoints/gate2" / f"e4_{tag}.json"
        run([str(PY), "-u", "scripts/eval_e1.py", "--ckpt", str(args.ckpt), "--split", "val",
             "--window-stride", str(args.window_stride), "--sigma-obs", str(sig), "--out", str(e1_out)])
        run([str(PY), "-u", "scripts/eval_e4.py", "--ckpt", str(args.ckpt), "--split", "train",
             "--sigma-obs", str(sig), "--out", str(e4_out), "--tb-run", f"sigma_obs_sweep"])
        g, nw = read_e1_gain(e1_out)
        e4 = read_e4(e4_out)
        row = dict(sigma_obs=sig, windows=nw, e1_gain_32_64=g["32-64"], e1_gain_64_128=g["64-128"],
                   e1_gain_128_256=g["128-256"], **e4)
        rows.append(row)
        print(json.dumps(row), flush=True)
    # Pareto: E4 pass and best min mid-lag E1
    ok = [r for r in rows if r["repeat_lt_first"]]
    good = [r for r in ok if r["e1_gain_32_64"] > 0 and r["e1_gain_64_128"] > 0]
    pool = good if good else ok
    best = None
    if pool:
        best = max(pool, key=lambda r: min(r["e1_gain_32_64"], r["e1_gain_64_128"]))
    summary = dict(ckpt=str(args.ckpt), sweep=rows, best_sigma_obs=best["sigma_obs"] if best else None,
                   best_row=best)
    path = out_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2))
    print(f"wrote {path}", flush=True)
    if best:
        print(f"recommended sigma_obs={best['sigma_obs']}", flush=True)


if __name__ == "__main__":
    main()
