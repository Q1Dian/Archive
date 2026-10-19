"""Train WCEM on cached EK100 windows (Stage A / gate tune / Stage B).

    python scripts/train.py --config configs/stage_a.yaml --resume checkpoints/stage_a/step6000.pt
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wcem.data.cache import WindowDataset, cached_batches, train_val_ids, video_catalog  # noqa: E402
from wcem.eval.stage_a import recall_gain, written_key_cosine, writes_per_minute  # noqa: E402
from wcem.models.wcem import WCEM, losses  # noqa: E402
from wcem.paths import CACHE, CHECKPOINTS, LOGS, MANIFESTS  # noqa: E402
from wcem.train_slow import slow_residual_loss  # noqa: E402
from wcem.train_util import (  # noqa: E402
    freeze_decoder_tune,
    freeze_joint_readout_tune,
    freeze_representation,
    freeze_stage_a,
    model_config,
    reinit_gate_biases,
    sample_log_lam,
)


def load_cfg(path):
    return yaml.safe_load(Path(path).read_text())


def make_loader(root, ids, cfg, train=True):
    ds = WindowDataset(root, ids, length=cfg["length"], surprise_key=cfg.get("surprise_key", "surprise"))
    kw = dict(batch_size=cfg["batch"], shuffle=train, num_workers=cfg.get("workers", 0),
              drop_last=True, pin_memory=False, persistent_workers=False,
              collate_fn=lambda batch: (torch.stack([b[0] for b in batch]),
                                        torch.stack([b[1] for b in batch]),
                                        torch.stack([b[2] for b in batch])))
    if kw["num_workers"] > 0:
        kw["prefetch_factor"] = 1
    return torch.utils.data.DataLoader(ds, **kw), ds


@torch.no_grad()
def eval_logs(model, out, z, logs):
    gain, clamp = recall_gain(model, out, z)
    logs = dict(logs)
    logs["writes_per_min"] = writes_per_minute(logs["writes_per_step"])
    logs["key_cosine"] = written_key_cosine(out["k"], out["p_open"])
    logs["sigma_clamp"] = clamp
    logs.update({f"gain_{k}": v for k, v in gain.items()})
    return logs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/stage_a.yaml"))
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--resume", type=Path)
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    run = cfg.get("run_name", "stage_a")
    out_dir = CHECKPOINTS / run
    torch.manual_seed(cfg.get("seed", 0))
    feature_dataset = cfg.get("feature_dataset", "ek100")
    root = CACHE / "features" / feature_dataset
    if feature_dataset == "assembly101":
        from wcem.data.assembly101 import manifest_cache_ids  # noqa: E402
        mf = Path(cfg.get("manifest", MANIFESTS / "assembly101.txt"))
        train_ids = manifest_cache_ids(mf)
        val_ids = set()
    else:
        train_ids, val_ids = train_val_ids()
    cache_gb = cfg.get("cache_gb", 0)
    train_loader = None
    slow_only = cfg.get("slow_only", False)
    if cache_gb:
        cat = video_catalog(root, train_ids, cfg["length"], cfg.get("surprise_key", "surprise"))
        print(f"[{run}] resident cache {cache_gb} GB over {len(cat)} videos batch {cfg['batch']}", flush=True)
        stream = cached_batches(
            root, train_ids, cfg["length"], cfg["batch"], cfg.get("surprise_key", "surprise"),
            cache_gb=cache_gb, seed=cfg.get("seed", 0), rotate_every=cfg.get("rotate_every", 64),
            with_dupsim=slow_only)
    else:
        train_loader, train_ds = make_loader(root, train_ids, cfg, True)
        print(f"[{run}] windows train {len(train_ds)}", flush=True)
        stream = None
    if args.bench:
        import time
        t0 = time.time()
        it = iter(stream) if stream is not None else iter(train_loader)
        for _ in range(5):
            next(it)
        print(f"loader {5 / (time.time() - t0):.2f} batches/s", flush=True)
        return
    device = "cuda"
    mcfg = model_config(cfg)
    print(f"log_lam in [{math.exp(mcfg.log_lam_range[0]):.4g}, {math.exp(mcfg.log_lam_range[1]):.4g}] "
          f"log_sigma in {mcfg.log_sigma_range} slow={mcfg.slow_enabled}", flush=True)
    model = WCEM(mcfg).to(device)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    step = 0
    gate_tune = cfg.get("gate_tune", False)
    resume_path = args.resume or (Path(cfg["init_ckpt"]) if cfg.get("init_ckpt") else None)
    if resume_path:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        warm_start = cfg.get("init_ckpt") and not args.resume and Path(cfg["init_ckpt"]).resolve() == Path(resume_path).resolve()
        step = 0 if warm_start else ckpt.get("step", 0)
        if gate_tune:
            reinit_gate_biases(model, mcfg.gate_bias)
            print(f"gate-tune: reinitialized MLP_g biases to {mcfg.gate_bias}", flush=True)
        elif not warm_start and not cfg.get("decoder_tune") and not cfg.get("joint_readout_tune") and not cfg.get("freeze_repr"):
            opt.load_state_dict(ckpt["opt"])
    if cfg.get("joint_readout_tune"):
        freeze_joint_readout_tune(model, tune_slow=cfg.get("tune_slow", True))
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"joint_readout_tune: decoders + slow ({n} params)", flush=True)
    elif cfg.get("decoder_tune"):
        freeze_decoder_tune(model)
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"decoder_tune: training dec_seq/dec_comp only ({n} params)", flush=True)
    elif cfg.get("slow_only"):
        freeze_stage_a(model)
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"slow_only: training SlowMemory P/U ({n} params)", flush=True)
    elif cfg.get("freeze_repr"):
        freeze_representation(model)
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        print("frozen representation stack (Stage B joint)", flush=True)
        if resume_path and args.resume and "opt" in ckpt:
            try:
                opt.load_state_dict(ckpt["opt"])
            except ValueError as e:
                print(f"optimizer resume skipped: {e}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"{run}.jsonl"

    def gates_trainable(on):
        for g in model.memory.gates:
            for p in g.parameters():
                p.requires_grad = on

    warmup = cfg.get("rate_warmup", 1000)
    freeze_gates = cfg.get("freeze_gates_after_warmup", True)

    def sync_gates(s):
        if cfg.get("decoder_tune") or cfg.get("joint_readout_tune") or cfg.get("slow_only"):
            gates_trainable(False)
        elif gate_tune or cfg.get("freeze_repr"):
            gates_trainable(True)
        elif s < warmup:
            gates_trainable(False)
        else:
            gates_trainable(not freeze_gates)

    sync_gates(step)

    def next_batch():
        nonlocal it
        try:
            return next(it)
        except StopIteration:
            it = iter(stream) if stream is not None else iter(train_loader)
            return next(it)

    it = iter(stream) if stream is not None else iter(train_loader)
    model.train()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    while step < cfg["steps"]:
        batch = next_batch()
        if slow_only:
            tokens, z, surprise, dupsim = batch
        else:
            tokens, z, surprise = batch
            dupsim = None
        tokens = tokens.to(device, non_blocking=True).float()
        z = z.to(device, non_blocking=True).float()
        surprise = surprise.to(device, non_blocking=True).float()
        lam = sample_log_lam(tokens.shape[0], cfg, device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if slow_only:
                dupsim = dupsim.to(device, non_blocking=True).float()
                total, logs = slow_residual_loss(
                    model, tokens, z, surprise, dupsim,
                    sim_thresh=cfg.get("dup_sim_thresh", 0.85),
                    bootstrap_p=cfg.get("bootstrap_prob", 0.8),
                )
                out = None
            else:
                out = model(tokens, z, surprise, lam)
                rw = cfg.get("rate_weight_max", 1.0) if (gate_tune or cfg.get("freeze_repr")
                                                        or cfg.get("decoder_tune")
                                                        or cfg.get("joint_readout_tune")) else min(
                    cfg.get("rate_weight_max", 1.0), step / max(warmup, 1))
                total, logs = losses(model, out, z, lam, rate_weight=rw)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()
        step += 1
        sync_gates(step)
        if step % cfg["log_every"] == 0 or step == 1:
            model.eval()
            if slow_only:
                rec = dict(logs)
            else:
                rec = eval_logs(model, {k: v.float() if torch.is_tensor(v) and v.is_floating_point() else v
                                        for k, v in out.items()}, z, logs)
            rec["step"] = step
            rec["run"] = run
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
            with log_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            model.train()
        if step % cfg["save_every"] == 0:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step, cfg=cfg),
                       out_dir / f"step{step}.pt")
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step, cfg=cfg),
                       out_dir / "latest.pt")
    print(f"{run} complete", flush=True)


if __name__ == "__main__":
    main()
