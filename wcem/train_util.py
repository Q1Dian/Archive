"""Shared Stage A helpers."""
import math

import torch

from wcem.models.wcem import Config, WCEM


def model_config(cfg):
    c = Config()
    c.log_sigma_range = (cfg.get("log_sigma_min", -3.0), cfg.get("log_sigma_max", 2.0))
    c.log_lam_range = (math.log(cfg.get("log_lam_min", 1e-3)), math.log(cfg.get("log_lam_max", 1e-1)))
    c.gate_bias = cfg.get("gate_bias", 2.0)
    c.checkpoint = cfg.get("checkpoint", True)
    c.slow_enabled = cfg.get("slow_enabled", False)
    c.slow_read = cfg.get("slow_read", False)
    c.precision_fusion = cfg.get("precision_fusion", False)
    c.kalman_gate = cfg.get("kalman_gate", False)
    c.sigma_obs = cfg.get("sigma_obs", 1.0)
    c.decay_on_write_only = cfg.get("decay_on_write_only", False)
    return c


def freeze_representation(model):
    """Stage B: slots, window, keys, cues, cell encoder."""
    for mod in (model.tokenizer, model.slot_head, model.window, model.w_k, model.w_v,
                model.w_cue, model.e_c, model.w_cue_comp):
        for p in mod.parameters():
            p.requires_grad = False


def freeze_decoder_tune(model: WCEM):
    """Kalman write path frozen; train recall decoders only."""
    for p in model.parameters():
        p.requires_grad = False
    for mod in (model.dec_seq, model.dec_comp):
        for p in mod.parameters():
            p.requires_grad = True


def freeze_joint_readout_tune(model: WCEM, tune_slow: bool = True):
    """Joint Stage B readout pass: decoders (+ optional slow) with write path frozen."""
    freeze_decoder_tune(model)
    if tune_slow:
        for p in model.slow.parameters():
            p.requires_grad = True


def freeze_stage_a(model: WCEM):
    """Stage B L-only: freeze perception, episodic write path, decoders; train slow P/U."""
    for p in model.parameters():
        p.requires_grad = False
    model.slow.enabled = model.cfg.slow_enabled
    for p in model.slow.parameters():
        p.requires_grad = True


def sample_log_lam(batch, cfg, device):
    lo, hi = model_config(cfg).log_lam_range
    return torch.empty(batch, device=device).uniform_(lo, hi)


def reinit_gate_biases(model: WCEM, bias: float):
    for gate in model.memory.gates:
        with torch.no_grad():
            gate.net[-1].bias.zero_()
            gate.net[-1].bias[0] = bias
