"""Full model on cached features (canvas sections 2-4).

Per step n the cache provides:
    tokens[n]   [144, 1024]  pooled V-JEPA tokens of step n (slot input)
    z[n]        [16, 256]    whitened recall target
    surprise[n] [4]          summary of the causal 60 s z-scored 0.5 s token
                             feature difference (Gate 0). Stage 0 error is an
                             M8 baseline, not this input.

Heads 0-1 are sequence heads (key from x_{n-2}), heads 2-3 completion heads
(key from the full z_n at write, from a 25% view at read).
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from wcem.memory.episodic import EpisodicMemory
from wcem.memory.gates import Decay
from wcem.models.blocks import MLP, SelfBlock
from wcem.models.perception import CellDecoder, LocalWindow, SlotTokenizer
from wcem.models.slow_memory import SlowMemory

LAG = 2


@dataclass
class Config:
    in_dim: int = 1024
    n_tokens: int = 144
    slots: int = 8
    slot_dim: int = 256
    dim: int = 512
    window: int = 8
    cells: int = 16
    cell_dim: int = 256
    dk: int = 128
    cue_dim: int = 256
    comp_frac: float = 0.25
    sigma_q: float = 0.01
    alpha0: float = 0.9998
    chunk: int = 64
    log_sigma_range: tuple = (-3.0, 2.0)
    log_lam_range: tuple = (math.log(1e-3), math.log(1e-1))
    lam1_ratio: float = 0.1
    key_align: float = 0.1
    gate_bias: float = 2.0
    checkpoint: bool = True
    slow_enabled: bool = False
    slow_read: bool = False
    precision_fusion: bool = False
    kalman_gate: bool = False
    sigma_obs: float = 1.0
    decay_on_write_only: bool = False


def lagged(x, lag=LAG):
    """x_{n-lag} along dim 1, repeating x_0 at the start."""
    return torch.cat([x[:, :1].expand(-1, lag, *x.shape[2:]), x[:, :-lag]], 1)


class CellEncoder(nn.Module):
    """Encode a (possibly partial) view of the 16 target cells. mask: True = hidden."""

    def __init__(self, cell_dim, cells, dim, out):
        super().__init__()
        self.inp = nn.Linear(cell_dim, dim)
        self.pos = nn.Parameter(torch.randn(1, cells, dim) * 0.02)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.block = SelfBlock(dim, 4)
        self.out = nn.Linear(dim, out)

    def forward(self, z, mask=None):
        lead = z.shape[:-2]
        h = self.inp(z.flatten(0, -3))
        if mask is not None:
            m = mask.flatten(0, -2)[..., None]
            h = torch.where(m, self.mask_token.to(h.dtype), h)
        h = self.block(h + self.pos)
        return self.out(h.mean(1)).unflatten(0, lead)


class RecallDecoder(nn.Module):
    """[S q, cue, L(q)] -> mu [cells, cell_dim], log_sigma [cells]."""

    def __init__(self, in_dim, cells, cell_dim, hidden=1024, log_sigma_range=(-3.0, 2.0)):
        super().__init__()
        self.cells, self.cell_dim, self.range = cells, cell_dim, log_sigma_range
        self.net = MLP(in_dim, hidden, cells * (cell_dim + 1))

    def forward(self, m):
        o = self.net(m).unflatten(-1, (self.cells, self.cell_dim + 1))
        return o[..., :-1], o[..., -1].clamp(*self.range)


def gaussian_nll(z, mu, log_sigma):
    """Per query: sum over cells of |z_c - mu_c|^2 / (2 sigma_c^2) + D * log sigma_c."""
    d = z.shape[-1]
    return ((z - mu).float().square().sum(-1) / (2 * torch.exp(2 * log_sigma.float())) + d * log_sigma.float()).sum(-1)


class WCEM(nn.Module):
    def __init__(self, cfg: Config = Config()):
        super().__init__()
        self.cfg = c = cfg
        self.tokenizer = SlotTokenizer(c.in_dim, c.n_tokens, c.slots, c.slot_dim)
        self.slot_head = CellDecoder(c.slot_dim, c.cells, c.cell_dim)
        self.window = LocalWindow(c.slot_dim, c.dim, c.window)
        self.w_k = nn.Linear(c.dim, 2 * c.dk)
        self.w_v = nn.Linear(c.dim, 4 * c.dk)
        self.w_cue = nn.Linear(c.dim, c.cue_dim)
        self.e_c = CellEncoder(c.cell_dim, c.cells, c.slot_dim, 2 * c.dk)
        self.w_cue_comp = CellEncoder(c.cell_dim, c.cells, c.slot_dim, c.cue_dim)
        self.lam_emb = nn.Sequential(nn.Linear(1, 16), nn.GELU(), nn.Linear(16, 16))
        self.decay = Decay(c.dim, 4, c.alpha0)
        self.memory = EpisodicMemory(2 * c.dim + 4 + 16, heads=4, dk=c.dk, dv=c.dk, sigma_q=c.sigma_q,
                                    chunk=c.chunk, gate_bias=c.gate_bias)
        self.slow = SlowMemory(4, c.dk, c.dk, enabled=c.slow_enabled)
        read_dim = 2 * c.dk + c.cue_dim + 2 * c.dk
        self.dec_seq = RecallDecoder(read_dim, c.cells, c.cell_dim, log_sigma_range=c.log_sigma_range)
        self.dec_comp = RecallDecoder(read_dim, c.cells, c.cell_dim, log_sigma_range=c.log_sigma_range)

    def _read_slow(self, k):
        if self.cfg.slow_enabled and self.cfg.slow_read:
            return self.slow(k)
        return torch.zeros_like(k[..., : self.cfg.dk])

    # ---- per-step encodings ----
    def encode(self, tokens, surprise):
        slots = self.tokenizer(tokens)
        return slots, self.window(slots, surprise)

    def seq_key(self, x_lag):
        return F.normalize(self.w_k(x_lag).unflatten(-1, (2, self.cfg.dk)), dim=-1)

    def comp_key(self, z, mask=None):
        return F.normalize(self.e_c(z, mask).unflatten(-1, (2, self.cfg.dk)), dim=-1)

    def random_view(self, shape, device):
        """True = hidden; exactly round(comp_frac * cells) cells visible."""
        cells = self.cfg.cells
        visible = max(1, round(self.cfg.comp_frac * cells))
        r = torch.rand(*shape, cells, device=device)
        return r.argsort(-1).argsort(-1) >= visible

    # ---- forward over a sequence ----
    def forward(self, tokens, z, surprise, log_lam, S0=None, g_force=None):
        """tokens [B, N, T, D], z [B, N, 16, 256], surprise [B, N, 4], log_lam [B].

        g_force [B, N, 2] overrides the pair gates (M8 write-trigger baselines).
        """
        B, N = z.shape[:2]
        slots, x = self.encode(tokens, surprise)
        x_lag = lagged(x)
        k = torch.cat([self.seq_key(x_lag), self.comp_key(z)], 2)
        v = F.normalize(self.w_v(x).unflatten(-1, (4, self.cfg.dk)), dim=-1)
        if self.cfg.precision_fusion and self.cfg.slow_enabled:
            Lk, var_L = self.slow.forward_with_variance(k)
        else:
            Lk = self.slow(k)
            var_L = None
        lam = self.lam_emb(log_lam[:, None])[:, None].expand(-1, N, -1)
        feats = torch.cat([x, x_lag, surprise, lam], -1)
        mem = self.memory(
            k, v, Lk, feats, self.decay(x), S0=S0, g_force=g_force,
            var_L=var_L, precision_fusion=self.cfg.precision_fusion and self.cfg.slow_enabled,
            kalman_gate=self.cfg.kalman_gate, sigma_obs=self.cfg.sigma_obs,
            decay_on_write_only=self.cfg.decay_on_write_only,
            use_checkpoint=self.cfg.checkpoint,
        )
        return dict(slots=slots, x=x, x_lag=x_lag, k=k, v=v, **mem)

    # ---- reads ----
    def read_seq(self, S, x_lag_q):
        """S [B, 4, dv, dk], x_lag_q [B, Q, dim] (x_{tau-2}) -> mu, log_sigma for z_tau."""
        q = self.seq_key(x_lag_q)
        m = self.memory.read(S[:, :2], q.transpose(0, 1)).transpose(0, 1)
        Lq = self._read_slow(torch.cat([q, q], 2))[:, :, :2]
        return self.dec_seq(torch.cat([m.flatten(-2), self.w_cue(x_lag_q), Lq.flatten(-2)], -1))

    def read_comp(self, S, z_q, mask):
        """Partial view (mask True = hidden) of z_tau -> mu, log_sigma for the full z_tau, and the query keys."""
        q = self.comp_key(z_q, mask)
        m = self.memory.read(S[:, 2:], q.transpose(0, 1)).transpose(0, 1)
        Lq = self._read_slow(torch.cat([q, q], 2))[:, :, 2:]
        cue = self.w_cue_comp(z_q, mask)
        mu, ls = self.dec_comp(torch.cat([m.flatten(-2), cue, Lq.flatten(-2)], -1))
        return mu, ls, q


def sample_queries(n, window, count, device, lag=LAG):
    """tau <= n - window: half uniform in [lag, n - window], half log-uniform in lag n - tau."""
    hi = n - window
    if hi < lag:
        return None
    half = count // 2
    uni = torch.randint(lag, hi + 1, (half,), device=device)
    lo_l, hi_l = math.log(window), math.log(max(n - lag, window) + 1)
    lags = torch.exp(torch.empty(count - half, device=device).uniform_(lo_l, hi_l)).floor().long()
    log_t = (n - lags).clamp(lag, hi)
    return torch.cat([uni, log_t])


def losses(model, out, z, log_lam, queries_per_read=32, S_override=None, rate_weight=1.0):
    """Coding-length objective for Stage A (L disabled, so the ensemble term is zero)."""
    c = model.cfg
    B, N = z.shape[:2]
    lam0 = log_lam.exp()
    S_chunks = out["S_chunks"] if S_override is None else S_override
    seq, comp, align = [], [], []
    for i in range(S_chunks.shape[0]):
        n = min((i + 1) * c.chunk, N) - 1
        tau = sample_queries(n, c.window, queries_per_read, z.device)
        if tau is None:
            continue
        tau = tau[None].expand(B, -1)
        S = S_chunks[i]
        x_q = out["x_lag"].gather(1, tau[..., None].expand(-1, -1, out["x_lag"].shape[-1]))
        z_q = z.gather(1, tau[..., None, None].expand(-1, -1, *z.shape[2:]))
        mu, ls = model.read_seq(S, x_q)
        seq.append(gaussian_nll(z_q, mu, ls))
        mask = model.random_view((B, tau.shape[1]), z.device)
        mu, ls, q = model.read_comp(S, z_q, mask)
        comp.append(gaussian_nll(z_q, mu, ls))
        k_full = out["k"][:, :, 2:].gather(1, tau[..., None, None].expand(-1, -1, 2, c.dk))
        align.append(1 - (q * k_full.detach()).sum(-1).mean(-1))
    seq, comp, align = torch.cat(seq, 1), torch.cat(comp, 1), torch.cat(align, 1)
    dims = c.cells * c.cell_dim
    writes = out["p_open"].sum(-1).mean(1)
    size = out["delta"].abs().sum((-1, -2)).mean(1)
    rate = (lam0 * writes + c.lam1_ratio * lam0 * size).mean()
    slot_loss = F.mse_loss(model.slot_head(out["slots"]).float(), z.float())
    total = (seq.mean() + comp.mean()) / dims + rate_weight * rate + slot_loss + c.key_align * align.mean()
    return total, dict(
        seq_nll_per_dim=seq.mean().item() / dims, comp_nll_per_dim=comp.mean().item() / dims,
        writes_per_step=writes.mean().item(), size_per_step=size.mean().item(),
        slot_mse=slot_loss.item(), key_align=align.mean().item(), rate=rate.item(),
        rate_weight=float(rate_weight), total=total.item(),
    )
