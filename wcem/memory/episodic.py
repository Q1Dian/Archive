"""Episodic state S: gated delta rule on the residual (canvas section 3.2).

    v'_n = v_n - L(k_n)
    r_n  = v'_n - alpha_n S_{n-1} k_n
    rho  = |r_n| / (|v'_n| + eps)
    g_n  = hard-concrete(MLP_g([feats_n, sg(|v'|, |r|, rho)]))     tied within a head pair
    beta = g_n * sigmoid(MLP_beta(...))                              per head
    delta = (g_n * beta_n) * r_n ;  delta~ = delta + g_n * N(0, sigma_q^2)   noise only on delta, after g
    S_n  = alpha_n S_{n-1} + delta~ k_n^T

Explicit recurrence, checkpointed in chunks so backprop stores one state per chunk.
"""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from wcem.memory.familiarity import FamiliarityState, kalman_gain
from wcem.memory.gates import hard_concrete, prob_open
from wcem.memory.precision import fuse_explained

EPS = 1e-6


class PairGate(nn.Module):
    def __init__(self, feat_dim, hidden, gate_bias):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feat_dim + 6, hidden), nn.GELU(), nn.Linear(hidden, 3))
        nn.init.zeros_(self.net[-1].weight)
        with torch.no_grad():
            self.net[-1].bias.copy_(torch.tensor([gate_bias, 0.0, 0.0]))

    def forward(self, feats, stats):
        out = self.net(torch.cat([feats, stats], -1))
        return out[..., 0], out[..., 1:]


class EpisodicMemory(nn.Module):
    def __init__(self, feat_dim, heads=4, dk=128, dv=128, hidden=256, sigma_q=0.01, chunk=64, gate_bias=-1.0):
        super().__init__()
        assert heads % 2 == 0
        self.heads, self.pairs, self.dk, self.dv = heads, heads // 2, dk, dv
        self.sigma_q, self.chunk = sigma_q, chunk
        self.gates = nn.ModuleList(PairGate(feat_dim, hidden, gate_bias) for _ in range(self.pairs))

    def init_state(self, batch, device):
        return torch.zeros(batch, self.heads, self.dv, self.dk, device=device)

    @staticmethod
    def read(S, q):
        """S [B, H, dv, dk], q [..., B, H, dk] -> [..., B, H, dv]."""
        return torch.einsum("bhvk,...bhk->...bhv", S, q)

    def step(self, S, k, v, Lk, feats, alpha, g_force=None, beta_force=None,
             var_L=None, precision_fusion=False, kalman_gate=False, sigma_obs=1.0, fam=None,
             decay_on_write_only=False):
        pred = alpha[..., None] * torch.einsum("bhvk,bhk->bhv", S, k)
        if precision_fusion and var_L is not None:
            vp = fuse_explained(v, Lk, var_L, pred, sigma_S=sigma_obs)
        else:
            vp = v - Lk
        r = vp - pred
        nvp, nr = vp.norm(dim=-1), r.norm(dim=-1)
        rho = nr / (nvp + EPS)
        stats = torch.stack([torch.log(nvp + EPS), torch.log(nr + EPS), rho], -1).detach()

        g, p, beta = [], [], []
        for i, gate in enumerate(self.gates):
            h = slice(2 * i, 2 * i + 2)
            log_alpha, b = gate(feats, stats[:, h].flatten(1))
            g.append(hard_concrete(log_alpha, self.training))
            p.append(prob_open(log_alpha))
            beta.append(torch.sigmoid(b))
        g, p, beta = torch.stack(g, 1), torch.stack(p, 1), torch.cat(beta, 1)
        if kalman_gate and fam is not None:
            fam_score = fam.score(k)
            kg = kalman_gain(fam_score, sigma_obs=sigma_obs)
            g = 0.5 * (kg[:, 0::2] + kg[:, 1::2])
            p = g
        if g_force is not None:
            g = g_force
        if beta_force is not None:
            beta = beta_force
        g_heads = g.repeat_interleave(2, dim=1)
        if kalman_gate and fam is not None:
            fam.update(k, g, alpha=alpha.mean().item())
        delta = (g_heads * beta)[..., None] * r
        write = delta
        if self.training and self.sigma_q > 0:
            write = write + g_heads[..., None] * self.sigma_q * torch.randn_like(delta)
        if decay_on_write_only:
            open_h = g_heads.clamp(0.0, 1.0)
            alpha_keep = 1.0 - open_h[..., None, None] * (1.0 - alpha[..., None, None])
            S = alpha_keep * S + torch.einsum("bhv,bhk->bhvk", write, k)
        else:
            S = alpha[..., None, None] * S + torch.einsum("bhv,bhk->bhvk", write, k)
        return S, delta, g, p, rho

    def _chunk(self, S, k, v, Lk, feats, alpha, g_force, beta_force, var_L, precision_fusion,
               kalman_gate, sigma_obs, fam, decay_on_write_only):
        out = []
        for t in range(k.shape[1]):
            vl = var_L[:, t] if var_L is not None else None
            S, *rest = self.step(
                S, k[:, t], v[:, t], Lk[:, t], feats[:, t], alpha[:, t],
                None if g_force is None else g_force[:, t],
                None if beta_force is None else beta_force[:, t],
                var_L=vl, precision_fusion=precision_fusion, kalman_gate=kalman_gate,
                sigma_obs=sigma_obs, fam=fam, decay_on_write_only=decay_on_write_only,
            )
            out.append(rest)
        delta, g, p, rho = (torch.stack(x, 1) for x in zip(*out))
        return S, delta, g, p, rho

    def forward(self, k, v, Lk, feats, alpha, S0=None, g_force=None, beta_force=None,
                var_L=None, precision_fusion=False, kalman_gate=False, sigma_obs=1.0,
                decay_on_write_only=False, use_checkpoint=True):
        """All inputs are [B, N, ...]. Returns states after each chunk and per-step write info.

        S_chunks[c] is the state after step (c + 1) * chunk - 1 (the last entry may be a partial chunk).
        """
        B, N = k.shape[:2]
        S = self.init_state(B, k.device) if S0 is None else S0
        fam = FamiliarityState(B, self.heads, self.dk, k.device, k.dtype) if kalman_gate else None
        states, deltas, gs, ps, rhos = [], [], [], [], []
        for a in range(0, N, self.chunk):
            sl = slice(a, a + self.chunk)
            args = (S, k[:, sl], v[:, sl], Lk[:, sl], feats[:, sl], alpha[:, sl],
                    None if g_force is None else g_force[:, sl],
                    None if beta_force is None else beta_force[:, sl],
                    var_L[:, sl] if var_L is not None else None,
                    precision_fusion, kalman_gate, sigma_obs, fam, decay_on_write_only)
            # Checkpointing breaks when gate weights start receiving gradients mid-run.
            ckpt = (use_checkpoint and torch.is_grad_enabled()
                    and not kalman_gate
                    and not any(p.requires_grad for p in self.gates.parameters()))
            if ckpt:
                S, delta, g, p, rho = checkpoint(self._chunk, *args, use_reentrant=False)
            else:
                S, delta, g, p, rho = self._chunk(*args)
            states.append(S)
            deltas.append(delta)
            gs.append(g)
            ps.append(p)
            rhos.append(rho)
        return dict(
            S_chunks=torch.stack(states, 0),
            delta=torch.cat(deltas, 1),
            g=torch.cat(gs, 1),
            p_open=torch.cat(ps, 1),
            rho=torch.cat(rhos, 1),
        )
