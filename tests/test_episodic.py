import torch
import torch.nn.functional as F

from wcem.memory.episodic import EpisodicMemory
from wcem.memory.gates import Decay, hard_concrete, prob_open

B, H, D, FEAT = 2, 4, 16, 8


def inputs(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    k = F.normalize(torch.randn(B, n, H, D, generator=g), dim=-1)
    v = F.normalize(torch.randn(B, n, H, D, generator=g), dim=-1)
    return k, v, torch.zeros_like(v), torch.randn(B, n, FEAT, generator=g), torch.ones(B, n, H)


def test_orthogonal_keys_recalled_exactly():
    n = D
    mem = EpisodicMemory(FEAT, heads=H, dk=D, dv=D, sigma_q=0.0, chunk=5).eval()
    _, v, Lk, feats, alpha = inputs(n)
    eye = torch.eye(D)
    k = eye[None, :, None, :].expand(B, n, H, D).contiguous()
    out = mem(k, v, Lk, feats, alpha, g_force=torch.ones(B, n, 2), beta_force=torch.ones(B, n, H))
    S = out["S_chunks"][-1]
    for i in range(n):
        assert torch.allclose(mem.read(S, k[:, i]), v[:, i], atol=1e-5)


def test_closed_gate_only_decays():
    n = 7
    mem = EpisodicMemory(FEAT, heads=H, dk=D, dv=D, sigma_q=0.1, chunk=3).train()
    k, v, Lk, feats, _ = inputs(n)
    alpha = torch.full((B, n, H), 0.9)
    S0 = torch.randn(B, H, D, D)
    out = mem(k, v, Lk, feats, alpha, S0=S0, g_force=torch.zeros(B, n, 2), use_checkpoint=False)
    assert torch.allclose(out["S_chunks"][-1], S0 * 0.9 ** n, atol=1e-6)
    assert out["delta"].abs().max() == 0


def test_decay_on_write_only_preserves_s_when_gate_closed():
    n = 7
    mem = EpisodicMemory(FEAT, heads=H, dk=D, dv=D, sigma_q=0.0, chunk=3).eval()
    k, v, Lk, feats, _ = inputs(n)
    alpha = torch.full((B, n, H), 0.5)
    S0 = torch.randn(B, H, D, D)
    out = mem(k, v, Lk, feats, alpha, S0=S0, g_force=torch.zeros(B, n, 2),
              decay_on_write_only=True, use_checkpoint=False)
    assert torch.allclose(out["S_chunks"][-1], S0, atol=1e-6)


def test_residual_write_is_zero_when_slow_memory_predicts_value():
    n = 3
    mem = EpisodicMemory(FEAT, heads=H, dk=D, dv=D, sigma_q=0.0).eval()
    k, v, _, feats, alpha = inputs(n)
    out = mem(k, v, v.clone(), feats, alpha, g_force=torch.ones(B, n, 2), beta_force=torch.ones(B, n, H))
    assert out["delta"].abs().max() < 1e-6


def test_checkpoint_matches_plain_backprop():
    n = 10
    k, v, Lk, feats, alpha = inputs(n)

    def run(use_ckpt):
        torch.manual_seed(0)
        mem = EpisodicMemory(FEAT, heads=H, dk=D, dv=D, sigma_q=0.05, chunk=4).train()
        kk = k.clone().requires_grad_(True)
        torch.manual_seed(1)
        out = mem(kk, v, Lk, feats, alpha, use_checkpoint=use_ckpt)
        loss = out["S_chunks"].square().sum() + out["delta"].abs().sum() + out["p_open"].sum()
        loss.backward()
        return kk.grad, [p.grad for p in mem.parameters()]

    ga, pa = run(True)
    gb, pb = run(False)
    assert torch.allclose(ga, gb, atol=1e-5)
    for a, b in zip(pa, pb):
        assert torch.allclose(a, b, atol=1e-5)


def test_gate_inputs_are_detached():
    # the gate must not send gradient into k through the residual statistics
    n = 4
    mem = EpisodicMemory(FEAT, heads=H, dk=D, dv=D, sigma_q=0.0).train()
    k, v, Lk, feats, alpha = inputs(n)
    k.requires_grad_(True)
    out = mem(k, v, Lk, feats, alpha, use_checkpoint=False)
    out["p_open"].sum().backward()
    assert k.grad is None or k.grad.abs().max() == 0


def test_hard_concrete():
    la = torch.full((100000,), 0.5)
    z = hard_concrete(la, training=True)
    assert ((z == 0) | (z == 1)).float().mean() > 0.3
    assert abs((z != 0).float().mean() - prob_open(la)[0]) < 0.01
    assert set(hard_concrete(torch.tensor([-1.0, 1.0]), training=False).tolist()) == {0.0, 1.0}


def test_decay_init():
    d = Decay(32, H, alpha0=0.9998)
    a = d(torch.randn(5, 32))
    assert torch.allclose(a, torch.full_like(a, 0.9998), atol=1e-6)
