import math

import torch

from wcem.models.wcem import WCEM, Config, losses, sample_queries

CFG = Config(in_dim=32, n_tokens=9, slots=4, slot_dim=32, dim=64, window=4, cells=16, cell_dim=8, dk=16,
             cue_dim=16, chunk=8)
B, N = 2, 24


def batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(B, N, CFG.n_tokens, CFG.in_dim, generator=g),
            torch.randn(B, N, CFG.cells, CFG.cell_dim, generator=g),
            torch.randn(B, N, 4, generator=g),
            torch.full((B,), math.log(0.01)))


def test_forward_shapes_and_loss_backward():
    torch.manual_seed(0)
    model = WCEM(CFG)
    tokens, z, s, lam = batch()
    out = model(tokens, z, s, lam)
    assert out["x"].shape == (B, N, CFG.dim)
    assert out["k"].shape == (B, N, 4, CFG.dk)
    assert out["S_chunks"].shape == (N // CFG.chunk, B, 4, CFG.dk, CFG.dk)
    total, logs = losses(model, out, z, lam, queries_per_read=6)
    assert torch.isfinite(total)
    total.backward()
    assert model.w_k.weight.grad.abs().sum() > 0
    assert model.e_c.inp.weight.grad.abs().sum() > 0
    assert model.memory.gates[0].net[-1].weight.grad.abs().sum() > 0


def test_surprise_at_n_can_use_step_n_but_not_later():
    """fd surprise at n compares tokens of n and n-1; later steps must not leak into x_lag / seq key."""
    torch.manual_seed(0)
    model = WCEM(CFG).eval()
    tokens, z, s, lam = batch()
    n = 15
    s2 = s.clone()
    s2[:, n + 1:] = torch.randn_like(s2[:, n + 1:])
    with torch.no_grad():
        a, b = model(tokens, z, s, lam), model(tokens, z, s2, lam)
    assert torch.allclose(a["x"][:, : n + 1], b["x"][:, : n + 1], atol=1e-5)
    assert torch.allclose(a["k"][:, n, :2], b["k"][:, n, :2], atol=1e-5)


def test_sequence_key_and_cue_do_not_see_step_n_or_later():
    # x_{n-2} drives the key and cue for step n; its clip ends when step n begins, so nothing from n-1 on may leak
    torch.manual_seed(0)
    model = WCEM(CFG).eval()
    tokens, z, s, lam = batch()
    n = 15
    t2, z2, s2 = tokens.clone(), z.clone(), s.clone()
    t2[:, n - 1:] = torch.randn_like(t2[:, n - 1:])
    z2[:, n - 1:] = torch.randn_like(z2[:, n - 1:])
    s2[:, n - 1:] = torch.randn_like(s2[:, n - 1:])
    with torch.no_grad():
        a, b = model(tokens, z, s, lam), model(t2, z2, s2, lam)
    assert torch.allclose(a["x_lag"][:, n], b["x_lag"][:, n], atol=1e-5)
    assert torch.allclose(a["k"][:, n, :2], b["k"][:, n, :2], atol=1e-5)
    assert torch.allclose(model.w_cue(a["x_lag"][:, n]), model.w_cue(b["x_lag"][:, n]), atol=1e-5)
    assert not torch.allclose(a["x"][:, n], b["x"][:, n])


def test_window_is_causal_and_bounded():
    torch.manual_seed(0)
    model = WCEM(CFG).eval()
    tokens, z, s, lam = batch()
    n = 12
    t2 = tokens.clone()
    t2[:, n + 1:] = torch.randn_like(t2[:, n + 1:])
    with torch.no_grad():
        xa, xb = model(tokens, z, s, lam)["x"], model(t2, z, s, lam)["x"]
    assert torch.allclose(xa[:, : n + 1], xb[:, : n + 1], atol=1e-5)


def test_sample_queries_bounds():
    for n in (10, 63, 511):
        tau = sample_queries(n, 8, 64, "cpu")
        assert tau.min() >= 2 and tau.max() <= n - 8
    assert sample_queries(5, 8, 4, "cpu") is None


def test_completion_view_has_fixed_visible_count():
    model = WCEM(CFG)
    m = model.random_view((3, 5), "cpu")
    assert (m.logical_not().sum(-1) == 4).all()
