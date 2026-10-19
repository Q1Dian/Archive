"""E2: λ sweep — trained selective writes vs uniform at matched write count (Gate 1)."""
import torch

from wcem.eval.e1 import uniform_writes
from wcem.models.wcem import gaussian_nll, sample_queries


@torch.no_grad()
def seq_nll_per_dim(model, out, z, S_chunks, queries_per_read=32):
    c = model.cfg
    B, N = z.shape[:2]
    dims = c.cells * c.cell_dim
    seq = []
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
    if not seq:
        return float("nan")
    return torch.cat(seq, 1).mean().item() / dims


@torch.no_grad()
def sweep_lam(model, tokens, z, surprise, log_lams, queries_per_read=32):
    """Per λ: trained gate vs uniform g_force at the same expected write count."""
    model.eval()
    B, N = z.shape[:2]
    rows = []
    for ll in log_lams:
        lam = torch.full((B,), float(ll), device=z.device)
        out = model(tokens, z, surprise, lam)
        wps = out["p_open"].sum(-1).mean().item()
        sps = out["delta"].abs().sum((-1, -2)).mean().item()
        trained_nll = seq_nll_per_dim(model, out, z, out["S_chunks"], queries_per_read)
        rate = max(wps / 2.0, 1e-4)
        mask = torch.stack([uniform_writes(N, rate) for _ in range(B)]).to(z.device)
        g = mask.float()[..., None].expand(-1, -1, 2)
        out_u = model(tokens, z, surprise, lam, g_force=g)
        wps_u = out_u["p_open"].sum(-1).mean().item()
        sps_u = out_u["delta"].abs().sum((-1, -2)).mean().item()
        uniform_nll = seq_nll_per_dim(model, out_u, z, out_u["S_chunks"], queries_per_read)
        rows.append(dict(
            log_lam=float(ll),
            lam=float(torch.exp(lam[0]).item()),
            writes_per_step=wps,
            size_per_step=sps,
            bytes_per_step=wps * sps,
            uniform_writes_per_step=wps_u,
            uniform_size_per_step=sps_u,
            uniform_bytes_per_step=wps_u * sps_u,
            trained_seq_nll_per_dim=trained_nll,
            uniform_seq_nll_per_dim=uniform_nll,
            beats_uniform=trained_nll < uniform_nll,
            delta_nll=uniform_nll - trained_nll,
        ))
    return rows
