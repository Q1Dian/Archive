"""Stage B: train slow memory L on pre-write residual |v - L(k)|."""
import torch
import torch.nn.functional as F

from wcem.models.wcem import lagged

SIM_THRESH = 0.85


def slow_residual_loss(model, tokens, z, surprise, dupsim, sim_thresh=SIM_THRESH, bootstrap_p=0.8):
    """Per-ensemble-member bootstrap masks on near-duplicate (dupsim) steps."""
    with torch.no_grad():
        _, x = model.encode(tokens, surprise)
        x_lag = lagged(x)
        k = torch.cat([model.seq_key(x_lag), model.comp_key(z)], 2)
        v = F.normalize(model.w_v(x).unflatten(-1, (4, model.cfg.dk)), dim=-1)
    preds, _ = model.slow.members(k)
    sim = dupsim[..., 0] if dupsim.dim() == 3 else dupsim
    repeat = sim >= sim_thresh
    losses, frac = [], []
    for e in range(preds.shape[0]):
        boot = torch.rand_like(repeat.float()) < bootstrap_p if bootstrap_p < 1.0 else torch.ones_like(repeat.float())
        m = (repeat & boot).float()
        vp = v - preds[e]
        per = vp.norm(dim=-1).square().mean(-1)
        losses.append((per * m).sum() / m.sum().clamp_min(1.0))
        frac.append(m.mean().item())
    total = sum(losses) / len(losses)
    return total, dict(
        slow_residual=total.item(),
        slow_spread=preds.var(0, unbiased=False).mean().item(),
        slow_repeat_frac=sum(frac) / len(frac),
    )
