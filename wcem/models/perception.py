"""Slot tokenizer, slot reconstruction head, local window, and step token x_n (canvas parts T and W)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from wcem.models.blocks import AttnPool, CrossBlock


class SlotTokenizer(nn.Module):
    """[..., T, D_in] step tokens -> [..., K, d] slots (Perceiver-style cross-attention)."""

    def __init__(self, in_dim=1024, n_tokens=144, slots=8, dim=256, layers=2, heads=4):
        super().__init__()
        self.proj = nn.Linear(in_dim, dim)
        self.pos = nn.Parameter(torch.randn(1, n_tokens, dim) * 0.02)
        self.queries = nn.Parameter(torch.randn(1, slots, dim) * 0.02)
        self.blocks = nn.ModuleList(CrossBlock(dim, heads) for _ in range(layers))

    def forward(self, tokens):
        lead = tokens.shape[:-2]
        t = self.proj(tokens.flatten(0, -3)) + self.pos
        s = self.queries.expand(t.shape[0], -1, -1)
        for blk in self.blocks:
            s = blk(s, t)
        return s.unflatten(0, lead)


class CellDecoder(nn.Module):
    """[..., M, d] tokens -> [..., cells, out] via learned cell queries."""

    def __init__(self, dim, cells=16, out=256, heads=4):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, cells, dim) * 0.02)
        self.block = CrossBlock(dim, heads)
        self.out = nn.Linear(dim, out)

    def forward(self, x):
        lead = x.shape[:-2]
        x = x.flatten(0, -3)
        q = self.block(self.queries.expand(x.shape[0], -1, -1), x)
        return self.out(q).unflatten(0, lead)


class LocalWindow(nn.Module):
    """Block-causal attention: the slots of step n attend to the slots of steps n-W+1 .. n (previous layer)."""

    def __init__(self, slot_dim=256, dim=512, window=8, layers=2, heads=8, surprise_dim=4):
        super().__init__()
        self.window = window
        self.inp = nn.Linear(slot_dim, dim)
        self.age = nn.Parameter(torch.randn(window, 1, dim) * 0.02)
        self.blocks = nn.ModuleList(CrossBlock(dim, heads) for _ in range(layers))
        self.pool = AttnPool(dim)
        self.surprise = nn.Linear(surprise_dim, dim)
        self.out = nn.Linear(dim, dim)

    def _unfold(self, h):
        """h [B, N, K, d] -> windows [B, N, W, K, d] and padding mask [N, W] (True = before the video start)."""
        B, N, K, d = h.shape
        pad = F.pad(h, (0, 0, 0, 0, self.window - 1, 0))
        win = pad.unfold(1, self.window, 1).permute(0, 1, 4, 2, 3)
        offsets = torch.arange(self.window, device=h.device) - (self.window - 1)
        steps = torch.arange(N, device=h.device)[:, None] + offsets
        return win, steps < 0

    def forward(self, slots, surprise):
        """slots [B, N, K, slot_dim], surprise [B, N, 4] (fd-based) -> x [B, N, dim]."""
        B, N, K, _ = slots.shape
        h = self.inp(slots)
        for blk in self.blocks:
            win, pad = self._unfold(h)
            ctx = (win + self.age.flip(0)).flatten(2, 3)
            mask = pad[None, :, :, None].expand(B, N, self.window, K).flatten(2, 3)
            h = blk(h.flatten(0, 1), ctx.flatten(0, 1), key_padding_mask=mask.flatten(0, 1)).unflatten(0, (B, N))
        x = self.pool(h.flatten(0, 1)).unflatten(0, (B, N))
        return self.out(x + self.surprise(surprise))
