"""Frozen V-JEPA 2.1 encoder.

Token order is (time, height, width). A clip of 16 frames at 384 px gives
8 x 24 x 24 tokens. The context-only call encodes the first 8 frames (temporal
tokens 0-3) on their own; with RoPE this equals the masked call that keeps
only those tokens (see tests/test_encoder.py).
"""
import sys

import torch
import torch.nn as nn

from wcem.paths import CHECKPOINTS, VJEPA2

if str(VJEPA2) not in sys.path:
    sys.path.insert(0, str(VJEPA2))

from src.hub.backbones import ARCH_NAME_MAP, _clean_backbone_key, _make_vjepa2_1_model  # noqa: E402

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)
CONTEXT_FRAMES = 8

SPECS = {
    "vjepa2_1_vit_large_384": dict(checkpoint_key="ema_encoder", predictor_depth=12, predictor_num_mask_tokens=8,
                                   n_output_distillation=1, teacher_embed_dim=1664),
    "vjepa2_1_vit_giant_384": dict(predictor_num_mask_tokens=8, n_output_distillation=4),
    "vjepa2_1_vit_gigantic_384": dict(predictor_num_mask_tokens=8, n_output_distillation=4),
}


def build_encoder(arch="vjepa2_1_vit_large_384", pretrained=True):
    spec = dict(SPECS[arch])
    key = spec.pop("checkpoint_key", "target_encoder")
    encoder, _ = _make_vjepa2_1_model(model_name=arch, img_size=384, return_all_tokens=True, pretrained=False, **spec)
    if pretrained:
        path = CHECKPOINTS / f"{ARCH_NAME_MAP[arch][1]}.pt"
        # mmap: the file also holds the ViT-G teacher and predictor (5 GB), only the encoder is read
        state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        encoder.load_state_dict(_clean_backbone_key(state[key]), strict=True)
        del state
    return encoder


class Encoder(nn.Module):
    def __init__(self, arch="vjepa2_1_vit_large_384", pretrained=True, hierarchical=False):
        super().__init__()
        self.net = build_encoder(arch, pretrained).eval().requires_grad_(False)
        self.net.return_hierarchical = hierarchical
        self.patch = self.net.patch_size
        self.tubelet = self.net.tubelet_size

    @property
    def dim(self):
        return self.net.embed_dim * (4 if self.net.return_hierarchical else 1)

    def _grid(self, x, out):
        b, _, t, h, w = x.shape
        return out.view(b, t // self.tubelet, h // self.patch, w // self.patch, -1)

    @torch.no_grad()
    def forward(self, x):
        """x: [B, 3, T, H, W] in [0, 1] -> [B, T/2, H/16, W/16, D]."""
        x = (x - MEAN.to(x)) / STD.to(x)
        return self._grid(x, self.net(x))

    def context(self, x):
        """Encode only the first 8 frames (temporal tokens 0-3) of each clip."""
        return self(x[:, :, :CONTEXT_FRAMES])

    @torch.no_grad()
    def masked(self, x, keep):
        """Reference path: full clip with only `keep` token indices visible. Returns [B, K, D]."""
        x = (x - MEAN.to(x)) / STD.to(x)
        return self.net(x, masks=keep)
