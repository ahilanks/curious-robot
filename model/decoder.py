"""Post-hoc pixel decoder for the frozen JEPA latent — DIAGNOSTIC ONLY (LeWM-style).

JEPA never reconstructs pixels; that is the point of the architecture. But a small
decoder trained AFTER the fact on frozen (z -> frame) pairs answers two questions the
latent metrics cannot: what the CEM plan *imagines* (decode the predictor's one-step
output under the chosen action) and what information the encoder preserves vs discards
(decode(z_now) vs the real frame — whatever the decoder cannot recover from z, the
planner cannot see either). No gradients ever flow into the encoder or world model:
train_decoder.py encodes with the WM under no_grad and fits only this module.

Geometry: z (192, the pixels-only StateEncoder output) -> 7x7 map -> five nearest-
neighbor upsample+conv stages (7-14-28-56-112-224) -> sigmoid RGB in [0,1] at 224x224.
~3.3M params — trains in minutes on MPS from a few thousand frames.
"""
from __future__ import annotations

import torch
from torch import nn


class LatentDecoder(nn.Module):
    def __init__(self, z_dim: int = 192, base: int = 256, out_hw: int = 224):
        super().__init__()
        if out_hw != 224:
            raise ValueError("LatentDecoder is fixed at 224 (7 * 2^5); resize outside if needed")
        chs = [base, 192, 128, 96, 64, 32]
        self.z_dim = z_dim
        self.fc = nn.Linear(z_dim, chs[0] * 7 * 7)
        blocks: list[nn.Module] = []
        for cin, cout in zip(chs[:-1], chs[1:]):
            blocks += [nn.Upsample(scale_factor=2, mode="nearest"),
                       nn.Conv2d(cin, cout, 3, padding=1),
                       nn.GroupNorm(8, cout), nn.SiLU()]
        blocks.append(nn.Conv2d(chs[-1], 3, 3, padding=1))
        self.net = nn.Sequential(*blocks)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """(B, z_dim) -> (B, 3, 224, 224) in [0,1]."""
        x = self.fc(z).view(z.shape[0], -1, 7, 7)
        return torch.sigmoid(self.net(x))

    @torch.no_grad()
    def to_uint8_hwc(self, z: torch.Tensor):
        """(B, z_dim) -> (B, 224, 224, 3) uint8 numpy — dashboard-ready frames."""
        return (self(z).permute(0, 2, 3, 1).clamp(0, 1) * 255).byte().cpu().numpy()
