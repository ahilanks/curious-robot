"""JEPA with a TOUCH channel fused into the encoder (curious-robot experiment, 2026-09-18).

LeWM's latent is projector(ViT CLS): pixels only. Here a small Embedder (the same module as the action
encoder) maps the dataset's touch signal (default: OGBench Cube's `proprio_gripper_contact`, a [0,1]
contact magnitude on the gripper, z-scored by the trainer's column normalizer) to embed_dim, and the
fused pre-projector feature is

    fused = cls + touch_scale * touch_encoder(touch)           ->  emb = projector(fused)

touch_scale is the sweep knob (0 = exactly LeWM). The touch encoder is learned, so touch_scale sets the
branch's initial share of the fused feature and its effective learning rate rather than a hard cap; the
sweep asks whether ANY fusion helps and how sensitive the outcome is to that prior. Everything else
(predictor, action encoder, projector, pred_proj, losses, rollout arithmetic) is JEPA's, untouched.

Inference helpers (rollout / get_cost) inherit JEPA's and work as long as the info dict carries the
touch key for the encoded frames (the imagined future needs none: touch is an encoder input only).
"""
from __future__ import annotations

import torch
from einops import rearrange

from jepa import JEPA


class JEPATouch(JEPA):
    def __init__(self, encoder, predictor, action_encoder, touch_encoder, touch_scale: float = 1.0,
                 touch_key: str = "proprio_gripper_contact", projector=None, pred_proj=None):
        super().__init__(encoder, predictor, action_encoder, projector, pred_proj)
        self.touch_encoder = touch_encoder
        self.touch_scale = float(touch_scale)
        self.touch_key = touch_key

    def encode(self, info):
        pixels = info["pixels"].float()
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...")
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        cls = output.last_hidden_state[:, 0]                                   # (b*t, D)
        touch = torch.nan_to_num(info[self.touch_key].float(), 0.0)           # (b, t, k) z-scored
        if touch.dim() == 2:                                                   # (b, t) -> (b, t, 1)
            touch = touch.unsqueeze(-1)
        t_emb = self.touch_encoder(touch)                                      # (b, t, D)
        fused = cls + self.touch_scale * rearrange(t_emb, "b t d -> (b t) d")
        emb = self.projector(fused)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)
        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])
        return info
