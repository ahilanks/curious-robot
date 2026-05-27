"""State encoder + JEPA world model (README §State encoder, §Dynamics).

State encoder (z_t in R^256):

    z_t = LN[ MLP( MLP(ViT(o_t)_cls)  ||  MLP(symlog(q_t, qdot_t, u^app_{t-1})) ) ]
                    (-> 192)                    (-> 64)         (joint fusion -> 256)

The visual branch is a from-scratch ViT-tiny (hidden 192, patch 14, 224x224) whose
CLS token is projected to 192 by an MLP head; the proprio branch (`ProprioEncoder`)
maps symlog(q, qdot, u_prev) to 64. The two are concatenated (192+64) and fused by
a joint MLP, then LayerNorm'd, giving the 256-d latent.

WorldModel is a `lewm.jepa.JEPA` whose encoder is this combined StateEncoder and
whose predictor/action_encoder/pred_proj come from `lewm.module`. It overrides
`encode` to read image+proprio; `predict` (predictor + pred_proj) is inherited.
"""
from __future__ import annotations

import torch
from torch import nn
from transformers import ViTModel, ViTConfig

from lewm.jepa import JEPA
from lewm.module import ARPredictor, Embedder, MLP
from model.proprio import ProprioEncoder


def build_vit_tiny(image_size: int = 224, patch_size: int = 14) -> ViTModel:
    """From-scratch (random-init) ViT-tiny; CLS token is the 192-d visual feature."""
    cfg = ViTConfig(
        hidden_size=192, num_hidden_layers=12, num_attention_heads=3,
        intermediate_size=768, image_size=image_size, patch_size=patch_size,
        num_channels=3, qkv_bias=True,
    )
    return ViTModel(cfg, add_pooling_layer=False)


class StateEncoder(nn.Module):
    def __init__(self, n_dof: int = 6, vis_dim: int = 192, prop_dim: int = 64,
                 image_size: int = 224, patch_size: int = 14):
        super().__init__()
        self.vit = build_vit_tiny(image_size, patch_size)
        cls_dim = self.vit.config.hidden_size  # 192
        self.visual_head = MLP(cls_dim, 4 * cls_dim, vis_dim)   # MLP(ViT_cls) -> 192
        self.proprio = ProprioEncoder(n_dof, out_dim=prop_dim)  # MLP(symlog(.)) -> 64
        self.out_dim = vis_dim + prop_dim                       # 256
        # Joint fusion MLP over the concatenated visual+proprio features (README's outer MLP).
        self.fuse = MLP(self.out_dim, 4 * self.out_dim, self.out_dim)
        self.norm = nn.LayerNorm(self.out_dim)

    def forward(self, image_norm: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        """image_norm: (B,3,H,W) normalized; proprio: (B,3*n_dof) -> z: (B, 256)."""
        cls = self.vit(image_norm, interpolate_pos_encoding=True).last_hidden_state[:, 0]
        v = self.visual_head(cls)
        p = self.proprio(proprio)
        return self.norm(self.fuse(torch.cat([v, p], dim=-1)))


class WorldModel(JEPA):
    """JEPA world model over the combined (image+proprio) latent."""

    def __init__(
        self,
        n_dof: int = 6,
        action_block: int = 5,
        z_dim: int = 256,
        vis_dim: int = 192,
        prop_dim: int = 64,
        history_size: int = 3,          # H_bwd
        depth: int = 6,
        heads: int = 8,
        dim_head: int = 32,
        mlp_dim: int = 1024,
        dropout: float = 0.1,
        image_size: int = 224,
        patch_size: int = 14,
    ):
        encoder = StateEncoder(n_dof, vis_dim, prop_dim, image_size, patch_size)
        predictor = ARPredictor(
            num_frames=history_size, input_dim=z_dim, hidden_dim=z_dim, output_dim=z_dim,
            depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, dropout=dropout,
        )
        action_encoder = Embedder(input_dim=n_dof * action_block, emb_dim=z_dim)
        pred_proj = MLP(z_dim, 2048, z_dim)
        super().__init__(encoder=encoder, predictor=predictor,
                         action_encoder=action_encoder, pred_proj=pred_proj)
        self.z_dim = z_dim
        self.history_size = history_size

    def encode(self, image_norm: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Combined-observation encode -> z (B, z_dim). Replaces JEPA's ViT-only encode."""
        return self.encoder(image_norm, proprio)
