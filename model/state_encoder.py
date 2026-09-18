"""State encoder + JEPA world model (README §State encoder, §Dynamics).

State encoder (z_t in R^256):

    z_t = MLP( MLP(ViT(o_t)_cls)  ||  MLP(symlog(q_t, qdot_t, u^app_{t-1})) )
                 (-> 192)                    (-> 64)         (joint fusion -> 256)

The visual branch is a from-scratch ViT-tiny (hidden 192, patch 14, 224x224) whose
CLS token is projected to 192 by an MLP head; the proprio branch (`ProprioEncoder`)
maps symlog(q, qdot, u_prev) to 64. The two are concatenated (192+64) and fused by
a joint MLP, giving the 256-d latent. There is no final per-sample LayerNorm: it
would pin every z to a fixed-radius sphere, which fights SIGReg's isotropic-Gaussian
objective (LeWM keeps only BatchNorm inside the projector MLP, no output norm).

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


# Predictor sizing. LeWM (lewm/config/train/model/lewm.yaml) builds the ARPredictor with
# depth=6, heads=16, dim_head=64, mlp_dim=2048. The defaults below match that. Checkpoints
# trained before --wm-pred-* existed used the SMALLER predictor (heads=8, dim_head=32,
# mlp_dim=1024); `pred_dims_from_args` reconstructs THAT from a checkpoint's saved args so old
# ckpts still load. Only heads/dim_head/mlp_dim/depth change the predictor's weight shapes;
# z_dim is set by the encoder (192 pixels-only / 256 with proprio), independent of these.
_PRED_DIMS_OLD = dict(depth=6, heads=8, dim_head=32, mlp_dim=1024)


def pred_dims_from_args(a) -> dict:
    """Resolve predictor dims from a saved args dict (`ck['args']`) or an argparse Namespace.
    Keys absent from the source fall back to the OLD small predictor — i.e. a checkpoint
    that predates `--wm-pred-*` rebuilds the exact predictor it was trained with."""
    get = a.get if isinstance(a, dict) else (lambda k, d=None: getattr(a, k, d))
    return dict(
        depth=int(get("wm_pred_depth", _PRED_DIMS_OLD["depth"]) or _PRED_DIMS_OLD["depth"]),
        heads=int(get("wm_pred_heads", _PRED_DIMS_OLD["heads"]) or _PRED_DIMS_OLD["heads"]),
        dim_head=int(get("wm_pred_dim_head", _PRED_DIMS_OLD["dim_head"]) or _PRED_DIMS_OLD["dim_head"]),
        mlp_dim=int(get("wm_pred_mlp_dim", _PRED_DIMS_OLD["mlp_dim"]) or _PRED_DIMS_OLD["mlp_dim"]),
    )


TOUCH_MODES = ("none", "effort")


def touch_kwargs_from_args(a) -> dict:
    """Touch-branch kwargs from a saved args dict / Namespace (absent keys -> no touch branch, so every
    checkpoint that predates --touch-input rebuilds exactly as before)."""
    get = a.get if isinstance(a, dict) else (lambda k, d=None: getattr(a, k, d))
    return dict(touch_input=str(get("touch_input", "none") or "none"),
                touch_scale=float(get("touch_scale", 1.0) if get("touch_scale", None) is not None else 1.0))


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
                 image_size: int = 224, patch_size: int = 14, use_proprio: bool = True,
                 touch_input: str = "none", touch_scale: float = 1.0, touch_norm=None):
        super().__init__()
        self.use_proprio = use_proprio
        self.n_dof = n_dof
        self.vit = build_vit_tiny(image_size, patch_size)
        cls_dim = self.vit.config.hidden_size  # 192
        # TOUCH BRANCH (2026-09-18, ported from lewm/jepa_touch.py after the LeWM-Cube sweep): a proprioceptive
        # "touch" signal is embedded by LeWM's Embedder (the action-encoder module: Conv1d k=1 + MLP) and ADDED to
        # the ViT CLS before the projector, fused = cls + touch_scale * Embedder(touch). touch_input='effort' = the
        # 6 applied joint efforts u^app (proprio[2n:3n], the servo's actuator_force; Present_Current on hardware)
        # divided by tau_max -> [-1, 1]. Pixels-only path only (the proprio path already fuses everything).
        if touch_input not in TOUCH_MODES:
            raise ValueError(f"touch_input must be one of {TOUCH_MODES}, got {touch_input!r}")
        if touch_input != "none" and use_proprio:
            raise ValueError("--touch-input needs the pixels-only encoder (--no-proprio)")
        self.touch_input, self.touch_scale = touch_input, float(touch_scale)
        if touch_input == "effort":
            self.touch_dim = n_dof
            self.touch_encoder = Embedder(input_dim=self.touch_dim, smoothed_dim=8, emb_dim=cls_dim)
            norm = torch.as_tensor(touch_norm, dtype=torch.float32) if touch_norm is not None else torch.full((n_dof,), 3.35)
            self.register_buffer("touch_norm", norm.reshape(-1).clone())    # tau_max per joint; travels with the ckpt
        if use_proprio:
            self.visual_head = MLP(cls_dim, 4 * cls_dim, vis_dim)   # MLP(ViT_cls) -> 192
            self.proprio = ProprioEncoder(n_dof, out_dim=prop_dim)  # MLP(symlog(.)) -> 64
            self.out_dim = vis_dim + prop_dim                       # 256
            # Joint fusion MLP over the concatenated visual+proprio features (README's outer MLP).
            self.fuse = MLP(self.out_dim, 4 * self.out_dim, self.out_dim)
        else:
            # LeWM-faithful PIXELS-ONLY state: z = projector(ViT_cls), replicating lewm.jepa.JEPA.encode
            # with LeWM's EXACT projector (lewm.yaml): MLP(hidden=2048, norm_fn=BatchNorm1d). The
            # BatchNorm1d does per-dim whitening ACROSS the batch -- the SSL anti-collapse mechanism
            # SIGReg's isotropy target relies on. (The MLP default norm_fn is LayerNorm = per-SAMPLE,
            # which gives no cross-batch anti-collapse -- NOT what LeWM uses.) D = vis_dim = 192.
            self.visual_head = MLP(cls_dim, 2048, vis_dim, norm_fn=nn.BatchNorm1d)
            self.out_dim = vis_dim                                  # 192

    def forward(self, image_norm: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        """image_norm: (B,3,H,W) normalized; proprio: (B,3*n_dof) -> z: (B, out_dim).
        proprio is ignored when use_proprio=False (kept in the signature so call sites are uniform)."""
        cls = self.vit(image_norm, interpolate_pos_encoding=True).last_hidden_state[:, 0]
        if self.touch_input != "none":
            t = self.touch_features(proprio)                                        # (B, touch_dim)
            cls = cls + self.touch_scale * self.touch_encoder(t.unsqueeze(1))[:, 0]  # Embedder wants (B, T, D)
        v = self.visual_head(cls)
        if not self.use_proprio:
            return v                                               # pixels-only (192-d), proprio dropped
        p = self.proprio(proprio)
        return self.fuse(torch.cat([v, p], dim=-1))


    def touch_features(self, proprio: torch.Tensor) -> torch.Tensor:
        """proprio (B, 3*n_dof) = [q, qdot, u_app] -> the touch signal the branch embeds."""
        if self.touch_input == "effort":
            u = proprio.float()[..., 2 * self.n_dof:3 * self.n_dof]
            return (u / self.touch_norm).clamp(-2.0, 2.0)
        raise RuntimeError("no touch branch")


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
        depth: int = 6,                  # predictor dims default to LeWM (lewm.yaml); pre-2026-06-26
        heads: int = 16,                 # runs used heads=8/dim_head=32/mlp_dim=1024 (~half the
        dim_head: int = 64,              # capacity). pred_dims_from_args() rebuilds the old size
        mlp_dim: int = 2048,             # from a checkpoint's saved args.
        dropout: float = 0.1,
        image_size: int = 224,
        patch_size: int = 14,
        use_proprio: bool = True,
        touch_input: str = "none",
        touch_scale: float = 1.0,
        touch_norm=None,
    ):
        if not use_proprio:
            z_dim = vis_dim          # LeWM-faithful pixels-only: D = 192 (= LeWM embed_dim)
        encoder = StateEncoder(n_dof, vis_dim, prop_dim, image_size, patch_size,
                               use_proprio=use_proprio, touch_input=touch_input, touch_scale=touch_scale,
                               touch_norm=touch_norm)
        predictor = ARPredictor(
            num_frames=history_size, input_dim=z_dim, hidden_dim=z_dim, output_dim=z_dim,
            depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, dropout=dropout,
        )
        action_encoder = Embedder(input_dim=n_dof * action_block, emb_dim=z_dim)
        # LeWM applies BatchNorm1d in BOTH projector and pred_proj (lewm.yaml). Match it in the
        # pixels-only path; the proprio path keeps the MLP-default LayerNorm (unchanged baseline).
        pred_proj = (MLP(z_dim, 2048, z_dim, norm_fn=nn.BatchNorm1d) if not use_proprio
                     else MLP(z_dim, 2048, z_dim))
        super().__init__(encoder=encoder, predictor=predictor,
                         action_encoder=action_encoder, pred_proj=pred_proj)
        self.z_dim = z_dim
        self.history_size = history_size

    def encode(self, image_norm: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Combined-observation encode -> z (B, z_dim). Replaces JEPA's ViT-only encode."""
        return self.encoder(image_norm, proprio)
