"""World-model backbone utilities: loading LeWM, image/action preprocessing,
latent rollouts and planning cost models (latent-L2 or learned critic).

Conventions (match stable-worldmodel + the LeWM training recipe):
  * images: uint8 HWC -> float, ImageNet-normalised, 224x224
  * actions: z-scored per primitive dim with a StandardScaler fitted on the
    dataset's action column (NaN rows dropped); one *block* = 5 consecutive
    primitive actions concatenated -> action_dim_block = 5 * |a|
  * a plan is (N=5 blocks, action_dim_block); rollouts go block by block
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn import preprocessing

# --------------------------------------------------------------------------
# checkpoint loading
# --------------------------------------------------------------------------

_OLD2NEW = [  # transformers>=5 renamed the HF ViT modules
    (r'encoder\.encoder\.layer\.(\d+)\.attention\.attention\.query', r'encoder.layers.\1.attention.q_proj'),
    (r'encoder\.encoder\.layer\.(\d+)\.attention\.attention\.key', r'encoder.layers.\1.attention.k_proj'),
    (r'encoder\.encoder\.layer\.(\d+)\.attention\.attention\.value', r'encoder.layers.\1.attention.v_proj'),
    (r'encoder\.encoder\.layer\.(\d+)\.attention\.output\.dense', r'encoder.layers.\1.attention.o_proj'),
    (r'encoder\.encoder\.layer\.(\d+)\.intermediate\.dense', r'encoder.layers.\1.mlp.fc1'),
    (r'encoder\.encoder\.layer\.(\d+)\.output\.dense', r'encoder.layers.\1.mlp.fc2'),
    (r'encoder\.encoder\.layer\.(\d+)\.', r'encoder.layers.\1.'),
]


def load_lewm(ckpt_dir: str | Path, device='cuda') -> nn.Module:
    """Load a LeWM checkpoint folder (config.json + weights.pt) in eval mode,
    frozen. Falls back to a key remap if the installed transformers renamed
    the ViT modules."""
    from hydra.utils import instantiate

    ckpt_dir = Path(ckpt_dir)
    cfg = json.loads((ckpt_dir / 'config.json').read_text())
    sd = torch.load(ckpt_dir / 'weights.pt', map_location='cpu')
    model = instantiate(cfg)
    try:
        model.load_state_dict(sd)
    except RuntimeError:
        remapped = {}
        for k, v in sd.items():
            nk = k
            for pat, rep in _OLD2NEW:
                nk = re.sub(pat, rep, nk)
            remapped[nk] = v
        model.load_state_dict(remapped)
    model = model.to(device).eval()
    model.requires_grad_(False)
    return model


# --------------------------------------------------------------------------
# preprocessing
# --------------------------------------------------------------------------

def image_transform(img_size: int = 224, dtype=torch.float32):
    import stable_pretraining as spt
    from torchvision.transforms import v2 as transforms

    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(dtype, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=img_size),
    ])


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def normalize_uint8_batch(x: torch.Tensor) -> torch.Tensor:
    """(B,H,W,3) or (B,3,H,W) uint8 -> (B,3,H,W) float ImageNet-normalised.
    Fast path for batched GPU encoding (no per-image PIL transforms)."""
    if x.shape[-1] == 3:
        x = x.permute(0, 3, 1, 2)
    x = x.float() / 255.0
    return (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)


def fit_action_scaler(action_col: np.ndarray) -> preprocessing.StandardScaler:
    """Same normaliser as swm's eval script / LeWM training (zscore, NaN rows dropped)."""
    scaler = preprocessing.StandardScaler()
    a = np.asarray(action_col, dtype=np.float64)
    a = a[~np.isnan(a).any(axis=1)]
    scaler.fit(a)
    return scaler


# --------------------------------------------------------------------------
# latent-space rollouts
# --------------------------------------------------------------------------

@torch.no_grad()
def encode_pixels(model, pixels: torch.Tensor) -> torch.Tensor:
    """pixels: (B,3,H,W) normalised float -> (B,D) latent (projector output)."""
    out = model.encoder(pixels.to(next(model.encoder.parameters()).dtype), interpolate_pos_encoding=True)
    return model.projector(out.last_hidden_state[:, 0])


def rollout_latent(model, z0: torch.Tensor, plan: torch.Tensor, history: int | None = None,
                   return_all: bool = False):
    """Autoregressive latent rollout from a single start latent.

    z0:   (B, D) start latent (already encoded, may be detached)
    plan: (B, N, A_block) z-scored action blocks (may require grad)
    returns ẑ_N (B, D)  [or the full list of N+1 latents if return_all]

    Mirrors LeWM.rollout with history_len=1: the predictor is a causal
    transformer over the last `history` (=num_frames=3) frames.
    """
    if history is None:
        history = getattr(model.predictor, 'num_frames', 3)
    act_emb = model.action_encoder(plan)  # (B, N, E)
    embs = [z0]
    N = plan.shape[1]
    for t in range(N):
        lo = max(0, t + 1 - history)
        ctx = torch.stack(embs[lo:], dim=1)          # (B, <=history, D)
        a_ctx = act_emb[:, lo:t + 1]                 # aligned action blocks
        embs.append(model.predict(ctx, a_ctx)[:, -1])
    return embs if return_all else embs[-1]


# --------------------------------------------------------------------------
# cost models for the hand-designed solvers (swm Costable protocol)
# --------------------------------------------------------------------------

class LatentCostModel(nn.Module):
    """`get_cost(info_dict, candidates)` for swm's CEM / MPPI / Gradient solvers.

    objective='latent': ||ẑ_N - z_g||²₂   (LeWM's own criterion)
    objective='value' : V_ψ(ẑ_N, z_g)     (learned quasimetric critic)
    info_dict follows swm: pixels (B,S,T,3,H,W) normalised, goal (B,S,T,3,H,W).
    Encodings of the start frame and goal are cached in the dict (as swm does),
    so the S candidate copies share one encoder pass.
    """

    def __init__(self, model, objective: str = 'latent', critic: nn.Module | None = None,
                 latent_window: int = 1):
        super().__init__()
        self.model = model
        self.objective = objective
        self.critic = critic
        self.latent_window = latent_window
        assert objective in ('latent', 'value')
        if objective == 'value':
            assert critic is not None

    # keep swm's Costable duck-typing happy
    def parameters(self, *a, **k):
        return self.model.parameters(*a, **k)

    def _encode_cached(self, info, key_pix, key_emb):
        if key_emb not in info:
            pix = info[key_pix]                     # (B,S,T,3,H,W)
            B = pix.shape[0]
            frame = pix[:, 0, -1]                   # last context frame of sample 0
            with torch.no_grad():
                z = encode_pixels(self.model, frame)
            info[key_emb] = z                       # (B,D)
        return info[key_emb]

    def get_cost(self, info: dict, candidates: torch.Tensor) -> torch.Tensor:
        """candidates: (B,S,N,A) -> costs (B,S)."""
        z0 = self._encode_cached(info, 'pixels', '_z0')
        zg = self._encode_cached(info, 'goal', '_zg')
        B, S, N, A = candidates.shape
        z0r = z0.unsqueeze(1).expand(B, S, -1).reshape(B * S, -1)
        zgr = zg.unsqueeze(1).expand(B, S, -1).reshape(B * S, -1)
        plan = candidates.reshape(B * S, N, A)
        if self.objective == 'latent':
            zN = rollout_latent(self.model, z0r, plan)
            cost = (zN - zgr).pow(2).sum(-1)
        else:
            zN = rollout_latent(self.model, z0r, plan)
            cost = self.critic(zN, zgr)
        return cost.view(B, S)

    def criterion(self, info):  # protocol completeness
        raise NotImplementedError
