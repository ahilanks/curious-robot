"""Post-hoc pixel decoder for the frozen JEPA latent — DIAGNOSTIC ONLY (LeWM App. D).

JEPA never reconstructs pixels; that is the point of the architecture. A small decoder
fitted AFTER the fact on frozen (z -> frame) pairs answers two questions the latent
metrics cannot: what the CEM plan *imagines* (decode the predictor's rollout under an
action sequence, LeWM Fig. 7) and what information the encoder preserves vs discards
(decode(z_now) vs the real frame, LeWM Fig. 8). No gradients ever flow into the encoder
or world model: train_decoder.py encodes under no_grad and fits only this module.

ARCHITECTURE — CHANGED 2026-09-14 TO THE LeWM RECIPE (arXiv 2603.19312, App. D "Decoder
(Visualization Only)"): the latent z (192, the pixels-only StateEncoder output = LeWM's
projected [CLS]) is projected to the hidden width and used as KEY and VALUE in
cross-attention; a fixed set of P = (224/16)^2 = 196 learnable query tokens, one per 16x16
patch of the target image, attends to it through several cross-attention layers with
residual MLP blocks; the resulting patch embeddings are linearly projected to 16x16x3 pixel
patches and rearranged into the 224x224 RGB image. Knobs the paper leaves open (hidden
width, depth, heads, memory tokens) are constructor args and are SAVED IN THE CHECKPOINT
under "arch" so `load_decoder` rebuilds the exact module.

The previous conv/upsample decoder is kept as `ConvLatentDecoder` (checkpoints without an
"arch" key, e.g. wr_sleepret2/decoder_wrs2.pt on HF, still load through `load_decoder`).
"""
from __future__ import annotations

import torch
from einops import rearrange
from torch import nn


class CrossBlock(nn.Module):
    """Pre-norm cross-attention (queries <- latent memory) + residual MLP."""

    def __init__(self, dim: int, heads: int, mlp_ratio: int = 4, self_attn: bool = False,
                 dropout: float = 0.0):
        super().__init__()
        self.ln_q, self.ln_m = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.cross = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ln_s = nn.LayerNorm(dim) if self_attn else None
        self.self_attn = (nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
                          if self_attn else None)
        self.ln_f = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(),
                                 nn.Linear(dim * mlp_ratio, dim))

    def forward(self, q: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        m = self.ln_m(mem)
        q = q + self.cross(self.ln_q(q), m, m, need_weights=False)[0]
        if self.self_attn is not None:                      # OFF by default (not in LeWM App. D)
            s = self.ln_s(q)
            q = q + self.self_attn(s, s, s, need_weights=False)[0]
        return q + self.mlp(self.ln_f(q))


class LatentDecoder(nn.Module):
    """LeWM App. D decoder: z (B, z_dim) -> RGB (B, 3, out_hw, out_hw) in [0, 1]."""

    KIND = "lewm"

    def __init__(self, z_dim: int = 192, hidden: int = 256, depth: int = 3, heads: int = 4,
                 mlp_ratio: int = 4, patch: int = 16, out_hw: int = 224, n_mem: int = 1,
                 self_attn: bool = False, dropout: float = 0.0):
        super().__init__()
        if out_hw % patch:
            raise ValueError(f"out_hw {out_hw} must be a multiple of patch {patch}")
        self.z_dim, self.hidden, self.patch, self.out_hw, self.n_mem = z_dim, hidden, patch, out_hw, n_mem
        self.grid = out_hw // patch
        self._arch = dict(kind=self.KIND, z_dim=z_dim, hidden=hidden, depth=depth, heads=heads,
                          mlp_ratio=mlp_ratio, patch=patch, out_hw=out_hw, n_mem=n_mem,
                          self_attn=self_attn, dropout=dropout)
        self.to_mem = nn.Linear(z_dim, n_mem * hidden)            # [CLS] -> key/value memory
        self.queries = nn.Parameter(torch.randn(1, self.grid ** 2, hidden) * 0.02)  # one per patch
        self.blocks = nn.ModuleList([CrossBlock(hidden, heads, mlp_ratio, self_attn, dropout)
                                     for _ in range(depth)])
        self.norm = nn.LayerNorm(hidden)
        self.to_pix = nn.Linear(hidden, patch * patch * 3)        # patch embedding -> pixels

    def config(self) -> dict:
        return dict(self._arch)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b = z.shape[0]
        mem = self.to_mem(z).view(b, self.n_mem, self.hidden)
        q = self.queries.expand(b, -1, -1)
        for blk in self.blocks:
            q = blk(q, mem)
        x = self.to_pix(self.norm(q))
        x = rearrange(x, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
                      h=self.grid, w=self.grid, p1=self.patch, p2=self.patch, c=3)
        return torch.sigmoid(x)

    @torch.no_grad()
    def to_uint8_hwc(self, z: torch.Tensor):
        """(B, z_dim) -> (B, out_hw, out_hw, 3) uint8 numpy — dashboard-ready frames."""
        return (self(z).permute(0, 2, 3, 1).clamp(0, 1) * 255).byte().cpu().numpy()


class ConvLatentDecoder(nn.Module):
    """LEGACY (pre-2026-09-14) conv decoder: z -> 7x7 map -> five NN-upsample+conv stages
    (7-14-28-56-112-224) -> sigmoid RGB at 224. Kept so older decoder checkpoints load."""

    KIND = "conv"

    def __init__(self, z_dim: int = 192, base: int = 256, out_hw: int = 224):
        super().__init__()
        if out_hw != 224:
            raise ValueError("ConvLatentDecoder is fixed at 224 (7 * 2^5)")
        chs = [base, 192, 128, 96, 64, 32]
        self.z_dim, self.out_hw = z_dim, out_hw
        self._arch = dict(kind=self.KIND, z_dim=z_dim, base=base, out_hw=out_hw)
        self.fc = nn.Linear(z_dim, chs[0] * 7 * 7)
        blocks: list[nn.Module] = []
        for cin, cout in zip(chs[:-1], chs[1:]):
            blocks += [nn.Upsample(scale_factor=2, mode="nearest"),
                       nn.Conv2d(cin, cout, 3, padding=1),
                       nn.GroupNorm(8, cout), nn.SiLU()]
        blocks.append(nn.Conv2d(chs[-1], 3, 3, padding=1))
        self.net = nn.Sequential(*blocks)

    def config(self) -> dict:
        return dict(self._arch)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).view(z.shape[0], -1, 7, 7)
        return torch.sigmoid(self.net(x))

    @torch.no_grad()
    def to_uint8_hwc(self, z: torch.Tensor):
        return (self(z).permute(0, 2, 3, 1).clamp(0, 1) * 255).byte().cpu().numpy()


def build_decoder(arch: dict | None, z_dim: int | None = None) -> nn.Module:
    """Rebuild a decoder from a saved `arch` dict; no arch (old checkpoints) -> conv."""
    arch = dict(arch or {})
    kind = arch.pop("kind", "conv")
    if z_dim is not None:
        arch["z_dim"] = z_dim
    if kind == "conv":
        return ConvLatentDecoder(**arch)
    if kind == "lewm":
        return LatentDecoder(**arch)
    raise ValueError(f"unknown decoder kind {kind!r}")


def load_decoder(src, device="cpu", z_dim: int | None = None):
    """Load a decoder checkpoint (path or dict) -> (module in eval mode, no grads; meta dict).
    Verifies z_dim against the run when given (a decoder is tied to ONE encoder's latent)."""
    ck = torch.load(src, map_location=device, weights_only=False) if not isinstance(src, dict) else src
    if z_dim is not None and int(ck["z_dim"]) != int(z_dim):
        raise ValueError(f"decoder z_dim {ck['z_dim']} != run z_dim {z_dim}")
    dec = build_decoder(ck.get("arch"), z_dim=int(ck["z_dim"])).to(device)
    dec.load_state_dict(ck["decoder"])
    dec.eval().requires_grad_(False)
    meta = {k: v for k, v in ck.items() if k != "decoder"}
    return dec, meta
