"""Proprioceptive branch of the state encoder (README §State encoder).

    MLP( symlog( q_t, qdot_t, u^app_{t-1} ) )  ->  R^{d_prop}   (d_prop = 64)

symlog squashes the very different scales of joint angle / velocity / torque
into a comparable range without clipping, so a single MLP can read all three.
"""
from __future__ import annotations

import torch
from torch import nn


def symlog(x: torch.Tensor) -> torch.Tensor:
    """symlog(x) = sign(x) * log(1 + |x|)  -- elementwise, scale-compressing."""
    return torch.sign(x) * torch.log1p(x.abs())


class ProprioEncoder(nn.Module):
    def __init__(self, n_dof: int = 6, out_dim: int = 64, hidden: int = 256):
        super().__init__()
        in_dim = 3 * n_dof  # [q, qdot, u_prev]
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        """proprio: (B, 3*n_dof) = [q, qdot, u_prev] -> (B, out_dim)."""
        return self.net(symlog(proprio.float()))
