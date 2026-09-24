"""Point-push: the smallest self-vs-object world.

A point agent (disk, radius 0.05) and one pushable disk (radius 0.08) in the unit square.
State = 4 numbers (agent x, y, block x, y); action = the agent's 2D step in [-1, 1]^2
(scaled by `step`). Quasi-static kinematics: no velocities, and the block moves only when
the agent presses into it (pushed out along the line of centres), so the state is Markov
and deterministic.

Invariants that keep it clean:
  * step * sqrt(2) < agent_r + block_r  -> the agent can never tunnel through the block.
  * the block's box sits one agent-diameter (+ margin) inside the agent's box, so the agent
    can always get behind the block; a push-only block in a corner would otherwise be an
    absorbing state.

Optional noisy TV (`tv=True`): `tv_dims` extra observation channels that are fresh N(0, 1)
noise while the agent is inside a disk in the bottom-right corner, zero elsewhere --
unlearnable information the agent can switch on by standing there.

Everything is batched in torch: one tensor op steps every env.
"""
from __future__ import annotations

import math

import torch


class PointPush:
    def __init__(self, n_envs: int, device="cpu", ep_len: int = 200, step: float = 0.05,
                 agent_r: float = 0.05, block_r: float = 0.08, margin: float = 0.01,
                 start_agent=(0.15, 0.15), start_block=(0.5, 0.5),
                 tv: bool = False, tv_center=(0.85, 0.15), tv_r: float = 0.12, tv_dims: int = 4,
                 seed: int = 0):
        assert step * math.sqrt(2) < agent_r + block_r, "step would let the agent tunnel"
        self.E, self.device, self.ep_len, self.step_size = n_envs, torch.device(device), ep_len, step
        self.agent_r, self.block_r, self.R = agent_r, block_r, agent_r + block_r
        self.agent_lo, self.agent_hi = agent_r, 1.0 - agent_r
        self.block_lo = block_r + 2 * agent_r + margin
        self.block_hi = 1.0 - self.block_lo
        assert self.agent_hi - self.block_hi >= self.R, "agent must fit between block and wall"
        self.start_agent = torch.tensor(start_agent, device=self.device)
        self.start_block = torch.tensor(start_block, device=self.device)
        self.tv, self.tv_r, self.tv_dims = tv, tv_r, (tv_dims if tv else 0)
        self.tv_center = torch.tensor(tv_center, device=self.device)
        self.obs_dim = 4 + self.tv_dims
        self.act_dim = 2
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        self.agent = self.start_agent.expand(n_envs, 2).clone()
        self.block = self.start_block.expand(n_envs, 2).clone()
        self.t = 0

    # ---------------------------------------------------------------- observation
    def in_tv(self) -> torch.Tensor:
        if not self.tv:
            return torch.zeros(self.E, dtype=torch.bool, device=self.device)
        return (self.agent - self.tv_center).norm(dim=-1) < self.tv_r

    def obs(self) -> torch.Tensor:
        parts = [2 * self.agent - 1, 2 * self.block - 1]          # positions in [-1, 1]
        if self.tv:
            noise = torch.randn(self.E, self.tv_dims, device=self.device, generator=self.gen)
            parts.append(noise * self.in_tv().unsqueeze(-1))
        return torch.cat(parts, dim=-1)

    def state(self) -> torch.Tensor:
        """Ground truth (agent xy, block xy) in [0, 1] -- for metrics only, never the agent's input."""
        return torch.cat([self.agent, self.block], dim=-1)

    # ---------------------------------------------------------------- dynamics
    def reset(self, random_start: bool = False) -> torch.Tensor:
        self.t = 0
        if not random_start:
            self.agent = self.start_agent.expand(self.E, 2).clone()
            self.block = self.start_block.expand(self.E, 2).clone()
            return self.obs()
        u = lambda: torch.rand(self.E, 2, device=self.device, generator=self.gen)
        self.block = self.block_lo + (self.block_hi - self.block_lo) * u()
        self.agent = self.agent_lo + (self.agent_hi - self.agent_lo) * u()
        for _ in range(100):                                     # rejection-sample non-overlapping starts
            bad = (self.agent - self.block).norm(dim=-1) < self.R
            if not bad.any():
                break
            self.agent[bad] = (self.agent_lo + (self.agent_hi - self.agent_lo) * u())[bad]
        return self.obs()

    def step(self, action: torch.Tensor):
        a_old, b_old = self.agent, self.block
        a = (a_old + self.step_size * action.clamp(-1, 1)).clamp(self.agent_lo, self.agent_hi)
        d = b_old - a
        dist = d.norm(dim=-1, keepdim=True)
        contact = (dist < self.R).squeeze(-1)                    # this move pressed into the block
        b = torch.where(contact.unsqueeze(-1), a + self.R * d / dist.clamp_min(1e-8), b_old)
        b = b.clamp(self.block_lo, self.block_hi)
        d2 = a - b                                               # block hit its bound: agent stops against it
        dist2 = d2.norm(dim=-1, keepdim=True)
        a = torch.where(dist2 < self.R, b + self.R * d2 / dist2.clamp_min(1e-8), a)
        self.agent, self.block = a, b
        self.t += 1
        info = {"contact": contact, "block_move": (b - b_old).norm(dim=-1), "in_tv": self.in_tv()}
        return self.obs(), info
