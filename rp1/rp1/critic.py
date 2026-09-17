"""Goal-conditioned quasimetric critic V_ψ(z, z_g) ≈ temporal cost-to-go (in
action blocks), trained offline by n-step TD + hindsight goals + asymmetric
(expectile) Huber regression — Appendix B.1 of the paper.

    V(z, g) = ||u(z) - u(g)||₂ + max_j ReLU(v_j(g) - v_j(z))      (MRN head)
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MRNCritic(nn.Module):
    def __init__(self, in_dim=192, hidden=256, embed=128, depth=2):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        layers += [nn.Linear(d, embed)]
        self.net = nn.Sequential(*layers)
        self.embed = embed

    def head(self, z):
        h = self.net(z)
        half = self.embed // 2
        return h[..., :half], h[..., half:]

    def forward(self, z, zg):
        u, v = self.head(z)
        ug, vg = self.head(zg)
        sym = torch.sqrt((u - ug).pow(2).sum(-1) + 1e-8)
        asym = F.relu(vg - v).max(-1).values
        return sym + asym


class NormCritic(nn.Module):
    """MRN critic reading standardised latents (Reacher setting); buffers fixed from the cache."""

    def __init__(self, critic: MRNCritic, mean=None, std=None):
        super().__init__()
        self.critic = critic
        D = critic.net[0].in_features
        self.register_buffer('mean', torch.zeros(1, D) if mean is None else mean.detach().clone().view(1, D))
        self.register_buffer('std', torch.ones(1, D) if std is None else std.detach().clone().view(1, D))

    def forward(self, z, zg):
        return self.critic((z - self.mean) / self.std, (zg - self.mean) / self.std)


def make_critic(in_dim=192, hidden=256, embed=128, depth=2, mean=None, std=None):
    return NormCritic(MRNCritic(in_dim, hidden, embed, depth), mean, std)


def c_gamma(n, gamma):
    if gamma == 1.0:
        return float(n) if np.isscalar(n) else n.float()
    if np.isscalar(n):
        return (1 - gamma ** n) / (1 - gamma)
    return (1 - gamma ** n.float()) / (1 - gamma)


def expectile_huber(diff, tau, delta=1.0):
    w = torch.abs(tau - (diff > 0).float())
    return (w * F.huber_loss(diff, torch.zeros_like(diff), reduction='none', delta=delta)).mean()


class TDSampler:
    """Samples (anchor, n-step successor, hindsight goal) triples from a LatentCache.

    Time unit = block (stride `block` primitive rows). Anchors are drawn from
    any primitive row of the training episodes; the successor / goal are taken
    at the same phase (t + k*block) so every triple is block-aligned.
    """

    def __init__(self, cache, episodes, n_step=50, gamma=1.0, cross_prob=0.3, seed=0):
        self.c = cache
        self.eps = np.asarray(episodes)
        self.n_step = n_step
        self.gamma = gamma
        self.cross_prob = cross_prob
        self.rng = np.random.default_rng(seed)
        self.ep_len = cache.ep_len[self.eps]
        self.ep_off = cache.ep_off[self.eps]
        # number of blocks remaining after step t: T_blocks(t) = (len-1-t)//block
        self.block = cache.block

    def sample(self, B):
        rng, blk = self.rng, self.block
        e = rng.integers(0, len(self.eps), B)
        L = self.ep_len[e]
        # anchors with at least one block of future
        t = (rng.random(B) * (L - 1 - blk)).astype(np.int64)
        t = np.clip(t, 0, None)
        T_rem = (L - 1 - t) // blk                     # blocks remaining in episode
        n_eff = np.minimum(self.n_step, T_rem)
        cross = rng.random(B) < self.cross_prob
        # in-episode goals: uniform over remaining horizon (1..T_rem blocks)
        delta = (rng.random(B) * T_rem).astype(np.int64) + 1
        delta = np.minimum(delta, T_rem)
        rows_a = self.ep_off[e] + t
        rows_s = self.ep_off[e] + t + n_eff * blk
        rows_g = self.ep_off[e] + t + delta * blk
        # cross-episode goals: random row of another training episode
        e2 = rng.integers(0, len(self.eps), B)
        t2 = (rng.random(B) * self.ep_len[e2]).astype(np.int64)
        rows_g = np.where(cross, self.ep_off[e2] + t2, rows_g)
        exact = (~cross) & (delta <= n_eff)
        return dict(
            za=self.c.z[torch.as_tensor(rows_a)], zs=self.c.z[torch.as_tensor(rows_s)],
            zg=self.c.z[torch.as_tensor(rows_g)],
            delta=torch.as_tensor(delta, device=self.c.device, dtype=torch.float32),
            n_eff=torch.as_tensor(n_eff, device=self.c.device, dtype=torch.float32),
            exact=torch.as_tensor(exact, device=self.c.device),
        )


class CriticTrainer:
    """Offline TD training of the MRN critic (Eq. 68–71)."""

    def __init__(self, critic, sampler, gamma=1.0, tau=0.1, lr=1e-3, polyak=0.005,
                 tau_final=None, lr_final=None, total_steps=6000, huber_delta=1.0,
                 device='cuda'):
        self.critic = critic.to(device)
        self.target = copy.deepcopy(self.critic).requires_grad_(False)
        self.sampler = sampler
        self.gamma, self.tau0, self.tau1 = gamma, tau, tau_final or tau
        self.lr0, self.lr1 = lr, lr_final or lr
        self.polyak = polyak
        self.total = total_steps
        self.huber_delta = huber_delta
        self.opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.step_i = 0
        self.extra_targets = None  # optional value-expansion hook

    def _anneal(self):
        f = min(1.0, self.step_i / max(1, self.total))
        tau = self.tau0 + (self.tau1 - self.tau0) * f
        lr = self.lr0 * (self.lr1 / self.lr0) ** f  # geometric anneal
        for g in self.opt.param_groups:
            g['lr'] = lr
        return tau

    def targets(self, b):
        with torch.no_grad():
            boot = c_gamma(b['n_eff'], self.gamma) + (self.gamma ** b['n_eff']) * self.target(b['zs'], b['zg'])
            y = torch.where(b['exact'], b['delta'], boot)
        return y

    def step(self, B=1024, extra=None):
        tau = self._anneal()
        b = self.sampler.sample(B)
        y = self.targets(b)
        v = self.critic(b['za'], b['zg'])
        loss = expectile_huber(v - y, tau, self.huber_delta)
        if extra is not None:  # value expansion: (z_imagined, zg, target) triples
            zi, zgi, yi, w = extra
            vi = self.critic(zi, zgi)
            loss = loss + w * expectile_huber(vi - yi, tau, self.huber_delta)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.opt.step()
        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.target.parameters()):
                pt.lerp_(p, self.polyak)
        self.step_i += 1
        return dict(loss=loss.item(), v_mean=v.mean().item(), y_mean=y.mean().item(),
                    exact_frac=b['exact'].float().mean().item())
