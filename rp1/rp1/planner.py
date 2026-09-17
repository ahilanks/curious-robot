"""RP1: a weight-tied residual plan refiner trained through the frozen world
model against the learned critic (Sec. 4–5, App. B.2 of the paper).

    a_{k+1} = clip_[-amax, amax]( a_k + f_θ(a_k, v_k, g_k) ),   k = 0..K-1,  a_0 = 0
    v_k = V(H(a_k, z_0), z_g),   g_k = ∇_{a_k} v_k        (v_k, g_k detached as inputs)
    J(θ) = E[ v_K + λ_mean · (1/K) Σ_{k=1..K} v_k ]      (grad flows through the WM rollouts)
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from .wm import encode_pixels, rollout_latent


class Refiner(nn.Module):
    """f_θ : (plan, value, value-gradient) -> plan residual. 3 FC layers, ReLU, width 512."""

    def __init__(self, plan_dim: int, hidden: int = 512):
        super().__init__()
        self.plan_dim = plan_dim
        self.net = nn.Sequential(
            nn.Linear(2 * plan_dim + 1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, plan_dim),
        )
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, a_flat, v, g_flat):
        return self.net(torch.cat([a_flat, g_flat, v.unsqueeze(-1)], dim=-1))


class RP1Planner(nn.Module):
    """Runs K refinement rounds through the frozen world model + critic."""

    def __init__(self, model, critic, refiner: Refiner, N: int, A: int, K: int = 8, amax: float = 1.8):
        super().__init__()
        self.model, self.critic, self.refiner = model, critic, refiner
        self.N, self.A, self.K, self.amax = N, A, K, amax

    def evaluate(self, z0, zg, a):
        """v = V(H(a, z0), zg) with graph; zN returned too."""
        zN = rollout_latent(self.model, z0, a)
        return self.critic(zN, zg), zN

    def plan(self, z0, zg, a0=None, train: bool = False):
        """Refine a plan. If train=True the whole computation graph is kept so
        that the loss on the v_k can be back-propagated into the refiner.
        Returns dict(plans=[a_0..a_K], values=[v_0..v_K], zN=[...])."""
        B = z0.shape[0]
        if a0 is None:
            a0 = torch.zeros(B, self.N, self.A, device=z0.device)
        a = a0.detach().clone().requires_grad_(True)
        plans, values, zNs = [a], [], []
        for k in range(self.K):
            with torch.enable_grad():
                v, zN = self.evaluate(z0, zg, a)
                (g,) = torch.autograd.grad(v.sum(), a, retain_graph=train)
                if not train:
                    v, zN = v.detach(), zN.detach()
            values.append(v); zNs.append(zN)
            delta = self.refiner(a.flatten(1), v.detach(), g.detach().flatten(1))
            a_next = (a + delta.view(B, self.N, self.A)).clamp(-self.amax, self.amax)
            a = a_next if train else a_next.detach().requires_grad_(True)
            plans.append(a)
        # final (K-th) evaluation — the paper counts it in its "9 rollouts"
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            vK, zNK = self.evaluate(z0, zg, a)
        values.append(vK); zNs.append(zNK)
        return dict(plans=plans, values=values, zN=zNs)


# --------------------------------------------------------------------------
# swm Solver protocol wrapper (drop-in for CEMSolver etc.)
# --------------------------------------------------------------------------

class RP1Solver:
    def __init__(self, planner: RP1Planner, device='cuda', latent_norm=None):
        self.planner = planner
        self.device = device
        self.latent_norm = latent_norm  # callable applied to latents before the critic (standardised critics)
        self._configured = False

    def configure(self, *, action_space, n_envs, config):
        self._n_envs = n_envs
        self._config = config
        self._action_dim = int(np.prod(action_space.shape[1:]))
        self._configured = True

    @property
    def n_envs(self):
        return self._n_envs

    @property
    def action_dim(self):
        return self._action_dim * self._config.action_block

    @property
    def horizon(self):
        return self._config.horizon

    def __call__(self, *a, **k):
        return self.solve(*a, **k)

    def encode(self, info):
        pix = info['pixels'].to(self.device)   # (B,T,3,H,W)
        goal = info['goal'].to(self.device)    # (B,T,3,H,W)
        z0 = encode_pixels(self.planner.model, pix[:, -1])
        zg = encode_pixels(self.planner.model, goal[:, -1])
        return z0, zg

    def solve(self, info_dict: dict, init_action=None) -> dict:
        z0, zg = self.encode(info_dict)
        out = self.planner.plan(z0, zg, train=False)
        aK = out['plans'][-1].detach()
        return dict(actions=aK.cpu(), values=[v.detach().cpu() for v in out['values']])


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

class ActorSampler:
    """Start latent + hindsight goal (Δ ∈ [1, max_delta] blocks, or cross-episode)."""

    def __init__(self, cache, episodes, max_delta=12, cross_prob=0.3, seed=0):
        self.c = cache
        self.eps = np.asarray(episodes)
        self.max_delta, self.cross_prob = max_delta, cross_prob
        self.rng = np.random.default_rng(seed)
        self.ep_len = cache.ep_len[self.eps]
        self.ep_off = cache.ep_off[self.eps]
        self.block = cache.block

    def sample(self, B):
        rng, blk = self.rng, self.block
        e = rng.integers(0, len(self.eps), B)
        L = self.ep_len[e]
        t = np.clip((rng.random(B) * (L - 1 - blk)).astype(np.int64), 0, None)
        T_rem = np.maximum((L - 1 - t) // blk, 1)
        delta = np.minimum((rng.random(B) * np.minimum(T_rem, self.max_delta)).astype(np.int64) + 1, T_rem)
        rows0 = self.ep_off[e] + t
        rowsg = self.ep_off[e] + t + delta * blk
        cross = rng.random(B) < self.cross_prob
        e2 = rng.integers(0, len(self.eps), B)
        t2 = (rng.random(B) * self.ep_len[e2]).astype(np.int64)
        rowsg = np.where(cross, self.ep_off[e2] + t2, rowsg)
        return self.c.z[torch.as_tensor(rows0)], self.c.z[torch.as_tensor(rowsg)]


def cosine_lr(step, total, lr0, lr_final):
    f = min(1.0, step / max(1, total))
    return lr_final + 0.5 * (lr0 - lr_final) * (1 + math.cos(math.pi * f))


class RP1Trainer:
    def __init__(self, planner: RP1Planner, sampler: ActorSampler, critic_trainer, cfg, device='cuda'):
        """critic_trainer: CriticTrainer whose .target (EMA) is the planner's critic (teacher)."""
        self.p, self.s, self.ct, self.cfg = planner, sampler, critic_trainer, cfg
        self.opt = torch.optim.Adam(planner.refiner.parameters(), lr=cfg['actor_lr'])
        self.step_i = 0
        self.replay = []  # (zN, zg) pairs for replay-probability starts
        self.device = device

    def step(self):
        cfg = self.cfg
        B = cfg['batch_size']
        z0, zg = self.s.sample(B)
        # replay: continue from imagined terminal states of earlier batches
        if cfg.get('replay_prob', 0) > 0 and self.replay:
            m = torch.rand(B, device=z0.device) < cfg['replay_prob']
            rz, rg = self.replay[np.random.randint(len(self.replay))]
            n = min(int(m.sum()), rz.shape[0])
            if n > 0:
                idx = torch.nonzero(m).squeeze(1)[:n]
                z0 = z0.clone(); zg = zg.clone()
                z0[idx] = rz[:n]; zg[idx] = rg[:n]
        lr = cfg['actor_lr'] if cfg.get('lr_schedule', 'const') == 'const' else \
            cosine_lr(self.step_i, cfg['steps'], cfg['actor_lr'], cfg['actor_lr'] * 0.1)
        for g in self.opt.param_groups:
            g['lr'] = lr
        out = self.p.plan(z0, zg, train=True)
        vals = torch.stack(out['values'][1:], 0)  # v_1..v_K  (K, B)
        loss = out['values'][-1].mean() + cfg['lambda_mean'] * vals.mean()
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = nn.utils.clip_grad_norm_(self.p.refiner.parameters(), 10.0)
        self.opt.step()
        info = dict(loss=loss.item(), v0=out['values'][0].mean().item(), vK=out['values'][-1].mean().item(),
                    lr=lr, gnorm=float(gn), amax_frac=(out['plans'][-1].abs() >= self.p.amax - 1e-6).float().mean().item())
        # critic co-training (live for cfg['critic_live_steps'] actor steps, then frozen)
        if self.step_i < cfg['critic_live_steps']:
            extra = None
            if cfg.get('value_expansion', 0) > 0:
                zN = out['zN'][-1].detach()
                with torch.no_grad():
                    nblk = float(self.p.N)
                    from .critic import c_gamma
                    y = c_gamma(nblk, self.ct.gamma) + (self.ct.gamma ** nblk) * self.ct.target(zN, zg)
                extra = (z0.detach(), zg.detach(), y, cfg['value_expansion'])
            for _ in range(cfg.get('critic_ratio', 1)):
                cinfo = self.ct.step(cfg['td_batch'], extra=extra)
            info['critic_loss'] = cinfo['loss']
        if cfg.get('replay_prob', 0) > 0:
            self.replay.append((out['zN'][-1].detach(), zg.detach()))
            self.replay = self.replay[-20:]
        self.step_i += 1
        return info
