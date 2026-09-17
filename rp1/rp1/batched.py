"""Vectorised RP1 training: R independent runs (planner seeds) trained in one
process, sharing every world-model rollout kernel. Mathematically identical to
R separate runs of rp1.planner.RP1Trainer / rp1.critic.CriticTrainer:

  * every run has its own refiner f_θr, its own co-trained critic + EMA teacher,
    its own samplers (seeded per run); parameters are stacked along a leading
    run axis and applied with batched matmuls, so Adam (element-wise) and the
    per-run gradient clipping act exactly as R independent optimisers.
  * the training loss J = v_K + λ/K Σ_k v_k is differentiated through the refined
    plans only via the value gradients g_k = ∂v_k/∂a_k the planner already
    computes (v_k, g_k are detached refiner inputs, so the chain rule gives
    ∂J/∂θ = Σ_k c_k g_kᵀ ∂a_k/∂θ). We therefore use the surrogate
    Σ_k c_k <stopgrad(g_k), a_k>, which has the same gradient and avoids a
    second backward pass through the world model.
  * the (rollout → critic → ∂/∂a) evaluation is captured once in a CUDA graph
    (static shapes) and replayed 9× per step — this is what makes the step
    launch-bound no longer.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .critic import c_gamma
from .wm import rollout_latent
from .planner import cosine_lr


class BLinear(nn.Module):
    """R independent nn.Linear layers: x (R,B,i) -> (R,B,o)."""

    def __init__(self, R, i, o, seeds=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(R, i, o))
        self.bias = nn.Parameter(torch.empty(R, 1, o))
        bound = 1 / math.sqrt(i)
        for r in range(R):
            g = torch.Generator().manual_seed(int(seeds[r]) * 1000 + i * 7 + o) if seeds is not None else None
            self.weight.data[r].uniform_(-bound, bound, generator=g)
            self.bias.data[r].uniform_(-bound, bound, generator=g)

    def forward(self, x):
        return torch.baddbmm(self.bias, x, self.weight)

    # conversion to / from single-run nn.Linear state (weight (o,i), bias (o,))
    def load_single(self, w, b, r=None):
        rs = range(self.weight.shape[0]) if r is None else [r]
        for rr in rs:
            self.weight.data[rr] = w.t().to(self.weight.device)
            self.bias.data[rr, 0] = b.to(self.bias.device)

    def export_single(self, r):
        return self.weight.data[r].t().contiguous().cpu(), self.bias.data[r, 0].contiguous().cpu()


class BatchedRefiner(nn.Module):
    """R copies of planner.Refiner (2P+1 -> 512 -> 512 -> P)."""

    def __init__(self, R, plan_dim, hidden=512, seeds=None):
        super().__init__()
        self.R, self.plan_dim = R, plan_dim
        self.l1 = BLinear(R, 2 * plan_dim + 1, hidden, seeds)
        self.l2 = BLinear(R, hidden, hidden, seeds)
        self.l3 = BLinear(R, hidden, plan_dim, seeds)
        nn.init.zeros_(self.l3.bias)

    def forward(self, a_flat, v, g_flat):  # (R,B,P), (R,B), (R,B,P)
        x = torch.cat([a_flat, g_flat, v.unsqueeze(-1)], dim=-1)
        x = F.relu(self.l1(x)); x = F.relu(self.l2(x))
        return self.l3(x)

    def export_single(self, r):
        sd = {}
        for j, l in enumerate((self.l1, self.l2, self.l3)):
            w, b = l.export_single(r)
            sd[f'net.{2 * j}.weight'] = w; sd[f'net.{2 * j}.bias'] = b
        return sd


class BatchedMRN(nn.Module):
    """R copies of critic.NormCritic(MRNCritic) sharing the latent mean/std buffers."""

    def __init__(self, R, in_dim=192, hidden=256, embed=128, depth=2, mean=None, std=None):
        super().__init__()
        self.R, self.embed = R, embed
        dims = [in_dim] + [hidden] * depth + [embed]
        self.layers = nn.ModuleList([BLinear(R, dims[j], dims[j + 1]) for j in range(len(dims) - 1)])
        self.register_buffer('mean', torch.zeros(1, 1, in_dim) if mean is None else mean.detach().clone().view(1, 1, in_dim))
        self.register_buffer('std', torch.ones(1, 1, in_dim) if std is None else std.detach().clone().view(1, 1, in_dim))

    def head(self, z):
        h = (z - self.mean) / self.std
        for j, l in enumerate(self.layers):
            h = l(h)
            if j < len(self.layers) - 1:
                h = F.relu(h)
        half = self.embed // 2
        return h[..., :half], h[..., half:]

    def forward(self, z, zg):  # (R,B,D) x2 -> (R,B)
        u, v = self.head(z); ug, vg = self.head(zg)
        sym = torch.sqrt((u - ug).pow(2).sum(-1) + 1e-8)
        asym = F.relu(vg - v).max(-1).values
        return sym + asym

    def load_single(self, sd, r=None):
        """sd: NormCritic state dict (critic.net.{0,2,4}.weight/bias, mean, std) broadcast to runs."""
        for j, l in enumerate(self.layers):
            l.load_single(sd[f'critic.net.{2 * j}.weight'], sd[f'critic.net.{2 * j}.bias'], r)
        if 'mean' in sd:
            self.mean.copy_(sd['mean'].view(1, 1, -1)); self.std.copy_(sd['std'].view(1, 1, -1))

    def export_single(self, r):
        sd = {'mean': self.mean.view(1, -1).cpu(), 'std': self.std.view(1, -1).cpu()}
        for j, l in enumerate(self.layers):
            w, b = l.export_single(r)
            sd[f'critic.net.{2 * j}.weight'] = w; sd[f'critic.net.{2 * j}.bias'] = b
        return sd


def clip_grad_per_run(params, max_norm):
    """Per-run gradient-norm clipping over stacked (R,...) parameters."""
    R = params[0].shape[0]
    sq = torch.zeros(R, device=params[0].device)
    for p in params:
        if p.grad is not None:
            sq += p.grad.pow(2).flatten(1).sum(1)
    norm = sq.sqrt()
    scale = (max_norm / (norm + 1e-6)).clamp(max=1.0)
    for p in params:
        if p.grad is not None:
            p.grad.mul_(scale.view(-1, *([1] * (p.grad.dim() - 1))))
    return norm


class ValueGrad:
    """(z0, zg, a) -> (v, dv/da) through the frozen WM rollout + batched critic.
    The forward (rollout -> critic) is optionally torch.compile'd (inductor fuses
    the ~80% elementwise kernels; numerically identical to eager in fp32), the
    backward is AOTAutograd's compiled backward, and the whole fwd+bwd is captured
    in a CUDA graph (static shapes) and replayed 9x per training step."""

    def __init__(self, model, critic: BatchedMRN, R, B, N, A, D, device='cuda', use_graph=True, use_compile=True):
        self.model, self.critic, self.R, self.B, self.N, self.A = model, critic, R, B, N, A
        self.s_a = torch.zeros(R * B, N, A, device=device, requires_grad=True)
        self.s_z0 = torch.zeros(R * B, D, device=device)
        self.s_zg = torch.zeros(R * B, D, device=device)
        self._fwd = torch.compile(self._fwd_eager, mode='default', dynamic=False) if use_compile else self._fwd_eager
        self.graph = None
        if use_graph:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self._compute()
            torch.cuda.current_stream().wait_stream(s)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.o_v, self.o_g = self._compute()

    def _fwd_eager(self, z0, zg, a):
        zN = rollout_latent(self.model, z0, a)
        return self.critic(zN.view(self.R, self.B, -1), zg.view(self.R, self.B, -1))

    def _compute(self):
        with torch.enable_grad():
            v = self._fwd(self.s_z0, self.s_zg, self.s_a)
            (g,) = torch.autograd.grad(v.sum(), self.s_a)
        return v.detach(), g

    def __call__(self, z0, zg, a):
        with torch.no_grad():
            self.s_z0.copy_(z0.reshape(self.R * self.B, -1)); self.s_zg.copy_(zg.reshape(self.R * self.B, -1))
            self.s_a.copy_(a.detach().reshape(self.R * self.B, self.N, self.A))
        if self.graph is not None:
            self.graph.replay()
            v, g = self.o_v.clone(), self.o_g.clone()
        else:
            v, g = self._compute()
        return v.view(self.R, self.B), g.view(self.R, self.B, self.N, self.A)


def use_math_sdpa():
    """For T<=3 tokens the fused attention kernels are pure overhead (~200us each);
    the decomposed path is numerically identical (fp32) and ~1.5x faster overall."""
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)


class RunGroup:
    """Runs sharing one actor config (e.g. all seeds of one goal horizon)."""

    def __init__(self, name, refiner: BatchedRefiner, acfg, samplers, seeds):
        self.name, self.ref, self.acfg, self.samplers, self.seeds = name, refiner, acfg, samplers, seeds
        self.R = refiner.R
        self.opt = torch.optim.Adam(refiner.parameters(), lr=acfg['actor_lr'])


class BatchedRP1Trainer:
    def __init__(self, model, groups, critic: BatchedMRN, teacher: BatchedMRN, td_samplers, ccfg, N, A, K,
                 device='cuda', use_graph=True, use_compile=True):
        self.model, self.groups, self.critic, self.teacher = model, groups, critic, teacher
        self.td_samplers = td_samplers
        self.R = sum(g.R for g in groups)
        assert self.R == critic.R == len(td_samplers)
        B = {g.acfg['batch_size'] for g in groups}; steps = {g.acfg['steps'] for g in groups}
        assert len(B) == 1 and len(steps) == 1, 'groups must share batch size and steps'
        self.B, self.steps = B.pop(), steps.pop()
        self.ccfg, self.N, self.A, self.K = ccfg, N, A, K
        self.slices, r0 = [], 0
        for g in groups:
            self.slices.append(slice(r0, r0 + g.R)); r0 += g.R
        self.amax = torch.tensor([g.acfg['amax'] for g in groups for _ in range(g.R)], device=device).view(-1, 1, 1, 1)
        self.lam = torch.tensor([g.acfg['lambda_mean'] for g in groups for _ in range(g.R)], device=device)
        self.copt = torch.optim.Adam(critic.parameters(), lr=ccfg['lr'])
        D = model.projector.net[-1].out_features if hasattr(model.projector, 'net') else 192
        self.vg = ValueGrad(model, teacher, self.R, self.B, N, A, D, device, use_graph, use_compile)
        self.step_i = 0
        self.replay = [[] for _ in range(self.R)]
        self.device = device
        self.teacher.requires_grad_(False)
        self.extra = None

    # ---- critic co-training (TD on cached latents, per run) ---------------------
    def critic_step(self):
        cc = self.ccfg
        f = min(1.0, self.step_i / max(1, cc['live_steps']))
        tau = cc['expectile'] + (cc['expectile_final'] - cc['expectile']) * f
        lr = cc['lr'] * (cc['lr_final'] / cc['lr']) ** f
        for g in self.copt.param_groups:
            g['lr'] = lr
        bs = [s.sample(cc['td_batch']) for s in self.td_samplers]
        st = lambda k: torch.stack([b[k] for b in bs], 0)
        za, zs, zg = st('za'), st('zs'), st('zg')
        delta, n_eff, exact = st('delta'), st('n_eff'), st('exact')
        gamma = cc['gamma']
        with torch.no_grad():
            boot = c_gamma(n_eff, gamma) + (gamma ** n_eff) * self.teacher(zs, zg)
            y = torch.where(exact, delta, boot)
        v = self.critic(za, zg)
        diff = v - y
        w = torch.abs(tau - (diff > 0).float())
        loss = (w * F.huber_loss(diff, torch.zeros_like(diff), reduction='none', delta=1.0)).mean(1).sum()
        if self.extra is not None:  # value expansion: (z_0 (R,B,D), zg, target y (R,B), weight)
            zi, zgi, yi, wexp = self.extra
            di = self.critic(zi, zgi) - yi
            wi = torch.abs(tau - (di > 0).float())
            loss = loss + wexp * (wi * F.huber_loss(di, torch.zeros_like(di), reduction='none', delta=1.0)).mean(1).sum()
        self.copt.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_per_run(list(self.critic.parameters()), 10.0)
        self.copt.step()
        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.teacher.parameters()):
                pt.lerp_(p, cc['ema'])
        return loss.item() / self.R

    # ---- actor step ------------------------------------------------------------
    def step(self):
        R, B, K = self.R, self.B, self.K
        z0s, zgs = [], []
        for g in self.groups:
            for s in g.samplers:
                z, zz = s.sample(B); z0s.append(z); zgs.append(zz)
        z0, zg = torch.stack(z0s, 0), torch.stack(zgs, 0)
        for g, sl in zip(self.groups, self.slices):
            if g.acfg.get('replay_prob', 0) > 0:
                for r in range(sl.start, sl.stop):
                    if self.replay[r]:
                        m = torch.rand(B, device=z0.device) < g.acfg['replay_prob']
                        rz, rg = self.replay[r][np.random.randint(len(self.replay[r]))]
                        idx = torch.nonzero(m).squeeze(1)
                        z0[r, idx] = rz[idx]; zg[r, idx] = rg[idx]
        lrs = []
        for g in self.groups:
            lr = g.acfg['actor_lr'] if g.acfg.get('lr_schedule', 'const') == 'const' else \
                cosine_lr(self.step_i, g.acfg['steps'], g.acfg['actor_lr'], g.acfg['actor_lr'] * 0.1)
            for pg in g.opt.param_groups:
                pg['lr'] = lr
            lrs.append(lr)
        a = torch.zeros(R, B, self.N, self.A, device=z0.device)
        surrogate = 0.0
        values = []
        for k in range(K + 1):
            v, g_ = self.vg(z0, zg, a)
            values.append(v)
            if k >= 1:
                c_k = self.lam / K + (1.0 if k == K else 0.0)                      # (R,)
                surrogate = surrogate + (c_k.view(R, 1) * (g_ * a).sum((2, 3))).mean(1).sum()
            if k == K:
                break
            delta = torch.cat([grp.ref(a[sl].flatten(2), v[sl], g_[sl].flatten(2)).view(grp.R, B, self.N, self.A)
                               for grp, sl in zip(self.groups, self.slices)], 0)
            a = torch.maximum(torch.minimum(a + delta, self.amax), -self.amax)
        for g in self.groups:
            g.opt.zero_grad(set_to_none=True)
        surrogate.backward()
        gns = []
        for g in self.groups:
            gns.append(clip_grad_per_run(list(g.ref.parameters()), 10.0))
            g.opt.step()
        vals = torch.stack(values, 0)  # (K+1, R, B)
        loss_true = vals[-1].mean(1) + self.lam * vals[1:].mean((0, 2))
        info = dict(loss=loss_true.mean().item(), v0=vals[0].mean().item(), vK=vals[-1].mean().item(),
                    vK_per_run=vals[-1].mean(1).tolist(), lr=lrs, gnorm=torch.cat(gns).mean().item(),
                    amax_frac=(a.abs() >= self.amax - 1e-6).float().mean().item())
        self.extra = None
        need_zN = any(g.acfg.get('replay_prob', 0) > 0 for g in self.groups) or \
            (self.step_i < self.ccfg['live_steps'] and self.ccfg.get('value_expansion', 0) > 0)
        if need_zN:
            with torch.no_grad():
                zN = rollout_latent(self.model, z0.reshape(R * B, -1), a.detach().reshape(R * B, self.N, self.A)).view(R, B, -1)
        if self.step_i < self.ccfg['live_steps']:
            if self.ccfg.get('value_expansion', 0) > 0:
                with torch.no_grad():
                    y = c_gamma(float(self.N), self.ccfg['gamma']) + (self.ccfg['gamma'] ** self.N) * self.teacher(zN, zg)
                self.extra = (z0, zg, y, self.ccfg['value_expansion'])
            for _ in range(self.ccfg.get('critic_ratio', 1)):
                info['critic_loss'] = self.critic_step()
        for g, sl in zip(self.groups, self.slices):
            if g.acfg.get('replay_prob', 0) > 0:
                for r in range(sl.start, sl.stop):
                    self.replay[r].append((zN[r], zg[r]))
                    self.replay[r] = self.replay[r][-20:]
        self.step_i += 1
        return info
