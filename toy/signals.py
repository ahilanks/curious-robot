"""Intrinsic signals for the point-push toy: what the agent gets paid for.

Every signal maps a finished rollout (obs (T+1, E, D), executed actions (T, E, 2), ground-truth
state (T+1, E, 4)) to a per-transition reward (T, E); `update` then learns from the rollout.

  none  -- zero reward: the untrained Gaussian policy, i.e. a random walk.
  pred  -- prediction error of a learned dynamics model f(o, a) -> o' (the campaign's r_cur /
           --goal-score mse): e(x) = mean_d (f(o, a) - o')^2, scored before this rollout's update.
  lp    -- learning progress |e_old(x) - e_now(x)|: the change in that same error between the
           current model and a slow EMA copy of its weights (gamma-progress, Kim et al. 2020,
           "Active World Model Learning with Progress Curiosity"; |.| as in train.py's
           --goal-score lp, which rewards forgetting as well as learning).
  ln    -- learnable novelty (Zhang & Levin 2026, "Intelligence from Learnable Novelty",
           arXiv 2607.18433 -- the estimator EpiJEPA uses as its anti-collapse score). Per episode,
           an online ridge readout (covariance-form RLS) maps frozen random-reservoir features
           phi(x_t) to the stacked next-tau window (x_{t+1} .. x_{t+tau}); the episode's epiplexity
           is S = 1/2 log2 det(I + eta W W^T) and each step is paid the increment S_j - S_{j-1},
           so the return is the whole-episode epiplexity. A batched port of their
           src/rl/reward.py + rc_epiplexity/{core,online,reservoirs}.py with their RL settings
           (reservoir width 32, ridge 0.3, eta 1, tau = 2x the random-policy characteristic time
           clipped to [8, 48], frozen random-rollout normalisation, unit target scale).
  count -- oracle reference, NOT available to a real agent: 1/sqrt(N + 1) visits of the
           ground-truth (agent, block) cell on a 10^4 grid.
"""
from __future__ import annotations

import copy
import math

import torch
from torch import nn


class Signal:
    name = "none"
    learns = False

    def reward(self, ro) -> tuple[torch.Tensor, dict]:
        T, E = ro["act"].shape[:2]
        return torch.zeros(T, E, device=ro["act"].device), {}

    def update(self, ro) -> dict:
        return {}


# ------------------------------------------------------------------ prediction error / LP
class Dynamics(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim + act_dim, hidden), nn.ELU(),
                                 nn.Linear(hidden, hidden), nn.ELU(),
                                 nn.Linear(hidden, obs_dim))

    def forward(self, o, a):
        return o + self.net(torch.cat([o, a], dim=-1))            # residual: predicts o' via delta


def transition_error(model, o, a, o2):
    return (model(o, a) - o2).pow(2).mean(-1)


class PredError(Signal):
    name = "pred"
    learns = True

    def __init__(self, obs_dim, act_dim, device, lr=1e-3, batch=1024, epochs=1, hidden=256):
        self.model = Dynamics(obs_dim, act_dim, hidden).to(device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.batch, self.epochs = batch, epochs

    @torch.no_grad()
    def errors(self, model, ro):
        return transition_error(model, ro["obs"][:-1], ro["act"], ro["obs"][1:])

    def reward(self, ro):
        e = self.errors(self.model, ro)
        return e, {"err": e}

    def _after_step(self):
        pass

    def update(self, ro):
        D = ro["obs"].shape[-1]
        o, o2 = ro["obs"][:-1].reshape(-1, D), ro["obs"][1:].reshape(-1, D)
        a = ro["act"].reshape(-1, ro["act"].shape[-1])
        n, losses = o.shape[0], []
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=o.device)
            for i in range(0, n, self.batch):
                idx = perm[i:i + self.batch]
                loss = transition_error(self.model, o[idx], a[idx], o2[idx]).mean()
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                self.opt.step()
                self._after_step()
                losses.append(loss.detach())
        return {"wm_loss": float(torch.stack(losses).mean())}


class LearningProgress(PredError):
    name = "lp"

    def __init__(self, obs_dim, act_dim, device, ema=0.01, **kw):
        super().__init__(obs_dim, act_dim, device, **kw)
        self.old = copy.deepcopy(self.model).requires_grad_(False)
        self.ema = ema                                          # lag ~ 1/ema WM gradient steps

    def reward(self, ro):
        e_now, e_old = self.errors(self.model, ro), self.errors(self.old, ro)
        return (e_old - e_now).abs(), {"err": e_now}

    @torch.no_grad()
    def _after_step(self):
        for p_old, p in zip(self.old.parameters(), self.model.parameters()):
            p_old.lerp_(p, self.ema)


class SignedProgress(LearningProgress):
    """gamma-progress exactly as published (Kim et al. 2020): e_old - e_now, signed. On pure noise
    the two errors are equal in expectation, so the mean reward is ~0; |.| (above) turns the
    model's jitter on noise into a positive payout."""
    name = "lps"

    def reward(self, ro):
        e_now, e_old = self.errors(self.model, ro), self.errors(self.old, ro)
        return e_old - e_now, {"err": e_now}


# ------------------------------------------------------------------ learnable novelty
class PreActNorm(nn.Module):
    """Normalise each sample over its channels before the nonlinearity (their reservoirs.py)."""

    def forward(self, x):
        return (x - x.mean(1, keepdim=True)) / torch.sqrt(x.var(1, unbiased=False, keepdim=True) + 1e-5)


def make_reservoir(in_dim, hidden, seed, device):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        net = nn.Sequential(nn.Linear(in_dim, hidden), PreActNorm(), nn.ELU(),
                            nn.Linear(hidden, hidden), PreActNorm(), nn.ELU(),
                            nn.Linear(hidden, hidden), PreActNorm(), nn.ELU(),
                            nn.Linear(hidden, hidden))
    return net.to(device).requires_grad_(False).eval()


class LearnableNovelty(Signal):
    name = "ln"

    def __init__(self, obs_dim, device, hidden=32, ridge=0.3, eta=1.0, reservoir_seed=1, eps=1e-8):
        self.D, self.H, self.ridge, self.eta, self.eps = obs_dim, hidden, ridge, eta, eps
        self.device = torch.device(device)
        self.phi = make_reservoir(obs_dim, hidden, reservoir_seed, device)
        self.tau = None

    @torch.no_grad()
    def calibrate(self, env, tau_bounds=(8, 48)):
        """Frozen statistics from uniform-random-action rollouts (their rl/calibrate.py). Rollouts
        start from uniformly random states rather than the fixed start: from the fixed start a
        random walk almost never moves the block, the block coordinates would get ~zero std, and
        per-coordinate standardisation would then blow any block motion up by ~1e8."""
        obs = [env.reset(random_start=True)]
        for _ in range(env.ep_len):
            u = 2 * torch.rand(env.E, env.act_dim, device=self.device, generator=env.gen) - 1
            obs.append(env.step(u)[0])
        X = torch.stack(obs)                                     # (T+1, E, D)
        # tau = 2 x the first lag at which the standardised RMS displacement reaches 1 state-std
        std = X.reshape(-1, self.D).std(0).clamp_min(1e-6)
        k_char = X.shape[0] - 1
        for k in range(1, X.shape[0]):
            rms = (((X[k:] - X[:-k]) / std) ** 2).mean().sqrt()
            if rms >= 1.0:
                k_char = k
                break
        self.tau = int(min(max(2 * k_char, tau_bounds[0]), tau_bounds[1]))
        tau, T1 = self.tau, X.shape[0]
        xs = X[:T1 - tau].reshape(-1, self.D)
        ys = torch.stack([X[t + 1:t + 1 + tau].transpose(0, 1).reshape(env.E, -1)
                          for t in range(T1 - tau)]).reshape(-1, tau * self.D)
        self.x_mu = xs.mean(0)
        self.x_inv = 1.0 / (xs.std(0, correction=0) + self.eps)
        h = self.phi((xs - self.x_mu) * self.x_inv)
        self.f_mu = h.mean(0)
        self.f_inv = 1.0 / (h.std(0, correction=0) * self.H ** 0.5 + self.eps)
        self.y_mu = ys.mean(0)                                   # unit target scale: y_inv = 1
        return {"tau": self.tau, "k_char": k_char, "n_pairs": xs.shape[0]}

    @torch.no_grad()
    def reward(self, ro):
        obs = ro["obs"]
        T1, E, D = obs.shape
        tau, H, M = self.tau, self.H, self.tau * D
        f = self.phi(((obs - self.x_mu) * self.x_inv).reshape(-1, D)).reshape(T1, E, H)
        f = (f - self.f_mu) * self.f_inv
        P = (torch.eye(H, device=obs.device) / self.ridge).repeat(E, 1, 1)
        W = torch.zeros(E, H, M, device=obs.device)
        eye = torch.eye(H, device=obs.device, dtype=torch.float64)
        S_prev = torch.zeros(E, device=obs.device, dtype=torch.float64)
        r = torch.zeros(T1 - 1, E, device=obs.device)
        for j in range(tau, T1):                                 # j = newest state; window anchored at t
            t = j - tau
            phi = f[t]
            y = obs[t + 1:j + 1].transpose(0, 1).reshape(E, M) - self.y_mu
            Pp = torch.einsum("ehk,ek->eh", P, phi)
            gain = Pp / (1.0 + (phi * Pp).sum(-1, keepdim=True))
            innov = y - torch.einsum("ehm,eh->em", W, phi)
            W = W + gain.unsqueeze(-1) * innov.unsqueeze(1)
            P = P - gain.unsqueeze(-1) * Pp.unsqueeze(1)
            P = 0.5 * (P + P.transpose(1, 2))
            Wd = W.double()
            L = torch.linalg.cholesky(eye + self.eta * Wd @ Wd.transpose(1, 2))
            S = L.diagonal(dim1=1, dim2=2).log().sum(-1) / math.log(2)   # = 1/2 log2 det(.)
            r[j - 1] = (S - S_prev).float()
            S_prev = S
        return r, {"S_episode": S_prev.float()}


# ------------------------------------------------------------------ oracle count
class CountOracle(Signal):
    name = "count"
    learns = True

    def __init__(self, env, bins=10):
        self.bins, self.env = bins, env
        self.N = torch.zeros(bins ** 4, device=env.device)

    def cell(self, s):
        e, b = self.env, self.bins
        ia = ((s[..., :2] - e.agent_lo) / (e.agent_hi - e.agent_lo) * b).long().clamp(0, b - 1)
        ib = ((s[..., 2:] - e.block_lo) / (e.block_hi - e.block_lo) * b).long().clamp(0, b - 1)
        return ((ia[..., 0] * b + ia[..., 1]) * b + ib[..., 0]) * b + ib[..., 1]

    def reward(self, ro):
        return 1.0 / torch.sqrt(self.N[self.cell(ro["state"][1:])] + 1.0), {}

    def update(self, ro):
        c = self.cell(ro["state"][1:]).reshape(-1)
        self.N.index_add_(0, c, torch.ones_like(c, dtype=self.N.dtype))
        return {}


def make_signal(name, env, device, args):
    if name == "none":
        return Signal()
    if name == "pred":
        return PredError(env.obs_dim, env.act_dim, device, lr=args.wm_lr, batch=args.wm_batch,
                         epochs=args.wm_epochs)
    if name in ("lp", "lps"):
        cls = LearningProgress if name == "lp" else SignedProgress
        return cls(env.obs_dim, env.act_dim, device, ema=args.lp_ema, lr=args.wm_lr,
                   batch=args.wm_batch, epochs=args.wm_epochs)
    if name == "ln":
        # reservoir + normalisation are anchored to a base seed, shared by every training seed
        # (as in their rl/calibrate.py)
        return LearnableNovelty(env.obs_dim, device, hidden=args.ln_hidden, ridge=args.ln_ridge,
                                eta=args.ln_eta, reservoir_seed=1)
    if name == "count":
        return CountOracle(env)
    raise ValueError(name)
