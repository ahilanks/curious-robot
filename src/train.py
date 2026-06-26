"""Curious Robot — from-scratch JEPA+SIGReg world model co-trained with SAC under
an intrinsic curiosity reward, on the MuJoCo SO-ARM101 (README is the spec).

Per decision step (action_block env steps), for every parallel env:
  1. encode z_t from (wrist image, proprio) with the current world model
  2. act a_t ~ pi(z_t) (SAC sample in sim training; deterministic mean tanh(MLP(z_t)) on
     hardware/eval); apply it (delta-target PD) over action_block env steps
  3. curiosity   r_cur = ||f(z_{t-H+1:t}, a_t) - z_{t+1}||^2     (1-step pred error)
     reward      r_t   = sum_k r_safe_k + lambda_cur * symlog(r_cur)
  4. store (o_t, q_t, qdot_t, a_t, r_t, o_{t+1}) with PER priority = r_cur
Periodically: co-train the WM (autoregressive MSE rollout loss + beta*SIGReg, with
an H_fwd curriculum), run SAC updates (PER), log to W&B, checkpoint to the HF Hub.

The `?` constants in the README (beta, lambda_cur, delta, ...) are CLI flags here with
working defaults, meant to be swept; they are intentionally NOT pinned in the README.
"""
import os
import platform
os.environ.setdefault("MUJOCO_GL", "glfw" if platform.system() == "Darwin" else "osmesa")

import argparse
import contextlib
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lewm.module import SIGReg                       # noqa: E402
from model.state_encoder import WorldModel, pred_dims_from_args  # noqa: E402
from src.probe import load_probe_hf                  # noqa: E402
from src.goal_explore import GoalArchive             # noqa: E402  (--goal-explore goal archive)
# Env backends are imported lazily in main() so the `hardware` backend does not require mujoco.

try:
    import wandb
except ImportError:
    wandb = None
try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def to_norm_pixel(px_uint8, device):
    """uint8 (...,H,W,3) -> normalized float (...,3,H,W)."""
    t = torch.as_tensor(np.ascontiguousarray(px_uint8), device=device)
    perm = list(range(t.ndim - 3)) + [t.ndim - 1, t.ndim - 3, t.ndim - 2]
    t = t.permute(*perm).float() / 255.0
    shp = [1] * (t.ndim - 3) + [3, 1, 1]
    return (t - IMAGENET_MEAN.to(device).view(shp)) / IMAGENET_STD.to(device).view(shp)


# ------------------------------ actor-critic (SAC: stochastic train, deterministic deploy)
class Actor(nn.Module):
    """Gaussian SAC policy with a deterministic-mean deployment path.

    `forward(z)` returns the DETERMINISTIC action tanh(mean(z)) — used at eval and on
    hardware, so every deployed action is reproducible. `sample(z)` returns a squashed
    Gaussian sample (a, logp, tanh(mean)) — used for sim-training collection and the SAC
    entropy term. Re-added 2026-06-14: the entropy bonus keeps the policy off the freeze
    attractor that the 2026-06-12 deterministic actor parked on; the learned mean is what
    deploys, the stochasticity lives only in training. log_std is a learned, state-dependent
    head; deterministic-era checkpoints (no log_std) load via load_actor_state and keep its
    fresh init (deployment never reads it)."""

    def __init__(self, z_dim, a_dim, hidden=256, log_std_range=(-5, 2), goal_dim=0):
        super().__init__()
        self.goal_dim = goal_dim          # >0 -> goal-conditioned pi(a | z, z*) (--goal-explore)
        self.trunk = nn.Sequential(nn.Linear(z_dim + goal_dim, hidden), nn.ReLU(),
                                   nn.Linear(hidden, hidden), nn.ReLU())
        self.mean = nn.Linear(hidden, a_dim)
        self.log_std = nn.Linear(hidden, a_dim)
        self.lo, self.hi = log_std_range

    def _h(self, z, zstar):
        return self.trunk(torch.cat([z, zstar], -1) if zstar is not None else z)

    def forward(self, z, zstar=None):
        """Deterministic action (deployed / eval / goal-conditioned) = tanh(mean(z[,z*]))."""
        return torch.tanh(self.mean(self._h(z, zstar)))

    def sample(self, z, zstar=None):
        """Stochastic action for training: (a, logp, deterministic_mean). Reparameterized
        squash with the standard tanh log-prob correction."""
        h = self._h(z, zstar)
        mean, log_std = self.mean(h), self.log_std(h).clamp(self.lo, self.hi)
        normal = torch.distributions.Normal(mean, log_std.exp())
        u = normal.rsample()
        a = torch.tanh(u)
        logp = (normal.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)).sum(-1, keepdim=True)
        return a, logp, torch.tanh(mean)


def load_actor_state(actor, sd):
    """Load an actor state_dict, tolerating checkpoints from either policy era: stochastic
    SAC (safe15: has log_std) loads it; the 2026-06-12..06-14 deterministic actor (no
    log_std) leaves the re-added log_std head at its fresh init. trunk/mean shapes are
    identical across eras, so strict=False only ever skips the log_std head."""
    actor.load_state_dict(sd, strict=False)


class TwinQ(nn.Module):
    """Twin Q critics. n_out=1 is the scalar critic; n_out=K (--multihead-q) gives one
    head per weighted reward component, each trained on its own per-component TD target
    with the SHARED next action — the actor still maximizes the sum over heads, so the
    optimum matches the scalar critic; the heads exist to make the value decomposition
    inspectable (which reward component is steering the policy)."""

    def __init__(self, z_dim, a_dim, hidden=256, n_out=1, goal_dim=0):
        super().__init__()
        self.goal_dim = goal_dim          # >0 -> goal-conditioned Q(z, a, z*) (--goal-explore)
        mk = lambda: nn.Sequential(nn.Linear(z_dim + a_dim + goal_dim, hidden), nn.ReLU(),
                                   nn.Linear(hidden, hidden), nn.ReLU(),
                                   nn.Linear(hidden, n_out))
        self.q1, self.q2 = mk(), mk()

    def forward(self, z, a, zstar=None):
        za = torch.cat([z, a, zstar], -1) if zstar is not None else torch.cat([z, a], -1)
        return self.q1(za), self.q2(za)


class RunningMeanStd:
    """Welford running mean/std for RND observation normalization -- the detail that makes RND
    work, since the frozen random target's outputs are only meaningful when its inputs are
    stably scaled. shape=() keeps whole-tensor (scalar) stats for the 84x84 frame; shape=(d,)
    keeps per-dim stats for proprio (whose pos/vel/torque dims live on very different scales).
    Updated on the live obs stream; the count grows unboundedly so the scale settles to the
    (early) visited distribution and then barely moves (Burda et al. 2018)."""
    def __init__(self, shape, device):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = 1e-4

    @torch.no_grad()
    def update(self, x):                                  # x: (..., *shape) -> stats over leading dims
        axes = tuple(range(x.ndim - self.mean.ndim))
        b_mean, b_var = x.mean(dim=axes), x.var(dim=axes, unbiased=False)
        b_count = x.numel() / max(self.mean.numel(), 1)
        delta = b_mean - self.mean
        tot = self.count + b_count
        self.mean = self.mean + delta * b_count / tot
        m2 = self.var * self.count + b_var * b_count + delta.pow(2) * self.count * b_count / tot
        self.var, self.count = m2 / tot, tot

    @torch.no_grad()
    def norm(self, x):
        return (x - self.mean) / (self.var.sqrt() + 1e-8)


class RNDObsNet(nn.Module):
    """RND target/predictor over the RAW observation (downsampled wrist image + proprio), NOT
    the co-trained latent z. The raw obs is a STABLE input space, so the frozen random target
    is a fixed goalpost and novelty tracks genuine state coverage instead of drifting with the
    encoder. Atari-style conv stack on an 84x84 grayscale frame (Burda et al. 2018) plus a
    small proprio MLP, concatenated into a linear head."""
    def __init__(self, prop_dim, out_dim=128, hidden=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 8, stride=4), nn.ReLU(),     # 84 -> 20
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),    # 20 -> 9
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),    # 9 -> 7
            nn.Flatten())
        self.prop = nn.Sequential(nn.Linear(prop_dim, hidden), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(64 * 7 * 7 + hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, out_dim))

    def forward(self, img84, prop):
        return self.head(torch.cat([self.conv(img84), self.prop(prop)], dim=-1))


# ------------------------------------------------------------------------ buffer
class ReplayBuffer:
    """Per-env ring buffers (so WM windows stay inside one env's contiguous stream)
    with global PER sampling for SAC. Stores raw obs; z is encoded fresh from the
    current WM at sample time, so the co-trained encoder can drift without staleness.
    PER priority = curiosity surprise (raw 1-step prediction error)."""

    def __init__(self, n_envs, cap_per_env, img_hw, a_dim, prop_dim, device, n_comp=0,
                 goal_explore=False):
        self.n_envs, self.C, self.device = n_envs, cap_per_env, device
        s = (n_envs, cap_per_env)
        self.pixels = np.zeros((*s, img_hw, img_hw, 3), np.uint8)
        self.proprio = np.zeros((*s, prop_dim), np.float32)
        self.action = np.zeros((*s, a_dim), np.float32)
        self.r = np.zeros(s, np.float32)
        self.rc = np.zeros((*s, n_comp), np.float32) if n_comp else None   # per-component rewards (--multihead-q)
        self.d = np.zeros(s, np.float32)
        self.is_start = np.zeros(s, bool)
        self.prio = np.zeros(s, np.float64)
        self.head = np.zeros(n_envs, np.int64)
        self.count = np.zeros(n_envs, np.int64)
        # --goal-explore: per-transition PURSUED goal obs o* + an episode id for HER future-relabel
        self.goal_explore = goal_explore
        if goal_explore:
            self.goal_px = np.zeros((*s, img_hw, img_hw, 3), np.uint8)
            self.goal_prop = np.zeros((*s, prop_dim), np.float32)
            self.goal_valid = np.zeros(s, bool)       # True = a REAL archive goal was pursued (not self-goal)
            self.ep_id = np.zeros(s, np.int64)        # shared id for all transitions of one episode
            self.cur_ep = np.zeros(n_envs, np.int64)

    def add(self, pixels, proprio, action, r, d, is_start, prio, rc=None,
            goal_px=None, goal_prop=None, goal_valid=None):
        for e in range(self.n_envs):
            i = self.head[e]
            self.pixels[e, i] = pixels[e]
            self.proprio[e, i] = proprio[e]
            self.action[e, i] = action[e]
            self.r[e, i] = r[e]
            if self.rc is not None:
                self.rc[e, i] = rc[e]
            self.d[e, i] = d[e]
            self.is_start[e, i] = is_start[e]
            self.prio[e, i] = prio[e]
            if self.goal_explore:
                if is_start[e]:                       # is_start marks an episode's FIRST step
                    self.cur_ep[e] += 1
                self.ep_id[e, i] = self.cur_ep[e]
                if goal_px is not None:
                    self.goal_px[e, i] = goal_px[e]
                    self.goal_prop[e, i] = goal_prop[e]
                if goal_valid is not None:
                    self.goal_valid[e, i] = goal_valid[e]
            self.head[e] = (i + 1) % self.C
            self.count[e] = min(self.count[e] + 1, self.C)

    @property
    def total(self):
        return int(self.count.sum())

    def _valid_pairs(self):
        """(e, i) transitions whose next slot (e, i+1) is written and valid."""
        es, isx = [], []
        for e in range(self.n_envs):
            n = int(self.count[e])
            if n < 2:
                continue
            if n == self.C:
                forbid = (int(self.head[e]) - 1) % self.C   # newest, no next yet
                idx = np.delete(np.arange(self.C), forbid)
            else:
                idx = np.arange(n - 1)
            es.append(np.full(len(idx), e)); isx.append(idx)
        if not es:
            return None
        return np.concatenate(es), np.concatenate(isx)

    def _future_goal_idx(self, e, i):
        """HER 'future' strategy: a random LATER transition in the SAME episode as (e, i),
        respecting the ring's time order (oldest..newest) and episode (ep_id) boundaries.
        Returns a ring index into env e, or None if (e, i) is the last step of its episode."""
        n = int(self.count[e])
        if n == 0:
            return None
        if n < self.C:
            order = np.arange(n)                                       # no wrap: index == time order
        else:
            order = (np.arange(self.C) + int(self.head[e])) % self.C   # wrapped: oldest..newest
        pos = np.where(order == i)[0]
        if len(pos) == 0:
            return None
        fut = order[pos[0] + 1:]                                       # strictly later in time
        if len(fut) == 0:
            return None
        same = fut[self.ep_id[e, fut] == self.ep_id[e, i]]            # same-episode only
        if len(same) == 0:
            return None
        return int(np.random.choice(same))

    def sample_future_goal(self, env_idx, k):
        """Controller goal (--goal-select future): per env, an obs it ACHIEVED `k` forward steps
        after a recent point in its OWN current episode -- reachable by forward dynamics and on the
        agent's manifold (HER 'future' applied to action selection, mirroring LeWM goal_offset).
        Returns (px[L,...], prop[L,...], valid[L]); valid=False where the episode is too short yet."""
        L = len(env_idx)
        px = np.zeros((L, *self.pixels.shape[2:]), np.uint8)
        pr = np.zeros((L, self.proprio.shape[2]), np.float32)
        valid = np.zeros(L, bool)
        if not self.goal_explore:
            return px, pr, valid
        for slot in range(L):
            e = int(env_idx[slot]); n = int(self.count[e])
            if n < 3:
                continue
            order = ((np.arange(self.C) + int(self.head[e])) % self.C) if n == self.C else np.arange(n)
            same = order[self.ep_id[e, order] == self.ep_id[e, order[-1]]]   # current episode, oldest..newest
            if len(same) < 3:
                continue
            kk = min(int(k), len(same) - 1)                                  # clamp to what the episode holds
            ap = np.random.randint(0, len(same) - kk)                        # anchor with kk forward room
            gi = int(same[ap + kk])                                          # achieved obs kk steps later
            px[slot] = self.pixels[e, gi]; pr[slot] = self.proprio[e, gi]; valid[slot] = True
        return px, pr, valid

    def sample_recent_goal(self, env_idx, k):
        """Controller goal (--goal-select recent): per env, the obs the agent visited EXACTLY `k`
        decisions ago in its CURRENT episode -- a CONTROLLED-small-distance, definitely-on-manifold
        goal whose latent reach_gap ~ k*step_jump. Sweeping k maps the planner's reachable radius
        (the threshold between the reach=1 at gap~0 and reach=0 at gap~8 regimes). Returns (px,prop,valid)."""
        L = len(env_idx)
        px = np.zeros((L, *self.pixels.shape[2:]), np.uint8)
        pr = np.zeros((L, self.proprio.shape[2]), np.float32)
        valid = np.zeros(L, bool)
        if not self.goal_explore:
            return px, pr, valid
        kk = max(int(k), 1)
        for slot in range(L):
            e = int(env_idx[slot]); n = int(self.count[e])
            if n < kk + 2:
                continue
            order = ((np.arange(self.C) + int(self.head[e])) % self.C) if n == self.C else np.arange(n)
            same = order[self.ep_id[e, order] == self.ep_id[e, order[-1]]]   # current episode, oldest..newest
            if len(same) < kk + 1:
                continue
            gi = int(same[-1 - kk])                                          # the state kk decisions BEFORE now
            px[slot] = self.pixels[e, gi]; pr[slot] = self.proprio[e, gi]; valid[slot] = True
        return px, pr, valid

    def sample_candidates(self, n):
        """--goal-select highmse_under_d: up to n WITHIN-episode transitions (source o_t, action a_t,
        outcome o_{t+1}) as MSE-buffer goal candidates -- re-scored by the CURRENT WM and filtered by
        latent distance to z_now in refresh_goals. Returns (spx, sprop, sact, gpx, gprop) numpy, or None."""
        vp = self._valid_pairs()
        if vp is None:
            return None
        e_all, i_all = vp
        ni_all = (i_all + 1) % self.C
        same = self.ep_id[e_all, i_all] == self.ep_id[e_all, ni_all]   # real successor (no episode/reset crossing)
        e_all, i_all, ni_all = e_all[same], i_all[same], ni_all[same]
        if len(e_all) == 0:
            return None
        sel = np.random.choice(len(e_all), size=min(int(n), len(e_all)), replace=False)
        e, i, ni = e_all[sel], i_all[sel], ni_all[sel]
        return (self.pixels[e, i], self.proprio[e, i], self.action[e, i],
                self.pixels[e, ni], self.proprio[e, ni])

    def sample_sac(self, batch, per_alpha, per_beta, her_frac=0.0):
        vp = self._valid_pairs()
        if vp is None or len(vp[0]) < batch:
            return None
        e_all, i_all = vp
        pr = (self.prio[e_all, i_all] + 1e-6) ** per_alpha
        probs = pr / pr.sum()
        sel = np.random.choice(len(e_all), size=batch, p=probs)
        e, i = e_all[sel], i_all[sel]
        ni = (i + 1) % self.C
        w = (len(e_all) * probs[sel]) ** (-per_beta)
        w = (w / w.max()).astype(np.float32)
        t = lambda x: torch.as_tensor(x, device=self.device)
        out = {
            "px": self.pixels[e, i], "prop": self.proprio[e, i],
            "px_n": self.pixels[e, ni], "prop_n": self.proprio[e, ni],
            "a": t(self.action[e, i]), "r": t(self.r[e, i])[:, None],
            "d": t(self.d[e, i])[:, None], "w": t(w)[:, None],
            "e": e, "i": i,                          # for optional TD-error priority writeback
        }
        if self.rc is not None:
            out["rc"] = t(self.rc[e, i])             # (B, K) per-component rewards
        if self.goal_explore:                        # raw pursued goal o*; HER-relabel a fraction
            gpx = self.goal_px[e, i].copy()
            gprop = self.goal_prop[e, i].copy()
            gvalid = self.goal_valid[e, i].astype(np.float32).copy()
            if her_frac > 0:
                for k in np.where(np.random.rand(batch) < her_frac)[0]:
                    j = self._future_goal_idx(int(e[k]), int(i[k]))
                    if j is not None:                # relabel goal -> an achieved future obs (always valid)
                        gpx[k] = self.pixels[e[k], j]
                        gprop[k] = self.proprio[e[k], j]
                        gvalid[k] = 1.0
            out["goal_px"], out["goal_prop"] = gpx, gprop
            out["goal_valid"] = t(gvalid)[:, None]   # (B,1) 1=real goal (archive/HER), 0=self-goal fallback
        return out

    def update_priorities(self, e, i, prio):
        """Overwrite priorities of sampled transitions (used by --per-priority td)."""
        self.prio[e, i] = np.maximum(np.asarray(prio, np.float64), 1e-6)

    def sample_wm(self, batch, T, mode="uniform", per_alpha=0.6):
        """Sample (px, proprio, action) windows of length T contiguous within one env
        (no episode-start crossing, no ring-seam crossing). `mode` selects how each window's
        START is drawn (its predicted/target frame sits at s+T-1):
          'uniform'   -- flat over valid starts (default; original behaviour).
          'curiosity' -- P ~ prio[target]^per_alpha: high one-step-MSE windows oversampled
                         (the surprise signal SAC's PER uses -> trains the WM where it is
                         worst, i.e. the goal regions).
          'recency'   -- P decays exponentially with how long ago the window was collected
                         (half-life = 25% of the env's fill), favouring fresh data."""
        starts = []
        for e in range(self.n_envs):
            n = int(self.count[e])
            if n < T + 1:
                continue
            if n < self.C:
                lo, hi, head = 0, n - T, -1            # linear region, no wrap
            else:
                lo, hi, head = 0, self.C - T, int(self.head[e])
            ndraw = 8 * batch // max(self.n_envs, 1) + 4
            if mode == "uniform":
                cands = np.random.randint(lo, hi + 1, size=ndraw)
            else:
                pos = np.arange(lo, hi + 1)
                tgt = pos + T - 1                                  # predicted (surprising) frame slot
                if mode == "curiosity":
                    w = np.power(self.prio[e, tgt], per_alpha) + 1e-6
                else:                                              # recency
                    rank = (int(self.head[e]) - 1 - tgt) % self.C  # 0 = newest filled slot
                    w = np.power(0.5, rank / max(n * 0.25, 1.0))
                cands = np.random.choice(pos, size=ndraw, p=w / w.sum())
            for s in cands:
                s = int(s)
                if head >= 0 and s < head <= s + T - 1:   # straddles the time seam
                    continue
                if self.is_start[e, s + 1:s + T].any():    # crosses an episode reset
                    continue
                starts.append((e, s))
                if len(starts) >= batch:
                    break
            if len(starts) >= batch:
                break
        if len(starts) < max(batch // 4, 1):
            return None
        e = np.array([p[0] for p in starts]); s = np.array([p[1] for p in starts])
        idx = s[:, None] + np.arange(T)[None, :]
        ee = e[:, None].repeat(T, 1)
        return (self.pixels[ee, idx], self.proprio[ee, idx],
                torch.as_tensor(self.action[ee, idx], device=self.device))


REWARD_COMPONENTS = ("cur", "safe", "rate", "energy")   # --multihead-q head order


def scrub_torque_obs(obs, n_dof):
    """--no-torque-obs: zero the u^app slice of proprio in place (obs -> [q, qd, 0]).
    Keeps the proprio/encoder shapes (old ckpts still load) while removing the channel
    that is a near-constant saturated sign bit on hardware (2026-06-11: 96-97% of
    joint-samples pegged at +/-3.35 under the kp-law recompute) and mostly saturated
    in sim too — the main sim->real obs-distribution mismatch."""
    obs["proprio"][..., 2 * n_dof:3 * n_dof] = 0.0
    return obs


# ------------------------------------------------------------------- WM helpers
@torch.no_grad()
def encode_obs(wm, px_uint8, proprio_np, device):
    return wm.encode(to_norm_pixel(px_uint8, device),
                     torch.as_tensor(proprio_np, device=device).float())


@torch.no_grad()
def curiosity_reward(wm, hist_z, hist_a, z_next):
    """r_cur = mean_d (f(z_{t-H+1:t}, a_t)[-1] - z_{t+1})^2, per env: the PER-DIM MEAN
    squared 1-step prediction error (same normalization as the WM loss). Keeps r_cur
    O(0.1-1) so symlog operates in its sensitive region (not the saturated tail of the
    d_z-summed version) -> a more discriminative curiosity reward. Returns (B,)."""
    z_ctx = hist_z.transpose(0, 1)                       # (B, H, D)
    a_emb = wm.action_encoder(hist_a.transpose(0, 1))    # (B, H, A_emb)
    pred = wm.predict(z_ctx, a_emb)[:, -1]               # (B, D)
    return (pred - z_next).pow(2).mean(-1)


@torch.no_grad()
def score_obs_mse(wm, gpx, gprop, spx, sprop, sact, device, H):
    """FAITHFUL one-step WM MSE for a batch of stored (source o_t, action a, goal o_{t+1})
    triples under the CURRENT wm: predict z_{t+1} from (z_t, a) and compare to the REAL
    encode(o_{t+1}). The H-frame backward context is fabricated by repeating z_t (the
    canonical reset-seeding), since the real rolling history is gone once an obs is archived.
    Same per-dim-mean normalization as curiosity_reward, so it is comparable to the r_cur
    captured at insert time -> a goal whose re-measured MSE has fallen has been mastered by
    the WM and can be evicted. Returns (B,) numpy. (--goal-explore archive re-measure.)"""
    z_src = encode_obs(wm, spx, sprop, device)               # (B, D)
    z_goal = encode_obs(wm, gpx, gprop, device)              # (B, D)
    z_ctx = z_src.unsqueeze(1).repeat(1, H, 1)               # (B, H, D)
    a = torch.as_tensor(sact, device=device).float().unsqueeze(1).repeat(1, H, 1)  # (B,H,a_dim)
    a_emb = wm.action_encoder(a)                             # (B, H, A_emb)
    pred = wm.predict(z_ctx, a_emb)[:, -1]                   # (B, D)
    return (pred - z_goal).pow(2).mean(-1).cpu().numpy()


@torch.no_grad()
def cem_plan(wm, hist_z, hist_a, z_goal, K, iters, elite, init_std, horizon, device, diag=None, gamma=0.0,
             min_std=0.0, mppi_temp=0.0):
    """CEM planner in LATENT space -- a faithful port of LeWM's stable_worldmodel.solver.CEMSolver
    (+ JEPA.rollout/criterion). Per replan: sample H-step action SEQUENCES from a per-step Gaussian,
    FORCE candidate 0 = the current mean (LeWM's candidates[:,0]=mean), roll each AUTOREGRESSIVELY
    through the fixed world model, score by the TERMINAL latent cost ||zhat_H - z_goal||^2 SUMMED over
    latent dims (= LeWM's F.mse_loss(...).sum), keep the `elite` lowest-cost sequences, and refit
    mean=elite.mean, std=elite.std. EXACTLY as LeWM: init std = var_scale, NO min-std floor, and NO
    action clamp inside the optimization (the candidate distribution is unbounded; the executed action
    is clamped by the caller). Returns the FULL planned sequence (n_envs, H, a_dim): the caller executes
    all H open-loop (LeWM receding_horizon == horizon) and re-plans when the buffer empties. The rollout
    is seeded from the last `Hb` REAL latents+actions (this codebase's choice; LeWM-cube uses
    history_len=1). No learned policy and NO reach reward -- goal-reaching is PURE planning."""
    Hb, n, D = hist_z.shape                                           # Hb = WM backward context (history_size)
    a_dim = hist_a.shape[-1]
    T = max(int(horizon), 1)
    z0 = hist_z.transpose(0, 1).unsqueeze(1).expand(n, K, Hb, D).reshape(n * K, Hb, D)        # latent history
    a0 = hist_a.transpose(0, 1).unsqueeze(1).expand(n, K, Hb, a_dim).reshape(n * K, Hb, a_dim)  # action history
    zg = z_goal.unsqueeze(1).expand(n, K, D).reshape(n * K, D)
    mu = torch.zeros(n, T, a_dim, device=device)
    std = torch.full((n, T, a_dim), init_std, device=device)         # LeWM: var = var_scale, used as the std
    # bf16 autocast on the rollout: the WM forwards run on tensor cores (~2-3x), and bf16 keeps
    # fp32 exponent range so an inference rollout is numerically safe. mu/std/topk stay fp32
    # (autocast leaves reductions in fp32). No-op off-CUDA.
    amp = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if device.type == "cuda" else contextlib.nullcontext())
    predict = getattr(wm, "predict_eager", wm.predict)   # EAGER: torch.compile + bf16 autocast -> NaN rollout
    with amp:
        for it in range(iters):
            seq = mu.unsqueeze(1) + std.unsqueeze(1) * torch.randn(n, K, T, a_dim, device=device)
            seq[:, 0] = mu                                            # LeWM: force candidate 0 = current mean
            sf = seq.reshape(n * K, T, a_dim)
            z_seq, a_seq = z0, a0                                         # (n*K, Hb, .) growing rollout buffers
            step_costs = []                                              # per-rollout-step ||z_{t+h+1} - z*||^2
            for h in range(T):                                           # autoregressive WM rollout
                a_seq = torch.cat([a_seq, sf[:, h:h + 1]], dim=1)        # apply candidate action a_{t+h}
                znext = predict(z_seq[:, -Hb:], wm.action_encoder(a_seq[:, -Hb:]))[:, -1:]
                z_seq = torch.cat([z_seq, znext], dim=1)                 # append predicted z_{t+h+1}
                step_costs.append((znext[:, -1] - zg).pow(2).sum(-1))    # (n*K,) running cost at step h+1
            # cost = discounted running sum sum_h gamma^(T-1-h) ||z_h - z*||^2 (terminal weight 1).
            # gamma=0 -> terminal-only, BYTE-IDENTICAL to the LeWM objective; gamma in (0,1] shapes the
            # path so CEM gets gradient toward the goal even when the H-step endpoint is out of reach.
            if gamma > 0:
                cost = sum((gamma ** (T - 1 - h)) * step_costs[h] for h in range(T)).reshape(n, K)
            else:
                cost = step_costs[-1].reshape(n, K)                      # terminal ||.||^2 (LeWM: SUM over dims)
            if diag is not None and it == 0:
                # action-sensitivity probe on the FIRST (widest, std=init_std) candidate batch:
                #   cost_cv       = spread of terminal cost across candidates (~0 => CEM has no signal to optimize)
                #   z_term_spread = RMS spread of the predicted terminal latent across candidates, i.e. how far
                #                   different ACTIONS move the endpoint (~0 => WM rollout ignores the action)
                #   reach_gap     = current ||z - z*||; if z_term_spread << reach_gap the goal is unreachable
                #                   no matter the action (the WM can't move the latent far enough).
                with torch.no_grad():
                    c = cost.float()
                    fin = torch.isfinite(c)                              # wide candidates can DIVERGE (inf/nan)
                    diag["finite_frac"] = float(fin.float().mean())      # frac of candidates the WM rolled out finitely
                    cm = torch.where(fin, c, torch.full_like(c, float("nan")))
                    mean = cm.nanmean(1)
                    std = (cm - mean[:, None]).pow(2).nanmean(1).sqrt()
                    diag["cost_cv"] = float((std / (mean.abs() + 1e-9)).nanmean())
                    zt = z_seq[:, -1].float().reshape(n, K, D)
                    zt = torch.where(torch.isfinite(zt), zt, torch.full_like(zt, float("nan")))
                    zvar = (zt - zt.nanmean(1, keepdim=True)).pow(2).nanmean(1).clamp_min(0)   # (n, D)
                    diag["z_term_spread"] = float(zvar.sum(-1).sqrt().nanmean())
                    z_now = z0.reshape(n, K, Hb, D)[:, 0, -1]
                    diag["reach_gap"] = float((z_now - z_goal).pow(2).sum(-1).clamp_min(0).sqrt().mean())
            if diag is not None and it == iters - 1:
                # CONVERGED-plan probe (narrow std): distinguishes 'reachable set too small' from
                #   'planner can't aim'. min_cand_to_goal = best candidate endpoint distance to the goal
                #   after CEM has refit -- if this stays ~reach_gap, no plan can get closer (size/WM-limited);
                #   if it drops well below reach_gap yet realized dist stays high, the limit is open-loop /
                #   direction (the WM thinks it reaches but execution doesn't). endpoint_disp = how far the
                #   mean converged endpoint sits from z_now (is the plan even committing to move?).
                with torch.no_grad():
                    ztf = z_seq[:, -1].float().reshape(n, K, D)
                    finite = torch.isfinite(ztf).all(-1)                            # (n, K)
                    d2g = (ztf - z_goal.unsqueeze(1)).pow(2).sum(-1).clamp_min(0).sqrt()  # (n, K)
                    d2g = torch.where(finite, d2g, torch.full_like(d2g, float("inf")))
                    diag["min_cand_to_goal"] = float(d2g.min(1).values.mean())
                    ztm = torch.where(finite.unsqueeze(-1), ztf, torch.full_like(ztf, float("nan"))).nanmean(1)  # (n, D)
                    z_now2 = z0.reshape(n, K, Hb, D)[:, 0, -1]
                    diag["endpoint_disp"] = float((ztm - z_now2).pow(2).sum(-1).clamp_min(0).sqrt().nanmean())
            if mppi_temp > 0:
                # MPPI / information-theoretic SOFT update: weight ALL candidates by exp(-cost/temp)
                # instead of CEM's hard top-k elites. Lower temp -> greedier (approaches argmin = max
                # model exploitation); higher temp -> softer averaging (less exploitation of the model's
                # optimistic corner). LeWM CEM == hard elites (mppi_temp=0).
                c = cost.float()
                c = torch.where(torch.isfinite(c), c, torch.full_like(c, float("inf")))
                cmin = c.min(1, keepdim=True).values                     # subtract min for numerical stability
                w = torch.softmax(-(c - cmin) / mppi_temp, dim=1).to(seq.dtype)   # (n, K)
                mu = (w[:, :, None, None] * seq).sum(1)                   # (n, T, a_dim) weighted mean
                var = (w[:, :, None, None] * (seq - mu.unsqueeze(1)).pow(2)).sum(1)
                std = var.clamp_min(0).sqrt()
            else:
                idx = cost.topk(elite, largest=False, dim=1).indices
                el = torch.gather(seq, 1, idx[:, :, None, None].expand(n, elite, T, a_dim))
                mu = el.mean(1)
                std = el.std(1)                                          # LeWM: NO min-std floor (unless --cem-min-std)
            if min_std > 0:
                std = std.clamp_min(min_std)                             # floor to keep exploration / curb std-collapse
    return mu                                                            # full H-step plan (open-loop receding horizon)


@torch.no_grad()
def to_rnd_obs(px_uint8, prop_np, img_rms, prop_rms, device, update=False):
    """Raw obs -> RND inputs. uint8 (...,H,W,3) wrist image -> normalized (B,1,84,84) grayscale
    (luminance, area-downsampled to the classic RND frame); proprio -> normalized (B,P). Both
    standardized by a running mean/std and clipped +-5. update=True on the live reward stream
    advances the obs-norm stats; update=False re-scores buffer samples for predictor training
    against those same (slowly-frozen) stats."""
    t = torch.as_tensor(np.ascontiguousarray(px_uint8), device=device).float() / 255.0
    gray = (t * torch.tensor([0.299, 0.587, 0.114], device=device)).sum(-1)         # (...,H,W) luminance
    gray = gray.reshape(-1, 1, gray.shape[-2], gray.shape[-1])                       # (B,1,H,W)
    # area-downsample to the classic 84x84 RND frame. MPS's adaptive_avg_pool2d (which mode="area"
    # dispatches to) rejects non-divisible input sizes (224 % 84 != 0), so on MPS do the pool on CPU
    # (device-identical result); CUDA/CPU run it natively.
    if gray.device.type == "mps":
        img = F.interpolate(gray.cpu(), size=(84, 84), mode="area").to(device)      # (B,1,84,84)
    else:
        img = F.interpolate(gray, size=(84, 84), mode="area")                       # (B,1,84,84)
    prop = torch.as_tensor(np.ascontiguousarray(prop_np), device=device).float().reshape(img.shape[0], -1)
    if update:
        img_rms.update(img); prop_rms.update(prop)
    return img_rms.norm(img).clamp(-5, 5), prop_rms.norm(prop).clamp(-5, 5)


@torch.no_grad()
def knn_state_entropy_reward(z, buf, k):
    """Coverage / particle state-entropy reward (RE3/APT-style): for each latent z_t,
    reward = log(1 + mean distance to its k nearest neighbours among a buffer of recent
    visited latents). It pays to be FAR from the visited manifold -> the policy is pushed
    to low-density (frontier) regions -> the visitation manifold inflates toward uniform
    coverage (max state entropy). Unlike prediction-error curiosity this measures novelty
    (distance) DIRECTLY, so it is immune to the noisy-TV problem and doesn't vanish as the
    WM learns. z:(B,D), buf:(M,D) -> (B,). Returns 0 until the buffer holds >k latents.
    Caveat: buf latents are encoded by an earlier (co-trained) encoder, so keep buf recent."""
    if buf.shape[0] <= k:
        return torch.zeros(z.shape[0], device=z.device)
    d = torch.cdist(z, buf)                                   # (B, M) pairwise latent dist
    kth = d.topk(k, dim=1, largest=False).values             # k nearest per row
    return torch.log1p(kth.mean(dim=1))                       # (B,)


@torch.no_grad()
def rnd_novelty_reward(rnd_pred, rnd_target, img84, prop):
    """RND novelty (Burda 2018): squared error of a trained predictor against a FROZEN
    randomly-initialised target, evaluated on the RAW obs (downsampled wrist image + proprio),
    NOT the latent z. High for obs unlike those visited -> 0 as a region is covered. The target
    is deterministic (immune to the noisy-TV problem) AND the input is the stable raw obs, so
    the target is a fixed goalpost the co-trained encoder can't move. (B,1,84,84),(B,P) -> (B,)."""
    return (rnd_pred(img84, prop) - rnd_target(img84, prop)).pow(2).mean(-1)


def wm_update(wm, sigreg, opt, batch, H_bwd, h, gamma_wm, beta, device, pertimestep=False, lambda_slow=0.0):
    """One AdamW step on L_wm = discounted plain-MSE autoregressive rollout + beta*SIGReg
    (LeWM-style: mean squared error per step over batch+feature dims, no symlog)."""
    px, prop, ac = batch
    B, T = px.shape[:2]
    px_n = to_norm_pixel(px, device).reshape(B * T, 3, px.shape[2], px.shape[3])
    prop_t = torch.as_tensor(prop, device=device).float().reshape(B * T, -1)
    emb = wm.encode(px_n, prop_t).reshape(B, T, -1)       # (B,T,D) WITH grad
    a_emb = wm.action_encoder(ac)
    ctx_z, ctx_a = emb[:, :H_bwd], a_emb[:, :H_bwd]
    cur = wm.predict(ctx_z, ctx_a)                         # cur[:, -1] = zhat_{t+1}
    roll, roll_a = ctx_z, ctx_a
    pred_loss, wsum = 0.0, 0.0
    for k in range(1, h + 1):
        zhat = cur[:, -1]
        z_k = emb[:, H_bwd - 1 + k]                        # real z_{t+k}
        g = gamma_wm ** k
        mse = (zhat - z_k).pow(2).mean()                   # LeWM plain MSE over batch+feature dims (no symlog)
        pred_loss = pred_loss + g * mse
        wsum += g
        if k < h:
            roll = torch.cat([roll[:, 1:], zhat.unsqueeze(1)], 1)
            roll_a = torch.cat([roll_a[:, 1:], a_emb[:, H_bwd - 1 + k].unsqueeze(1)], 1)
            cur = wm.predict(roll, roll_a)
    pred_loss = pred_loss / wsum
    with torch.no_grad():   # baselines on the SAME discounted h-step schedule (same MSE metric)
        z_last = emb[:, H_bwd - 1]
        idl = sum((gamma_wm ** k) * (z_last - emb[:, H_bwd - 1 + k]).pow(2).mean()
                  for k in range(1, h + 1)) / wsum                       # persistence (predict z_t)
        # constant-mean baseline: predict the batch-mean latent for each target step. pred_loss
        # ~ this => the predictor collapsed to the mean (learned nothing input-dependent); the gap
        # between idl (persistence) and mbl localizes WHICH trivial solution it fell into.
        mbl = sum((gamma_wm ** k)
                  * (emb[:, H_bwd - 1 + k].mean(0, keepdim=True) - emb[:, H_bwd - 1 + k]).pow(2).mean()
                  for k in range(1, h + 1)) / wsum
    # SIGReg over the FULL rollout window pooled into ONE sample axis: (B,T,D) -> (1, B*T, D).
    # B*T (>=512 at the default 128 batch) > D=256, so the isotropy test spans the whole latent
    # space each step; the old per-timestep (T,B,D) view fed it B=128<256 samples, leaving >=128
    # latent directions unconstrained every update. The Epps-Pulley statistic already scales by n
    # (the *proj.size(-2) in SIGReg), which makes it sample-count-invariant under the isotropic
    # null -- so pooling needs NO beta retune (matches the pre-pool magnitude when z is healthy)
    # and, because the *n factor tracks the (systematic) non-Gaussianity of a collapsed z, it
    # pushes ~T x harder precisely when rank is low. No /T: that would just weaken SIGReg ~T-fold.
    sig = (sigreg(emb.transpose(0, 1))                  # LeWM per-timestep (T,B,D): isotropize each step's
           if pertimestep else                          # cross-trajectory batch; never pools consecutive frames
           sigreg(emb.reshape(1, B * T, -1)))           # default: pool the rollout window (B*T>D for the rank test)
    dz = (emb[:, 1:] - emb[:, :-1]).pow(2).mean()       # slowness: mean sq per-step latent jump (real frames)
    loss = pred_loss + beta * sig + lambda_slow * dz    # SIGReg anchors against the dz->0 collapse
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in wm.parameters() if p.requires_grad], 1.0)
    opt.step()
    return float(pred_loss.item()), float(sig.item()), float(idl), float(mbl)


def grad_caps_temporal_loss(traj, eps, valid=None):
    """Grad-CAPS displacement-normalized temporal-smoothness penalty (actor-only).

    traj: (B, L, n_dof) — consecutive policy sub-actions along time (the real applied
    path; here L = 2*action_block, the deterministic-mean block pi_mean(z_t) concatenated
    with pi_mean(z_{t+1})). For each interior triple (s_{k-1}, s_k, s_{k+1}):

        acc_k  = ||s_{k-1} - 2 s_k + s_{k+1}||_2     # == ||Da_t - Da_{t+1}||_2 (curvature)
        disp_k = ||s_{k+1} - s_{k-1}||_2             # net displacement across the window
        L_k    = acc_k * tanh( 1 / (disp_k + eps) )

    A smooth ramp has acc~0 -> ~0 loss at ANY speed (wide motion is free); an in-place
    zigzag has large acc and tiny disp -> tanh(1/eps)~1 -> curvature paid in full. disp is
    a SCALAR magnitude per window (norm over joints), NOT a per-dim reciprocal: a parked
    joint's 1/eps would otherwise dominate ||1/(d+eps)|| and saturate every window. eps caps
    the 1/disp blow-up; tanh keeps the factor in [0,1). Norms carry +1e-12 so the gradient
    is finite at acc=0 / disp=0. `valid` (B, L-2) masks windows (e.g. the join straddling an
    episode reset) before the mean."""
    s0, s1, s2 = traj[:, :-2], traj[:, 1:-1], traj[:, 2:]            # (B, L-2, n_dof)
    acc = ((s0 - 2.0 * s1 + s2).pow(2).sum(-1) + 1e-12).sqrt()       # (B, L-2) curvature
    disp = ((s2 - s0).pow(2).sum(-1) + 1e-12).sqrt()                 # (B, L-2) net displacement
    per_window = acc * torch.tanh(1.0 / (disp + eps))               # (B, L-2)
    if valid is None:
        return per_window.mean()
    return (per_window * valid).sum() / valid.sum().clamp_min(1.0)


def sac_update(buf, wm, actor, critic, critic_tgt, actor_opt, critic_opt,
               args, step, device):
    """Run args.updates_per_step SAC actor-critic gradient steps on PER samples from buf.
    SAC with a FIXED entropy temperature alpha (getattr default 0.2): the actor maximizes
    Q + alpha*H over reparameterized squashed-Gaussian samples and the critic learns the
    soft value (target min Q_bar - alpha*log pi). The entropy bonus is what keeps the policy
    off the freeze attractor — re-added 2026-06-14; the 2026-06-12 deterministic actor
    parked. Deployment still acts with the deterministic mean; only training samples. Twin
    critics + Polyak target kept. The encoder is frozen w.r.t. these updates: z is encoded
    under no_grad from the CURRENT wm.
    Returns {"critic_loss", "actor_loss", "zb"} from the last completed update, or
    None if the gate is closed (warmup / buffer below batch) or sampling came up dry.
    Shared by the online loop here and offline fine-tuning (offline_train.py)."""
    if step < args.start_steps or buf.total < args.batch_size:
        return None
    alpha = getattr(args, "alpha", 0.2)        # fixed entropy temperature (no learnable alpha)
    goal = getattr(args, "goal_explore", False)   # goal-conditioned deterministic (TD3-style) path
    per_beta = min(1.0, args.per_beta_start
                   + (1 - args.per_beta_start) * step / max(args.total_steps, 1))
    out = None
    for _ in range(args.updates_per_step):
        b = buf.sample_sac(args.batch_size, args.per_alpha, per_beta,
                           her_frac=(args.her_frac if goal else 0.0))
        if b is None:
            break
        zb = encode_obs(wm, b["px"], b["prop"], device)
        znb = encode_obs(wm, b["px_n"], b["prop_n"], device)
        if goal:
            # --- goal-conditioned DETERMINISTIC (TD3/DDPG-style) update; NO entropy. (Kept for the
            #     --goal-explore SAC variant; the --cem controller does NOT use this path.) ---
            zstar = encode_obs(wm, b["goal_px"], b["goal_prop"], device)     # (B, D) detached (no_grad encode)
            with torch.no_grad():
                an = actor(znb, zstar)                                        # deterministic next action
                tn = getattr(args, "td3_target_noise", 0.0)
                if tn > 0:                                                    # optional TD3 target smoothing
                    c = getattr(args, "td3_noise_clip", 0.5)
                    an = (an + (tn * torch.randn_like(an)).clamp(-c, c)).clamp(-1.0, 1.0)
                q1n, q2n = critic_tgt(znb, an, zstar)
                r_reach = -(znb - zstar).norm(dim=-1, keepdim=True) * b["goal_valid"] * (1 - b["d"])
                r = args.lambda_reach * r_reach + b["r"]
                y = r + (1 - b["d"]) * args.gamma * torch.min(q1n, q2n)
            q1, q2 = critic(zb, b["a"], zstar)
            critic_loss = (b["w"] * ((q1 - y).pow(2) + (q2 - y).pow(2))).mean()
            critic_opt.zero_grad(); critic_loss.backward(); critic_opt.step()
            ap = actor(zb, zstar)
            q1p, q2p = critic(zb, ap, zstar)
            actor_loss = (-torch.min(q1p, q2p)).mean()                       # deterministic PG (no entropy)
        else:
            with torch.no_grad():
                an, logpn, _ = actor.sample(znb)               # a' ~ pi (entropy bonus on z')
                q1n, q2n = critic_tgt(znb, an)
                # scalar critic: (B,1) soft target. multihead: per-head TD targets from the
                # per-component rewards, min-over-twins per head (pessimism per component); the
                # shared next action keeps the heads' sum == the scalar value. Entropy is global
                # (not a reward component) so it is spread equally across the K heads (/K) -> the
                # head-sum equals the scalar soft target; identical to safe15 at K=1.
                ent = args.gamma * (1 - b["d"]) * alpha * logpn          # (B,1) discounted entropy
                y = (b.get("rc", b["r"]) + (1 - b["d"]) * args.gamma * torch.min(q1n, q2n)
                     - ent / q1n.shape[-1])
            q1, q2 = critic(zb, b["a"])
            critic_loss = (b["w"] * ((q1 - y).pow(2) + (q2 - y).pow(2)).mean(-1, keepdim=True)).mean()
            critic_opt.zero_grad(); critic_loss.backward(); critic_opt.step()
            if args.per_priority == "td":   # ablation: replace curiosity priority with |TD error|
                td = (0.5 * (q1 + q2) - y).sum(-1).abs().detach().cpu().numpy()
                buf.update_priorities(b["e"], b["i"], td)
            ap, logpp, _ = actor.sample(zb)
            q1p, q2p = critic(zb, ap)
            q_min = torch.min(q1p.sum(-1, keepdim=True), q2p.sum(-1, keepdim=True))   # sum heads, then twin-min
            actor_loss = (alpha * logpp - q_min).mean()                              # maximize Q + alpha*H (== safe15 at K=1)
        if not goal and getattr(args, "actor_rate_reg", 0) > 0:
            # action-rate as an actor-loss regularizer instead of a reward term: penalize
            # the policy's own within-block sub-action jerk directly — smoothness pressure
            # that never enters r or propagates through Q (the curiosity balance untouched).
            sub = ap.view(ap.shape[0], args.action_block, -1)
            actor_loss = actor_loss + args.actor_rate_reg * sub.diff(dim=1).pow(2).mean()
        gc_val = 0.0
        if not goal and getattr(args, "lambda_temp", 0) != 0:
            # Grad-CAPS temporal smoothness (displacement-normalized): penalize the CURVATURE
            # of the policy's real applied sub-action path across the t->t+1 decision boundary
            # [pi_mean(z_t) | pi_mean(z_{t+1})], scaled by 1/displacement so a smooth wide ramp
            # is free and only low-travel zigzag (the in-place jitter that cheaply satisfies
            # 1-step curiosity) is paid. Actor-only: gradients flow through the policy mean,
            # Q/r untouched. STOCHASTIC-TRUNK ADAPTATION: the smoothness target is the DEPLOYED
            # path, so build the trajectory from the deterministic MEAN tanh(mean(z)) (== actor(z)
            # forward), NOT the squashed sample `ap`/`an` that feeds the entropy term. Both blocks
            # are encoded WITH grad here (zb/znb come from no_grad encode_obs, but actor params
            # are free) so the penalty backprops into pi.
            nb = args.action_block
            mp = actor(zb)                                            # current-state mean block (WITH grad)
            mpn = actor(znb)                                          # next-state mean block (WITH grad)
            traj = torch.cat([mp.view(mp.shape[0], nb, -1),
                              mpn.view(mpn.shape[0], nb, -1)], dim=1)  # (B, 2*nb, n_dof)
            vmask = torch.ones(traj.shape[0], 2 * nb - 2, device=traj.device)
            done = b["d"].view(-1) > 0.5            # the t->t+1 join spans an episode reset -> not a real path
            if done.any():
                vmask[done, nb - 2] = 0.0; vmask[done, nb - 1] = 0.0
            gc = grad_caps_temporal_loss(traj, args.grad_caps_eps, vmask)
            actor_loss = actor_loss + args.lambda_temp * gc
            gc_val = float(gc.item())
        actor_opt.zero_grad(); actor_loss.backward(); actor_opt.step()
        with torch.no_grad():
            for p, pt in zip(critic.parameters(), critic_tgt.parameters()):
                pt.mul_(1 - args.tau).add_(args.tau * p)
        out = {"critic_loss": float(critic_loss.item()),
               "actor_loss": float(actor_loss.item()), "grad_caps": gc_val, "zb": zb.detach()}
        if q1p.shape[-1] > 1:            # per-head mean Q: which component steers the policy
            out["q_heads"] = (0.5 * (q1p + q2p)).mean(0).detach().cpu().numpy()
    return out


@torch.no_grad()
def collapse_metrics(z):
    """Encoder diagnostics on a batch of latents z (B, D), computed on CPU (linalg has gaps
    on MPS). Returns (z_std, rankme):
      z_std  -- mean per-dim std = latent scale (->0 collapsed).
      rankme -- RankMe effective rank (Garrido et al. 2023, the SSL standard, same
                Balestriero/LeCun lineage as LeJEPA/LeWM): exp(Shannon entropy of the
                L1-normalized singular values of z, UNCENTERED), in [1, min(B,D)]; logged
                as a fraction of z_dim. Finite-sample biased (at B~=D even isotropic z reads
                ~0.8*D), so the probe must hold B >> D samples (see --probe-size)."""
    z = z.detach().float().cpu()
    std = z.std(0)
    sv = torch.linalg.svdvals(z)                              # RankMe: UNCENTERED singular values
    p = sv / (sv.sum() + 1e-12) + 1e-7                        # L1-normalized + eps (Garrido 2023)
    rankme = torch.exp(-(p * p.log()).sum()).item()
    return float(std.mean()), float(rankme)


# ----------------------------------------------------------------- checkpointing
def save_buffer(buf, out_dir):
    """Dump the collected transitions (chronological, per env) to out_dir/buffer_<N>.npz
    for offline training (e.g. ship to a RunPod GPU). Reconstructs each env's ring into
    oldest->newest order via start=(head-count)%C; is_start marks episode boundaries so a
    consumer never builds a WM window across a reset."""
    px, prop, act, r, d, isx, lengths = [], [], [], [], [], [], []
    for e in range(buf.n_envs):
        n = int(buf.count[e])
        if n == 0:
            continue
        start = (int(buf.head[e]) - n) % buf.C
        idx = (start + np.arange(n)) % buf.C
        px.append(buf.pixels[e, idx]); prop.append(buf.proprio[e, idx])
        act.append(buf.action[e, idx]); r.append(buf.r[e, idx])
        d.append(buf.d[e, idx]); isx.append(buf.is_start[e, idx]); lengths.append(n)
    if not lengths:
        print("[frozen] buffer empty -> nothing to save", flush=True)
        return None
    path = out_dir / f"buffer_{sum(lengths):07d}.npz"
    np.savez_compressed(path,
                        pixels=np.concatenate(px), proprio=np.concatenate(prop),
                        action=np.concatenate(act), r=np.concatenate(r),
                        d=np.concatenate(d), is_start=np.concatenate(isx),
                        env_lengths=np.asarray(lengths, np.int64))
    print(f"[frozen] saved {sum(lengths)} transitions -> {path} "
          f"({path.stat().st_size / 1e6:.1f} MB)", flush=True)
    return path


def save_and_upload(state, out_dir, step, repo_id, run_name, enable_hf, keep_local):
    """Save a checkpoint, upload to HF under <run_name>/ckpt_<step>.pt, then (unless
    keep_local) delete the local copy once the upload succeeds -- so disk stays
    bounded over a long run. On upload failure the local file is kept as a fallback."""
    path = out_dir / f"ckpt_{step:07d}.pt"
    torch.save(state, path)
    uploaded = False
    if enable_hf and repo_id and os.environ.get("HF_TOKEN"):
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=os.environ["HF_TOKEN"])
            api.create_repo(repo_id, repo_type="model", exist_ok=True)
            api.upload_file(path_or_fileobj=str(path), repo_id=repo_id,
                            path_in_repo=f"{run_name}/ckpt_{step:07d}.pt")
            uploaded = True
            print(f"[hf] uploaded {run_name}/ckpt_{step:07d}.pt -> {repo_id}", flush=True)
        except Exception as ex:
            print(f"[hf] upload failed (non-fatal, keeping local): {ex}", flush=True)
    if uploaded and not keep_local:
        try:
            path.unlink()
        except OSError:
            pass
        return None
    return path


def resolve_ckpt(ckpt=None, name="baseline", step=None, hf_repo=None):
    """Return a local checkpoint path: the explicit `ckpt` if given, else download
    <name>/ckpt_<step>.pt (or the latest step for that run) from the HF Hub
    (hf_repo, else $HF_UPLOAD_REPO_ID). Shared by play_policy.py / eval_predictor.py."""
    if ckpt:
        return ckpt
    repo = hf_repo or os.environ.get("HF_UPLOAD_REPO_ID")
    if not repo:
        raise SystemExit("no --ckpt and no HF repo (set --hf-repo or HF_UPLOAD_REPO_ID in .env)")
    from huggingface_hub import HfApi, hf_hub_download
    token = os.environ.get("HF_TOKEN")
    files = [f for f in HfApi(token=token).list_repo_files(repo)
             if f.startswith(f"{name}/") and f.endswith(".pt")]
    if not files:
        raise SystemExit(f"no checkpoints for run '{name}' in {repo}")
    target = f"{name}/ckpt_{step:07d}.pt" if step is not None else sorted(files)[-1]
    if target not in files:
        raise SystemExit(f"{target} not found in {repo}; available: {sorted(files)}")
    print(f"[hf] downloading {target} from {repo}", flush=True)
    return hf_hub_download(repo_id=repo, filename=target, token=token)


def load_init_ckpt(args, wm, actor, critic, critic_tgt, device):
    """Warm-start / resume: load WM + actor-critic weights so the online loop CONTINUES a
    prior run instead of starting cold (train.py otherwise only ever SAVES checkpoints,
    never loads). Resolves a local --init-ckpt or an HF --resume-name[/--resume-step].
    Returns h_fwd to overwrite the freshly-initialised value. NOTE: optimizer state is
    not in the checkpoint, so Adam moments restart — lower LRs for bring-up if needed."""
    path = resolve_ckpt(args.init_ckpt, args.resume_name or args.name, args.resume_step, args.hf_repo)
    ck = torch.load(path, map_location=device, weights_only=False)
    # Tolerant WM load: the encoder / pred_proj / action_encoder always match, so a frozen-encoder
    # resume is unaffected. Only a RESIZED predictor mismatches — e.g. resuming a pre-2026-06-26
    # small-predictor (8/32/1024) checkpoint into the LeWM-faithful default (16/64/2048). Load every
    # shape-matching tensor; re-init the rest (the predictor then re-bootstraps). Pass matching
    # --wm-pred-* to warm-start the old predictor exactly instead.
    own = wm.state_dict()
    keep = {k: v for k, v in ck["wm"].items() if k in own and own[k].shape == v.shape}
    wm.load_state_dict(keep, strict=False)
    if len(keep) != len(own):
        mods = sorted({k.split(".")[0] for k in own if k not in keep})
        print(f"[resume] WM partial load: {len(keep)}/{len(own)} tensors matched; re-initialised "
              f"{len(own) - len(keep)} (submodules: {mods}) — predictor size changed, it will "
              f"re-bootstrap. Pass --wm-pred-heads/--wm-pred-dim-head/--wm-pred-mlp-dim to match "
              f"the checkpoint exactly.", flush=True)
    h_fwd = int(ck.get("h_fwd", args.h_fwd_start))
    if getattr(args, "goal_explore", False):
        # goal-conditioned actor/critic have a WIDER first layer (z_dim + goal_dim). Load them
        # only from a checkpoint whose shapes match (another goal-explore run); otherwise warm-
        # start the WM (a trained WM gives meaningful r_cur from step 0) and start the goal-
        # conditioned policy fresh. (A strict critic load on a z-only ckpt would crash; a
        # strict=False actor load would silently reinit the wider trunk anyway.)
        want = actor.trunk[0].weight.shape[1]
        have = ck["actor"]["trunk.0.weight"].shape[1]
        if want == have:
            load_actor_state(actor, ck["actor"])
            critic.load_state_dict(ck["critic"]); critic_tgt.load_state_dict(ck["critic_tgt"])
            print(f"[resume] goal-explore: loaded wm+actor+critic from {path} "
                  f"(saved step {ck.get('step', '?')}, h_fwd={h_fwd})", flush=True)
        else:
            print(f"[resume] goal-explore: WM warm-started from {path} (saved step "
                  f"{ck.get('step', '?')}); actor/critic input {have}!={want} -> fresh "
                  f"goal-conditioned policy.", flush=True)
        return h_fwd
    load_actor_state(actor, ck["actor"])     # drops the log_std head of old stochastic ckpts
    critic.load_state_dict(ck["critic"])
    critic_tgt.load_state_dict(ck["critic_tgt"])
    print(f"[resume] loaded wm+actor+critic from {path} "
          f"(saved step {ck.get('step', '?')}, h_fwd={h_fwd})", flush=True)
    return h_fwd


def tile_frames(imgs):
    """Tile (n_envs, H, W, 3) per-env camera frames into one (rows*H, cols*W, 3) grid,
    so a single train-video clip shows all parallel envs at once."""
    n, H, W, C = imgs.shape
    cols = int(np.ceil(np.sqrt(n))); rows = int(np.ceil(n / cols))
    grid = np.zeros((rows * H, cols * W, C), imgs.dtype)
    for i in range(n):
        r, c = divmod(i, cols)
        grid[r * H:(r + 1) * H, c * W:(c + 1) * W] = imgs[i]
    return grid


# -------------------------------------------------------------------------- main
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[device] {device}  MUJOCO_GL={os.environ.get('MUJOCO_GL')}", flush=True)
    if device.type == "cuda":
        # Use the A100 tensor cores: TF32 for fp32 matmuls (the CEM rollout is matmul-bound,
        # 10x5 WM forwards x ~2400 seqs per decision). ~2-4x with negligible accuracy loss.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass

    if args.env_backend == "hardware":
        if args.n_envs != 1:
            print(f"[hardware] forcing n_envs=1 (was {args.n_envs}; one physical arm)", flush=True)
            args.n_envs = 1
        from env.hardware_env import HardwareSO101Env
        VecEnv = HardwareSO101Env
    else:
        from env.parallel_env import VectorMujocoEnv, SubprocVectorMujocoEnv
        VecEnv = SubprocVectorMujocoEnv if args.env_backend == "subproc" else VectorMujocoEnv
    if args.start_steps < 0:                      # default-aware: skip random warmup on a real arm
        args.start_steps = 0 if args.env_backend == "hardware" else 1000
    # Safety params are regime-dependent (sim vs real arm are MEASURED to differ): the real-arm
    # (delta=9, lambda_safe=2.2) pair freezes sim policies, while sim's calibrated balance is
    # (delta=15, lambda_safe=0.1). Resolve by backend unless the user set them explicitly, so one
    # binary is safe in both regimes (a bare sim run no longer inherits the hardware deadband).
    _hw = args.env_backend == "hardware"
    if args.safety_delta is None:
        args.safety_delta = 9.0 if _hw else 15.0
    if args.lambda_safe is None:
        args.lambda_safe = 2.2 if _hw else 0.1
    print(f"[safety] {args.env_backend} backend -> delta={args.safety_delta}, "
          f"lambda_safe={args.lambda_safe}", flush=True)
    # --- goal-conditioned Go-Explore: resolve gated coercions BEFORE the env / nets are built
    #     (action_max feeds VecEnv below; the rest gate the deterministic objective downstream) ---
    if args.cem:
        args.goal_explore = True             # --cem reuses the archive + goal sampling infra; SAC is skipped
        print(f"[cem] LeWM CEM controller ON: samples={args.cem_samples} iters={args.cem_iters} "
              f"elites={args.cem_elites} init_std={args.cem_init_std} horizon={args.cem_horizon} "
              f"(open-loop receding, reward=MSE only, SAC OFF)", flush=True)
    if args.goal_explore:
        if args.env_backend == "hardware":
            raise SystemExit("[goal-explore] refusing on hardware: dropping dq_max -> action_max=1.0 "
                             "(full-radian joint deltas) is unsafe on a real arm. Sim only.")
        args.deterministic_act = True       # no stochastic collection (the actor.sample path is unused)
        args.alpha = 0.0                    # kills the entropy terms in sac_update
        args.explore_noise = 0.0           # no collection noise
        args.lambda_safe = 0.0             # drop r_safe entirely
        args.multihead_q = False           # the dense scalar reach reward has no component decomposition
        args.per_priority = "curiosity"    # td-priority would fight the archive's r_cur scoring
        if args.action_max == 0.3:         # drop the dq_max scaling unless the user explicitly overrode it
            args.action_max = 1.0
        if args.goal_update_every <= 0:
            args.goal_update_every = args.max_episode_steps
        if args.goal_rescore_every <= 0:
            args.goal_rescore_every = args.goal_update_every
        print(f"[goal-explore] ON: K={args.goal_archive_size} update_every={args.goal_update_every} "
              f"rescore_every={args.goal_rescore_every} her_frac={args.her_frac} "
              f"lambda_reach={args.lambda_reach} action_max={args.action_max} "
              f"(deterministic, alpha=0, lambda_safe=0)", flush=True)
    if args.wm_cam == "overhead":
        print(f"[wm-cam] encoder/WM input = OVERHEAD (fixed third-person) camera, not the wrist cam "
              f"(prototype: LeWM-style smoother latent for goal-reaching).", flush=True)
    env_kwargs = dict(n_envs=args.n_envs, frame_skip=args.frame_skip,
                      action_max=args.action_max, encode_cam=args.wm_cam,
                      safety_delta=args.safety_delta, seed=args.seed,
                      threads=args.env_threads)
    if args.env_backend != "hardware":     # sim only: identical deterministic object layout across envs
        env_kwargs["fixed_objects"] = args.fixed_objects
        if args.fixed_objects:
            print("[env] fixed-objects ON: identical deterministic scene for every env & reset "
                  "(variance = arm only)", flush=True)
    if args.env_backend == "subproc":      # GPU EGL render only in the CUDA-free subproc workers
        env_kwargs["render_backend"] = args.render_backend
        print(f"[render] subproc workers render via {args.render_backend.upper()} "
              f"({'GPU, ~100x faster than osmesa' if args.render_backend == 'egl' else 'CPU'})", flush=True)
    env = VecEnv(**env_kwargs)
    n_dof = env.n_dof
    a_dim = n_dof * args.action_block
    prop_dim = 3 * n_dof
    H = args.history_size

    wm = WorldModel(n_dof=n_dof, action_block=args.action_block,
                    history_size=H, dropout=args.wm_dropout,
                    use_proprio=not args.no_proprio,
                    **pred_dims_from_args(args)).to(device)
    if args.wm_grad_checkpoint:  # off by default: ViT-tiny encode activations are sub-GB vs 80GB free,
        try:                     # so recompute-on-backward is pure slowdown here (the H_fwd rollout is in latent space)
            wm.encoder.vit.gradient_checkpointing_enable()
        except Exception as ex:
            print(f"[wm] grad checkpoint not enabled: {ex}", flush=True)
    wm.eval()                                  # train() only inside wm_update
    if args.freeze_encoder:
        # Stop-gradient on the obs->z encoder (StateEncoder): the latent SPACE becomes STATIONARY
        # (no more SIGReg inflation / encoder drift), so CEM plans against a FIXED geometry --
        # LeWM-style frozen WM. The predictor/action_encoder/pred_proj keep training to sharpen
        # rollouts IN that fixed latent. Done BEFORE wm_opt so the optimizer excludes these params;
        # the encoder is also kept in eval() during wm_update (dropout off -> deterministic z).
        for p in wm.encoder.parameters():
            p.requires_grad_(False)
        wm.encoder.eval()
        nfz = sum(p.numel() for p in wm.encoder.parameters()) / 1e6
        if not (args.init_ckpt or args.resume_name):
            print("[freeze] WARNING: --freeze-encoder without --resume-name/--init-ckpt freezes a "
                  "RANDOM encoder (the latent never learns) -- resume from a checkpoint.", flush=True)
        print(f"[freeze] encoder frozen (stop-grad): {nfz:.2f}M params; latent STATIONARY. "
              f"Predictor + action_encoder still train.", flush=True)
    wm.predict_eager = wm.predict          # uncompiled ref -- the CEM bf16 rollout MUST use this:
                                           # torch.compile + bf16 autocast miscompiles wm.predict to NaN
                                           # (verified 2026-06-21), which silently makes CEM a no-op
                                           # (all candidate costs NaN -> topk arbitrary -> random plan).
    if device.type == "cuda" and not args.no_compile:
        # compile only helps wm_update's fp32 rollout here (CEM is compute- not launch-bound, ~1.0x);
        # the CEM rollout stays EAGER (predict_eager) because compile+bf16 -> NaN. dynamic=True tolerates
        # the changing batch/H shapes; suppress_errors falls back to eager on any graph break.
        try:
            import torch._dynamo as _dynamo      # NB: `import torch._dynamo` would bind `torch` local to main()
            _dynamo.config.suppress_errors = True
            wm.predict = torch.compile(wm.predict, dynamic=True)
            print("[wm] torch.compile(wm.predict) ON for training; CEM rollout uses EAGER predict "
                  "(compile+bf16 autocast miscompiles to NaN)", flush=True)
        except Exception as ex:
            print(f"[wm] torch.compile disabled ({ex})", flush=True)
    sigreg = SIGReg(knots=17, num_proj=1024).to(device)
    z_dim = wm.z_dim

    goal_dim = z_dim if args.goal_explore else 0   # z* re-encoded to the same 256-dim latent space
    actor = Actor(z_dim, a_dim, goal_dim=goal_dim).to(device)
    n_q_out = len(REWARD_COMPONENTS) if args.multihead_q else 1
    critic = TwinQ(z_dim, a_dim, n_out=n_q_out, goal_dim=goal_dim).to(device)
    critic_tgt = TwinQ(z_dim, a_dim, n_out=n_q_out, goal_dim=goal_dim).to(device)
    critic_tgt.load_state_dict(critic.state_dict())
    for p in critic_tgt.parameters():
        p.requires_grad_(False)

    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    # --encoder-thaw-every interleaves frozen acting with periodic encoder co-adaptation. The optimizer
    # must then include the (initially frozen) encoder params so AdamW can update them during thawed windows;
    # params with grad=None (frozen windows) are skipped automatically, so this is a no-op when frozen.
    if args.encoder_thaw_every > 0:
        # two LR groups: the encoder gets a (typically smaller) thaw LR so co-adaptation NUDGES the latent
        # rather than lurching it (the lurch is what 50%-duty full-LR thaw did -> locality collapse).
        _enc = list(wm.encoder.parameters()); _enc_ids = {id(p) for p in _enc}
        _other = [p for p in wm.parameters() if id(p) not in _enc_ids]
        _thaw_lr = args.encoder_thaw_lr if args.encoder_thaw_lr > 0 else args.wm_lr
        wm_opt = torch.optim.AdamW([{"params": _other, "lr": args.wm_lr},
                                    {"params": _enc, "lr": _thaw_lr}], weight_decay=1e-3)
    else:
        wm_opt = torch.optim.AdamW([p for p in wm.parameters() if p.requires_grad],
                                   lr=args.wm_lr, weight_decay=1e-3)

    # RND novelty (only built/trained/used when --lambda-rnd != 0): frozen random target + chasing
    # predictor, over the RAW obs (downsampled wrist image + proprio) -- a STABLE input space,
    # unlike the co-trained latent z which would drift the target out from under the predictor.
    # At lambda_rnd==0 these are unused (no reward term, no training) so the default path is untouched.
    rnd_target = rnd_pred = rnd_opt = rnd_img_rms = rnd_prop_rms = None
    if args.lambda_rnd:
        rnd_target = RNDObsNet(prop_dim, args.rnd_out_dim, args.rnd_hidden).to(device)
        for p in rnd_target.parameters():
            p.requires_grad_(False)
        rnd_pred = RNDObsNet(prop_dim, args.rnd_out_dim, args.rnd_hidden).to(device)
        rnd_opt = torch.optim.Adam(rnd_pred.parameters(), lr=args.rnd_lr)
        rnd_img_rms = RunningMeanStd((), device)             # obs-norm: scalar stats for the 84x84 frame
        rnd_prop_rms = RunningMeanStd((prop_dim,), device)   # obs-norm: per-dim stats for proprio

    # --- resume / warm-start: load wm+sac weights BEFORE the loop (else cold start) ---
    resume_h_fwd = None
    if args.init_ckpt or args.resume_name:
        resume_h_fwd = load_init_ckpt(args, wm, actor, critic, critic_tgt, device)

    # --- frozen-policy data collection: act + buffer, NO gradient updates ---
    policy_loaded = bool(args.init_ckpt or args.resume_name)
    if args.frozen_policy and not policy_loaded:
        msg = ("--frozen-policy with no --init-ckpt/--resume-name -> a RANDOM-init actor "
               "would drive the arm with no learning to correct it")
        if args.env_backend == "hardware":
            raise SystemExit(f"[frozen] REFUSING on hardware: {msg}. "
                             "Load a policy, e.g. --resume-name safe15 --resume-step 100000.")
        print(f"[frozen] WARNING: {msg}.", flush=True)
    if args.frozen_policy:
        print("[frozen] data-collection mode: NO gradient updates (WM/SAC/curriculum "
              "all skipped); acting + buffering only. Buffer -> out_dir/buffer_<N>.npz on exit.",
              flush=True)

    cap = int(np.clip(args.buffer_frac * args.total_steps, 1000, 50_000))
    cap_per_env = max(cap // args.n_envs, args.history_size + args.h_fwd_max + 8)
    if args.frozen_policy or args.save_buffer:   # collection KEEPS everything -> size the ring to the whole run
        keep = min(args.total_steps, 50_000)     # (the per-env ring otherwise overwrites the oldest in place)
        if args.total_steps > 50_000:
            print(f"[frozen] WARNING: requested {args.total_steps} steps but the buffer holds {keep}/env; "
                  f"oldest will be overwritten. Split into shorter runs to keep all transitions.", flush=True)
        cap_per_env = max(cap_per_env, keep)
    buf = ReplayBuffer(args.n_envs, cap_per_env, env.wrist_resolution, a_dim, prop_dim, device,
                       n_comp=len(REWARD_COMPONENTS) if args.multihead_q else 0,
                       goal_explore=args.goal_explore)
    print(f"[buffer] {args.n_envs} x {cap_per_env} = {args.n_envs * cap_per_env} transitions", flush=True)
    # goal archive(s): ONE global pool shared by all envs (default), or one PER ENV
    # (--per-env-archive) so each env explores/targets only the high-MSE states IT visited.
    archives = ([GoalArchive(args.goal_archive_size, env.wrist_resolution, prop_dim, a_dim)
                 for _ in range(args.n_envs if args.per_env_archive else 1)]
                if args.goal_explore else None)
    arch_for = (lambda e: archives[e % len(archives)]) if archives else (lambda e: None)
    if archives is not None:
        print(f"[goal-explore] {'PER-ENV' if args.per_env_archive else 'GLOBAL'} archive: "
              f"{len(archives)} buffer(s) x K={args.goal_archive_size}", flush=True)
    if archives is not None and (args.init_ckpt or args.resume_name):   # reload archive(s) on resume
        try:
            _ap = resolve_ckpt(args.init_ckpt, args.resume_name or args.name, args.resume_step, args.hf_repo)
            _ack = torch.load(_ap, map_location="cpu", weights_only=False)
            if "goal_archive" in _ack:
                _sd = _ack["goal_archive"]
                _sd = _sd if isinstance(_sd, list) else [_sd]      # back-compat: old single-pool checkpoint
                for _a, _s in zip(archives, _sd):
                    _a.load_state_dict(_s)
                print(f"[resume] goal archive loaded: {sum(a.n for a in archives)} goals "
                      f"across {len(archives)} buffer(s)", flush=True)
        except Exception as _ex:
            print(f"[resume] goal archive not loaded ({_ex}); starting empty", flush=True)

    run_name = args.name
    out_dir = Path(args.out_dir) if args.out_dir else Path("runs") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    n_params = sum(p.numel() for p in wm.parameters())
    print(f"[run] name={run_name}  out_dir={out_dir}  wm_params={n_params/1e6:.2f}M", flush=True)
    if args.env_backend == "hardware":
        loaded = bool(args.init_ckpt or args.resume_name)
        warns = []
        if args.action_max > 0.15:
            warns.append(f"action_max={args.action_max} is large for a real arm")
        if not loaded:
            warns.append("no policy loaded (random actor)")
        print(f"[hardware] SAFETY: n_envs=1, control_dt={env.dt_safe}s, "
              f"action_max={args.action_max} (<= +/-{args.action_max} rad/joint/step), "
              f"start_steps={args.start_steps}, policy={'loaded' if loaded else 'RANDOM'}. "
              f"Keep the e-stop within reach." + ("  [!] " + "; ".join(warns) if warns else ""),
              flush=True)

    # acting mode: SAC sample in sim training (entropy exploration that un-parks the policy
    # and decorrelates the parallel envs); deterministic mean on hardware, in frozen-policy
    # collection, and whenever --deterministic-act is set (deployment / eval).
    stochastic_act = (not args.deterministic_act and not args.frozen_policy
                      and args.env_backend != "hardware")
    print(f"[policy] acting={'stochastic sample (training)' if stochastic_act else 'deterministic mean (deploy/eval)'}"
          f"; alpha={args.alpha}", flush=True)

    run = None
    if not args.no_wandb and wandb is not None and os.environ.get("WANDB_API_KEY"):
        # name = the short keyword; every constant/variable goes in the config table.
        run = wandb.init(project=args.wandb_project or os.environ.get("WANDB_PROJECT", "curious-robot"),
                         entity=os.environ.get("WANDB_ENTITY"), name=run_name, group=run_name,
                         dir=str(out_dir),
                         config={**vars(args), "z_dim": z_dim, "a_dim": a_dim,
                                 "n_params_wm": n_params, "device": str(device)})
        print(f"[wandb] {run.url}", flush=True)

    def wlog(d, step):
        if run is not None:
            run.log(d, step=step)

    # --- live per-env state (history is (H, n_envs, .) so resets touch one row) ---
    obs = env.reset()
    if args.no_torque_obs:
        scrub_torque_obs(obs, n_dof)
    z = encode_obs(wm, obs["image"], obs["proprio"], device)        # (n_envs, z_dim)
    hist_z = z.unsqueeze(0).repeat(H, 1, 1)
    hist_a = torch.zeros(H, args.n_envs, a_dim, device=device)
    # --- CEM open-loop receding-horizon plan buffer (LeWM WorldModelPolicy): a (n_envs, H, a_dim) plan
    #     per env, consumed one block-action per decision; cem_ptr starts "exhausted" so step 0 plans,
    #     and an env re-plans when its buffer empties / it just reset / its goal refreshed this step. ---
    cem_H = max(args.cem_horizon, 1)
    # replan stride: re-plan every cem_stride decisions (executing only that many of the H-step plan),
    # decoupling execution from lookahead. 0 -> LeWM open-loop (stride == H). 1 -> true receding-horizon MPC.
    cem_stride = cem_H if args.cem_replan_every <= 0 else min(args.cem_replan_every, cem_H)
    cem_buf = torch.zeros(args.n_envs, cem_H, a_dim, device=device)
    cem_ptr = np.full(args.n_envs, cem_H, np.int64)
    cem_diag = {}                                     # latest CEM action-sensitivity probe (cem/cost_cv, z_term_spread, reach_gap)
    knn_buf = torch.zeros(0, z_dim, device=device)    # recent-latent ring buffer for the k-NN coverage reward (--lambda-knn)
    is_start = np.ones(args.n_envs, bool)
    ep_len = np.zeros(args.n_envs, np.int64)
    ep_ret = np.zeros(args.n_envs, np.float32)

    # --- goal-conditioned Go-Explore: one pursued goal o* (+ its re-encoded z*) per env ---
    goal_img_hw = env.wrist_resolution
    goal_px_env = np.zeros((args.n_envs, goal_img_hw, goal_img_hw, 3), np.uint8)
    goal_prop_env = np.zeros((args.n_envs, prop_dim), np.float32)
    zstar_env = torch.zeros(args.n_envs, z_dim, device=device)
    has_goal = np.zeros(args.n_envs, bool)
    goal_evictions = 0
    goal_recent = ({k: deque(maxlen=400) for k in ("dist", "reach", "dist_env", "qpos_dist", "qpos_reach")}
                   if args.goal_explore else None)
    # reachable-radius curriculum state: the CURRENT goal offset k (mutable). Pure goal-RANGE schedule --
    # grown by the controller in the main loop; refresh_goals reads curric_k[0]. No loss/arch involvement.
    curric_k = [args.goal_curric_start if (args.goal_curriculum and args.goal_curric_start > 0)
                else args.goal_future_k]
    curric_d = [float(args.goal_curric_d_start)]    # --goal-select highmse_under_d latent-distance budget (grown like curric_k)

    def reach_eps_now(step):
        """Annealed reach threshold for the goal/reach_rate DIAGNOSTIC: linearly from
        --goal-reach-eps-start down to --goal-reach-eps over anneal_frac*total_steps, then
        held at the floor. eps_start<=0 -> constant --goal-reach-eps (default, unchanged)."""
        e0, e1 = args.goal_reach_eps_start, args.goal_reach_eps
        if e0 <= 0 or e0 <= e1:
            return e1
        span = max(int(args.goal_reach_eps_anneal_frac * args.total_steps), 1)
        return max(e1, e0 - (e0 - e1) * min(step, span) / span)

    def refresh_goals(env_idx):
        """Draw a fresh goal obs o* from the archive for each env in env_idx (sets goal_px_env /
        has_goal). z* is RE-ENCODED from o* every step in the main loop (drift-immune), not here.
        No-op until the archive fills.
        --goal-select 'mse' (default): P(k)~softmax(score/temp) = highest-WM-MSE (farthest/novel).
        'near': P(k)~softmax(-||z_e - z_k||/temp) over archive latents re-encoded HERE (no_grad) =
        goals NEAREST the env's current latent z (reachable within the CEM horizon)."""
        if archives is None:
            return
        if args.goal_select in ("future", "recent"):     # controlled-distance goal from the env's own buffer
            ei = np.asarray([int(e) for e in env_idx])
            if args.goal_select == "recent":             # k decisions AGO (reach_gap ~ k*step_jump, small/controlled)
                gpx, gprop, gvalid = buf.sample_recent_goal(ei, curric_k[0])    # curriculum-controlled offset
            else:                                         # k decisions ahead of a random anchor (on-manifold)
                gpx, gprop, gvalid = buf.sample_future_goal(ei, curric_k[0])
            for slot in range(len(ei)):
                if gvalid[slot]:
                    e = int(ei[slot])
                    goal_px_env[e] = gpx[slot]
                    goal_prop_env[e] = gprop[slot]
                    has_goal[e] = True
            return
        if args.goal_select == "highmse_under_d":
            # MSE-BUFFER curriculum: sample candidate transitions, re-score their CURRENT-WM one-step MSE
            # (same metric as score_obs_mse / curiosity_reward), and for each env pursue the HIGHEST-MSE
            # candidate whose goal latent is within the curriculum distance budget curric_d of z_now
            # (fallback: nearest). Encoder frozen => ||z_cand - z_now|| is a stable reachability metric.
            cand = buf.sample_candidates(args.goal_cand_n)
            if cand is None:
                return
            spx, sprop, sact, gpx, gprop = cand
            with torch.no_grad():
                z_src = encode_obs(wm, spx, sprop, device)                 # (N, D)
                z_cand = encode_obs(wm, gpx, gprop, device)               # (N, D) goal latents
                z_ctx = z_src.unsqueeze(1).repeat(1, H, 1)                # (N, H, D)
                ac = torch.as_tensor(sact, device=device).float().unsqueeze(1).repeat(1, H, 1)
                pred = wm.predict(z_ctx, wm.action_encoder(ac))[:, -1]     # (N, D) one-step pred
                mse = (pred - z_cand).pow(2).mean(-1)                      # (N,) current-WM MSE
                for e in env_idx:
                    e = int(e)
                    dist = (z[e].unsqueeze(0) - z_cand).norm(dim=-1)       # (N,) ||z_cand - z_now||
                    under = dist < curric_d[0]
                    j = int(torch.where(under, mse, torch.full_like(mse, -1e30)).argmax()) \
                        if bool(under.any()) else int(dist.argmin())       # highest MSE under d, else nearest
                    goal_px_env[e] = gpx[j]
                    goal_prop_env[e] = gprop[j]
                    has_goal[e] = True
            return
        near = args.goal_select == "near"
        zk_cache = {}                                    # archive latents per distinct archive (global -> encode once)
        for e in env_idx:
            a = arch_for(e)
            if a.n == 0:
                continue
            if near:
                with torch.no_grad():
                    key = id(a)
                    if key not in zk_cache:
                        zk_cache[key] = encode_obs(wm, a.gpx[:a.n], a.gprop[:a.n], device)   # (n, D)
                    dist = (z[e].unsqueeze(0) - zk_cache[key]).norm(dim=-1)                   # ||z_e - z_k||
                    p = torch.softmax(-dist / max(args.goal_temp, 1e-6), 0).cpu().numpy()
                k = int(np.random.choice(a.n, p=p))
            else:
                k = a.sample(args.goal_temp)
                if k is None:
                    continue
            goal_px_env[e] = a.gpx[k]
            goal_prop_env[e] = a.gprop[k]
            has_goal[e] = True

    h_fwd = resume_h_fwd if resume_h_fwd is not None else args.h_fwd_start   # curriculum horizon (resumed if warm-started)
    if args.h_fwd_override > 0:               # force the WM-rollout-training horizon, IGNORING the resumed value
        h_fwd = args.h_fwd_override           # (resume normally pins h_fwd to the ckpt's 1-step stage)
        print(f"[h_fwd] OVERRIDE -> training WM on {h_fwd}-step rollouts (was resume/start={resume_h_fwd or args.h_fwd_start})", flush=True)
    pred_hist = deque(maxlen=args.flatline_window)    # for the flatline bump trigger
    updates_at_stage = 0
    rnd_upd = 0                                       # RND predictor update counter (for --rnd-train-every)
    prev_sub_a = np.zeros((args.n_envs, n_dof), np.float64)   # last sub-action of the previous block (action-rate boundary)
    tau_max_arr = np.asarray(env.tau_max, np.float32)
    prev_qpos_dec = None                                      # last decision's final joint pose (for pose_step travel)
    recent_qpos = deque(maxlen=200)                           # rolling final-pose history -> pose_spread / pose_range
    step_jump_recent = deque(maxlen=400)                      # per-step ||z_t - z_{t+1}|| -> latent temporal locality
    recent = {k: deque(maxlen=400) for k in
              ("r_cur", "r_safe", "cur_contrib", "contacts", "table_contacts",
               "motion", "ret", "frac_block", "frac_table",
               "rate", "rate2", "energy", "qd_mean", "tau_sat", "qd_rev",
               "r_rate", "r_energy", "pose_step",
               "r_rnd", "rnd_contrib", "r_knn", "knn_contrib")}
    recent_mse = {k: deque(maxlen=2000) for k in ("mse_block", "mse_table", "mse_none")}
    t0 = time.time()
    last_wm = last_sac = last_rnd = None
    last_zb = last_qh = None
    smooth_walk = torch.zeros(args.n_envs, n_dof, device=device)   # OU state for --collect-smooth (persists across decisions)
    video_on = imageio is not None and args.video_every > 0
    wrist_buf = deque(maxlen=args.video_steps)      # train-video clips (per-env frames, tiled)
    over_buf = deque(maxlen=args.video_steps)
    probe_px = probe_prop = None                    # fixed diverse probe set for encoder/eff_rank_probe
    probe_buf = []                                  # warmup-rollout fallback if the HF probe is unavailable
    if args.probe_size > 0:                         # prefer the canonical uniform-pose probe cached on HF
        loaded = (load_probe_hf(args.hf_repo or os.environ.get("HF_UPLOAD_REPO_ID"), args.probe_id)
                  if not args.no_hf else None)
        # use the cached HF probe ONLY if it holds >= probe_size obs; an undersized probe (e.g. the
        # 256-obs probe_v1, == z_dim) would re-introduce the finite-sample rank bias the larger
        # probe_size is meant to remove, so fall through to the warmup-rollout probe instead.
        if loaded is not None and len(loaded[0]) >= args.probe_size:
            probe_px, probe_prop = loaded[0][:args.probe_size], loaded[1][:args.probe_size]
            if args.no_torque_obs:               # probe obs must match the scrubbed training obs
                probe_prop = probe_prop.copy()
                probe_prop[..., 2 * n_dof:3 * n_dof] = 0.0
            print(f"[probe] loaded {len(probe_px)} uniform-pose obs from HF ({args.probe_id})", flush=True)
        elif loaded is not None:
            print(f"[probe] HF probe '{args.probe_id}' has only {len(loaded[0])} obs < probe_size="
                  f"{args.probe_size}; using warmup-rollout probe instead (rank needs >> z_dim samples)",
                  flush=True)
        else:
            print(f"[probe] HF probe '{args.probe_id}' unavailable; falling back to warmup-rollout probe",
                  flush=True)

    def learner_updates(step, h_fwd):
        """SAC + periodic WM gradient steps on buffered (past) data; returns the
        possibly-bumped h_fwd. Called between env.step_block_async/step_block_wait so
        these GPU updates overlap the env workers rendering the next decision. Update
        count and schedule are identical to the serial loop; they just see the buffer
        minus the single in-flight transition (added after wait) -- negligible off-policy."""
        nonlocal last_wm, last_sac, last_rnd, last_zb, last_qh, updates_at_stage, rnd_upd
        # --- world-model co-training: autoregressive MSE rollout + beta*SIGReg ---
        if args.consolidate_every > 0:                    # multi-epoch consolidation regime (LeWM-like)
            do_wm = step % args.consolidate_every == 0
            n_grad = args.consolidate_epochs * max(1, buf.total // args.wm_batch_size)  # ~epochs over the frozen buffer
        else:                                             # default online schedule
            do_wm = step % args.wm_update_every == 0
            n_grad = args.wm_grad_steps
        if step >= args.start_steps and do_wm:
            did_wm = False
            for _ in range(n_grad):                       # consolidation: ~epochs*buffer/batch grad steps in one burst
                batch = buf.sample_wm(args.wm_batch_size, H + h_fwd, args.wm_sample, args.per_alpha)
                if batch is None:
                    break
                wm.train()
                _thawed = args.encoder_thaw_every > 0 and (step % args.encoder_thaw_every) < args.encoder_thaw_dur
                if args.encoder_thaw_every > 0:        # INTERLEAVED freeze/thaw: co-adapt the encoder in bursts
                    for p in wm.encoder.parameters():
                        p.requires_grad_(_thawed)       # frozen windows -> no grad -> AdamW skips the encoder
                    wm.encoder.train(_thawed)           # dropout on only while thawed
                elif args.freeze_encoder:
                    wm.encoder.eval()          # keep the frozen encoder deterministic (dropout off)
                # optionally REDUCE SIGReg during thaw windows: the isotropy (unit-variance) pressure is what
                # scatters consecutive frames when the encoder co-adapts on directed motion -> locality dies.
                # A lower beta in thaw windows lets the encoder learn prediction-locality without that scattering.
                # -1 (default) = always use args.sigreg_weight.
                _beta = (args.encoder_thaw_beta if (_thawed and args.encoder_thaw_beta >= 0)
                         else args.sigreg_weight)
                last_wm = wm_update(wm, sigreg, wm_opt, batch, H, h_fwd,
                                    args.gamma_wm, _beta, device,
                                    pertimestep=args.sigreg_pertimestep,
                                    lambda_slow=args.lambda_slow)
                wm.eval()
                pred_hist.append(last_wm[0]); updates_at_stage += 1; did_wm = True
            # curriculum: bump H_fwd when pred loss flatlines over the last window
            if (did_wm and h_fwd < args.h_fwd_max and len(pred_hist) == pred_hist.maxlen
                    and updates_at_stage >= pred_hist.maxlen):
                arr = np.asarray(pred_hist); half = len(arr) // 2
                older, newer = arr[:half].mean(), arr[half:].mean()
                if abs((older - newer) / max(abs(older), 1e-9)) < args.flatline_tol:
                    h_fwd += 1; updates_at_stage = 0; pred_hist.clear()
                    print(f"[curriculum] step={step} H_fwd -> {h_fwd}", flush=True)
        # --- SAC updates (PER; encoder is frozen w.r.t. SAC, z encoded under no_grad) ---
        res = (None if (args.cem or args.collect_smooth) else  # --cem / --collect-smooth: WM-only training (no SAC)
               sac_update(buf, wm, actor, critic, critic_tgt, actor_opt, critic_opt,
                          args, step, device))
        if res is not None:
            last_sac = (res["critic_loss"], res["actor_loss"], res.get("grad_caps", 0.0))
            last_zb = res["zb"]
            last_qh = res.get("q_heads")
        # --- RND predictor training (only when --lambda-rnd != 0): one step every --rnd-train-every
        #     SAC updates, on the RAW next-obs of a fresh PER batch. The trunk sac_update is a
        #     module-level fn with no RND in scope, so we draw our own buf.sample_sac batch here and
        #     train against b["px_n"]/b["prop_n"] (the next-obs raw pixels/proprio the buffer stores). ---
        if args.lambda_rnd and step >= args.start_steps and buf.total >= args.batch_size:
            rnd_upd += 1
            if rnd_upd % args.rnd_train_every == 0:
                b = buf.sample_sac(args.batch_size, args.per_alpha, per_beta=1.0)
                if b is not None:
                    ri, rp = to_rnd_obs(b["px_n"], b["prop_n"], rnd_img_rms, rnd_prop_rms, device)
                    err = (rnd_pred(ri, rp) - rnd_target(ri, rp)).pow(2).mean(-1)   # (B,) per-sample RND error
                    # scale each sample's loss by err/MSE (its novelty vs the batch-mean predictor MSE),
                    # stop-grad so it's a pure weight (avg 1): focuses predictor capacity on currently-novel
                    # samples. Adam absorbs the shared 1/MSE scalar but NOT the per-sample reweighting, so
                    # unlike a global loss-divide this actually changes training. Clamp caps outliers.
                    w = (err.detach() / (err.detach().mean() + 1e-8)).clamp(max=args.rnd_loss_clip)
                    rnd_loss = (w * err).mean()
                    rnd_opt.zero_grad(); rnd_loss.backward(); rnd_opt.step()
                    last_rnd = float(err.mean().item())   # report the UNWEIGHTED predictor MSE (comparable)
        return h_fwd

    # graceful stop for data-collection runs: 1st Ctrl-C finishes the in-flight decision and
    # saves; a 2nd Ctrl-C (default handler restored) force-quits.
    _stop = {"flag": False}
    if args.frozen_policy or args.save_buffer:
        import signal
        def _on_sigint(signum, frame):
            _stop["flag"] = True
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            print("\n[frozen] stop requested -> finishing this decision then saving "
                  "(Ctrl-C again to force-quit).", flush=True)
        signal.signal(signal.SIGINT, _on_sigint)

    for step in range(args.total_steps):
        if _stop["flag"]:
            print(f"[frozen] graceful stop at step {step} ({buf.total} transitions).", flush=True)
            break
        cur_px, cur_prop = obs["image"], obs["proprio"]          # o_t (before acting)

        # --- goal-explore: (re)sample goals on the cadence, then RE-ENCODE z* EVERY step from the
        #     raw goal obs (drift-immune collection); self-goal z* = z where no archive goal yet
        #     (reach is masked for those in sac_update). Log dist/reach over real-goal envs. ---
        if args.goal_explore:
            if step % args.goal_update_every == 0:
                refresh_goals(np.arange(args.n_envs))
            with torch.no_grad():
                if has_goal.any():
                    hg = torch.as_tensor(has_goal, device=device)
                    zstar_env[hg] = encode_obs(wm, goal_px_env[has_goal], goal_prop_env[has_goal], device)
                nog = ~has_goal
                if nog.any():
                    nog_t = torch.as_tensor(nog, device=device)
                    zstar_env[nog_t] = z[nog_t]
                if has_goal.any():
                    gdist = (z - zstar_env).norm(dim=-1).cpu().numpy()       # ||z_t - z*|| per env
                    goal_recent["dist"].append(float(gdist[has_goal].mean()))
                    goal_recent["reach"].append(float((gdist[has_goal] < reach_eps_now(step)).mean()))
                    goal_recent["dist_env"].append(np.where(has_goal, gdist, np.nan))  # per-env series; NaN where self-goal (excluded)
                    # PHYSICAL (joint-space) success: proprio[..., :n_dof] = raw qpos (radians). A goal counts
                    # reached when EVERY joint is within --goal-success-qpos-eps of the goal's stored qpos
                    # (inf-norm). Ground-truth check independent of the latent metric -- DIAGNOSTIC only.
                    qd = np.abs(cur_prop[:, :n_dof] - goal_prop_env[:, :n_dof]).max(axis=1)   # (n_envs,) max joint err
                    goal_recent["qpos_dist"].append(float(qd[has_goal].mean()))
                    goal_recent["qpos_reach"].append(float((qd[has_goal] < args.goal_success_qpos_eps).mean()))
            # --- reachable-radius CURRICULUM: pure goal-RANGE schedule (NOTHING in the loss/architecture;
            #     target stays LATENT goal/reach_rate, never qpos). Start goals INSIDE the planner's ~5-unit
            #     closing radius (small k) and grow k outward (further-back goal = larger reach_gap) once the
            #     windowed LATENT reach clears the threshold -- extends the reachable radius as the planner earns it.
            if (args.goal_curriculum and step > 0 and step % args.goal_curric_patience == 0
                    and len(goal_recent["reach"]) >= 50):
                wr = float(np.mean(goal_recent["reach"]))
                if wr >= args.goal_curric_thresh:
                    if args.goal_select == "highmse_under_d":         # grow the latent-DISTANCE budget d
                        if curric_d[0] < args.goal_curric_d_max:
                            curric_d[0] = min(curric_d[0] + args.goal_curric_d_step, args.goal_curric_d_max)
                            goal_recent["reach"].clear()
                            print(f"[goal-curriculum] step={step} latent_reach={wr:.2f} -> distance d={curric_d[0]:.2f}", flush=True)
                    elif curric_k[0] < args.goal_curric_max_k:        # grow the decisions-ago radius k (recent/future)
                        curric_k[0] += 1
                        goal_recent["reach"].clear()             # measure the harder (larger-k) stage fresh
                        print(f"[goal-curriculum] step={step} latent_reach={wr:.2f} -> radius k={curric_k[0]}", flush=True)

        # --- act: SAC stochastic sample during sim training (the entropy-driven exploration
        #     that un-parks the policy and decorrelates the parallel envs), the DETERMINISTIC
        #     mean on hardware / frozen deployment / eval (stochastic_act is False there).
        #     --warmup-random still opts in a uniform warmup; --explore-noise adds extra
        #     collection noise on top (sim only). ---
        with torch.no_grad():
            if args.collect_smooth:          # scripted SMOOTH collector: sub-action-level OU walk,
                subs = []                    # persisted across decisions -> smooth within AND across blocks
                for _ in range(args.action_block):
                    smooth_walk.mul_(args.smooth_beta).add_((1.0 - args.smooth_beta) * torch.randn_like(smooth_walk))
                    smooth_walk.clamp_(-1.0, 1.0)
                    subs.append(smooth_walk.clone())
                a = torch.stack(subs, dim=1).reshape(args.n_envs, -1)   # (n_envs, action_block*n_dof) = a_dim
            elif args.warmup_random and step < args.start_steps:
                a = torch.rand(args.n_envs, a_dim, device=device) * 2 - 1
            elif args.cem:                   # CEM open-loop receding horizon (LeWM WorldModelPolicy):
                # plan H block-actions toward goal latent z* via the WM, buffer them, execute one per
                # decision, and re-plan an env only when its buffer empties / it just reset / goals
                # refreshed this step (LeWM clears the plan deque on terminated / _needs_flush).
                need = cem_ptr >= cem_stride                     # replan stride (== cem_H unless --cem-replan-every)
                need |= is_start                                 # a just-reset env: stale plan + fresh goal
                if step % args.goal_update_every == 0:           # goals just refreshed for ALL envs
                    need[:] = True
                if need.any():
                    ridx = np.where(need)[0]
                    rt = torch.as_tensor(ridx, device=device)
                    cem_buf[rt] = cem_plan(wm, hist_z[:, rt], hist_a[:, rt], zstar_env[rt],
                                           args.cem_samples, args.cem_iters, args.cem_elites,
                                           args.cem_init_std, args.cem_horizon, device, diag=cem_diag,
                                           gamma=args.cem_gamma, min_std=args.cem_min_std,
                                           mppi_temp=args.cem_mppi_temp)
                    cem_ptr[ridx] = 0
                a = cem_buf[torch.arange(args.n_envs, device=device),
                            torch.as_tensor(cem_ptr, device=device)].clamp(-1.0, 1.0)  # safety: bound to trained range
                cem_ptr += 1
            elif args.goal_explore:
                a = actor(z, zstar_env)      # deterministic goal-conditioned action a=tanh(mu(z,z*))
            elif stochastic_act:
                a, _, _ = actor.sample(z)
            else:
                a = actor(z)                 # deterministic mean (deployment / eval)
            if args.explore_noise > 0:       # optional extra collection noise (sim pretrain only)
                a = (a + args.explore_noise * torch.randn_like(a)).clamp(-1.0, 1.0)
        if args.action_max_warmup_steps > 0:
            if step < args.action_max_warmup_steps:
                frac = args.action_max_start_frac + (args.action_max_end_frac - args.action_max_start_frac) \
                       * (step / args.action_max_warmup_steps)
            else:
                frac = args.action_max_end_frac          # hold the (possibly <1.0) ceiling after warmup
            a = a * frac                     # action_max schedule: ramp start_frac -> end_frac over warmup, then hold
        hist_a = torch.cat([hist_a[1:], a.unsqueeze(0)], 0)
        a_np = a.detach().cpu().numpy()
        a_env = a_np.reshape(args.n_envs, args.action_block, n_dof)

        # --- async actor-learner: launch the env rollout for this decision, then run
        #     the GPU updates on buffered data WHILE the workers render -> overlap CPU/GPU ---
        env.step_block_async(a_env)
        if not args.frozen_policy:                # frozen: skip ALL gradient work (data collection only)
            h_fwd = learner_updates(step, h_fwd)
        obs, sub_infos = env.step_block_wait()
        if args.no_torque_obs:
            scrub_torque_obs(obs, n_dof)

        # --- accumulate safety reward + interaction + smoothness stats over the action_block ---
        r_safe = np.zeros(args.n_envs, np.float32)
        contacts = np.zeros(args.n_envs, np.float32)
        table_contacts = np.zeros(args.n_envs, np.float32)
        motion = np.zeros(args.n_envs, np.float32)
        energy = np.zeros(args.n_envs, np.float32)      # mean_i |tau_i * qd_i| (mechanical power)
        qd_mean = np.zeros(args.n_envs, np.float32)
        tau_sat = np.zeros(args.n_envs, np.float32)
        qd_rev = np.zeros(args.n_envs, np.float32)      # sign-flip fraction of qd between substeps
        prev_qd = None
        for info in sub_infos:
            r_safe += info["safety_reward"]
            contacts += info["object_contacts"].astype(np.float32)
            table_contacts += info["table_contacts"].astype(np.float32)
            motion += info["object_motion"]
            tau, qd = info["applied_torque"], info["qvel"]
            energy += np.abs(tau * qd).mean(-1)
            qd_mean += np.abs(qd).mean(-1)
            tau_sat += (np.abs(tau) > 0.95 * tau_max_arr).mean(-1).astype(np.float32)
            if prev_qd is not None:
                qd_rev += ((qd * prev_qd) < 0).mean(-1).astype(np.float32)
            prev_qd = qd
        r_safe /= args.action_block      # one r_safe per decision (README: Env(a_t) -> r_safe_t)
        energy /= args.action_block
        qd_mean /= args.action_block
        tau_sat /= args.action_block
        qd_rev /= max(args.action_block - 1, 1)

        # --- action-rate (legged_gym-style smoothness): mean per-dim squared delta over
        #     consecutive sub-actions, including the boundary pair with the previous
        #     block's last sub-action. On an episode's first decision the boundary (and
        #     the 2nd-order term spanning it) is masked out — never penalize the
        #     cross-reset jump. Computed for every run (logged); enters the reward only
        #     when --w-action-rate/--w-action-rate2 > 0. ---
        seq = np.concatenate([prev_sub_a[:, None, :], a_env.astype(np.float64)], axis=1)  # (n_envs, 1+B, n_dof)
        d1 = np.diff(seq, axis=1)                          # (n_envs, B, n_dof)
        d2 = np.diff(seq, n=2, axis=1)                     # (n_envs, B-1, n_dof)
        sq1, sq2 = (d1 ** 2).mean(-1), (d2 ** 2).mean(-1)  # per-pair, per-dim mean
        m1, m2 = np.ones_like(sq1), np.ones_like(sq2)
        m1[is_start, 0] = 0.0
        m2[is_start, 0] = 0.0
        rate = ((sq1 * m1).sum(1) / m1.sum(1)).astype(np.float32)
        rate2 = ((sq2 * m2).sum(1) / np.maximum(m2.sum(1), 1.0)).astype(np.float32)
        prev_sub_a = a_env[:, -1].astype(np.float64).copy()

        # --- ACTUAL joint travel (is the arm parked or going somewhere?). pose_step =
        #     how far the joint vector moved THIS decision; ~0 = frozen/dithering in place.
        #     pose_spread/pose_range (logged below) = how much of config space the recent
        #     window covers. These read q directly, so they catch in-place jitter that
        #     qd_mean misses (high qd + ~0 pose_step = oscillating, not exploring). ---
        qpos_dec = sub_infos[-1]["qpos"].astype(np.float64)        # (n_envs, n_dof) final pose of the block
        if prev_qpos_dec is None:
            prev_qpos_dec = qpos_dec
        pose_step = np.where(is_start, 0.0,
                             np.linalg.norm(qpos_dec - prev_qpos_dec, axis=-1)).astype(np.float32)
        prev_qpos_dec = qpos_dec
        recent_qpos.append(qpos_dec.copy())

        # freeze a FIXED diverse probe set (early warmup obs) so encoder/eff_rank_probe
        # measures encoder health independent of how narrow the policy's behavior gets.
        if args.probe_size > 0 and probe_px is None:
            probe_buf.append((obs["image"].copy(), obs["proprio"].copy()))
            if len(probe_buf) * args.n_envs >= args.probe_size:
                probe_px = np.concatenate([p for p, _ in probe_buf])[:args.probe_size]
                probe_prop = np.concatenate([q for _, q in probe_buf])[:args.probe_size]
                probe_buf = None
                print(f"[probe] froze {len(probe_px)} obs for encoder/eff_rank_probe", flush=True)

        z_next = encode_obs(wm, obs["image"], obs["proprio"], device)
        step_jump_recent.append(float((z - z_next).norm(dim=-1).mean()))          # latent temporal locality probe
        r_cur = curiosity_reward(wm, hist_z, hist_a, z_next).cpu().numpy()        # (n_envs,) >= 0
        cur_term = args.lambda_cur * np.log1p(r_cur)         # lambda_cur * symlog(r_cur)  (r_cur>=0)

        # --- goal archive: insert this transition (goal = surprising OUTCOME o_{t+1}) scored by
        #     its TRUE one-step MSE r_cur; periodically re-measure every goal with the current WM
        #     and evict mastered ones. obs is still o_{t+1} here (done envs reset further below). ---
        if archives is not None:
            # ref = same-construction re-measure of THIS transition (eviction baseline); score = true r_cur
            ref = score_obs_mse(wm, obs["image"], obs["proprio"], cur_px, cur_prop, a_np, device, H)
            if args.per_env_archive:                        # route env e's transition to ITS buffer
                for e in range(args.n_envs):
                    archives[e].insert_batch(gpx=obs["image"][e:e + 1], gprop=obs["proprio"][e:e + 1],
                                             spx=cur_px[e:e + 1], sprop=cur_prop[e:e + 1],
                                             sact=a_np[e:e + 1], score=r_cur[e:e + 1], ref=ref[e:e + 1])
            else:                                           # global pool: all envs' transitions in one batch
                archives[0].insert_batch(gpx=obs["image"], gprop=obs["proprio"],
                                         spx=cur_px, sprop=cur_prop, sact=a_np, score=r_cur, ref=ref)
            if step > 0 and step % args.goal_rescore_every == 0:
                _scorer = (lambda gpx, gprop, spx, sprop, sact:
                           score_obs_mse(wm, gpx, gprop, spx, sprop, sact, device, H))
                for a in archives:
                    if a.n > 0:
                        goal_evictions += a.rescore_and_evict(_scorer, args.goal_drop_frac)
        r_rate = -(args.w_action_rate * rate + args.w_action_rate2 * rate2)       # smoothness penalties (0 unless flagged)
        r_energy = -args.w_energy * energy
        safe_term = args.lambda_safe * r_safe
        # --- intrinsic exploration bonuses (COMPOSABLE; each added iff its weight != 0, so the
        #     default reward is byte-identical to before) ---
        if args.lambda_rnd:                                   # RND novelty over the raw next-obs (frozen target)
            rnd_img, rnd_prop = to_rnd_obs(obs["image"], obs["proprio"],
                                           rnd_img_rms, rnd_prop_rms, device, update=True)
            r_rnd = rnd_novelty_reward(rnd_pred, rnd_target, rnd_img, rnd_prop).cpu().numpy()
            # scale the raw error into log1p's active range (obs-RND error ~5e-3 sits in log1p's
            # linear dead-zone) so symlog actually compresses and lambda_rnd stays O(10..20).
            rnd_term = args.lambda_rnd * np.log1p(args.rnd_reward_scale * r_rnd)
        else:
            r_rnd = np.zeros_like(r_cur); rnd_term = np.zeros_like(r_cur)
        if args.lambda_knn:                                   # k-NN coverage / state-entropy (alt to RND)
            r_knn = knn_state_entropy_reward(z_next, knn_buf, args.knn_k).cpu().numpy()
            knn_buf = torch.cat([knn_buf, z_next.detach()], 0)[-args.knn_buffer:]    # ring of recent latents
        else:
            r_knn = np.zeros_like(r_cur)
        knn_term = args.lambda_knn * r_knn
        if args.goal_explore:
            # buffer reward = the goal-INDEPENDENT curiosity term only; the dense reach term
            # -||z_{t+1}-z*|| is (re)computed in sac_update so HER relabeling takes effect.
            reward = cur_term
        else:
            reward = safe_term + cur_term + r_rate + r_energy + rnd_term + knn_term
        comps = np.stack([cur_term, safe_term, r_rate, r_energy], -1).astype(np.float32)  # REWARD_COMPONENTS order

        # contact-conditioned curiosity MSE (r_cur = ||zhat-z||^2): is the model more
        # surprised poking a block than scraping the table or moving in free space?
        # buckets are exclusive, classified block > table > neither (the user's order).
        touch_block = contacts > 0
        touch_table = (~touch_block) & (table_contacts > 0)
        for mask, key in ((touch_block, "mse_block"), (touch_table, "mse_table"),
                          (~(touch_block | touch_table), "mse_none")):
            if mask.any():
                recent_mse[key].extend(r_cur[mask].tolist())

        # --- store transition (truncation-as-done time limit) ---
        ep_len += 1
        ep_ret += reward
        done = (ep_len >= args.max_episode_steps).astype(np.float32)
        if args.goal_explore:                # store the PURSUED goal o* (self-goal where none yet)
            g_px = np.where(has_goal[:, None, None, None], goal_px_env, cur_px)
            g_prop = np.where(has_goal[:, None], goal_prop_env, cur_prop)
            g_valid = has_goal.copy()        # reach reward only counts where a real goal was pursued
        else:
            g_px = g_prop = g_valid = None
        buf.add(pixels=cur_px, proprio=cur_prop, action=a_np,
                r=reward, d=done, is_start=is_start.copy(), prio=r_cur,
                rc=comps if args.multihead_q else None,
                goal_px=g_px, goal_prop=g_prop, goal_valid=g_valid)
        for key, val in (("r_cur", r_cur), ("r_safe", r_safe), ("cur_contrib", cur_term),
                         ("contacts", contacts), ("table_contacts", table_contacts),
                         ("motion", motion), ("ret", reward),
                         ("frac_block", touch_block), ("frac_table", touch_table),
                         ("rate", rate), ("rate2", rate2), ("energy", energy),
                         ("qd_mean", qd_mean), ("tau_sat", tau_sat), ("qd_rev", qd_rev),
                         ("r_rate", r_rate), ("r_energy", r_energy), ("pose_step", pose_step),
                         ("r_rnd", r_rnd), ("rnd_contrib", rnd_term),
                         ("r_knn", r_knn), ("knn_contrib", knn_term)):
            recent[key].append(float(np.mean(val)))

        # --- advance latent + history; reset timed-out envs ---
        z = z_next
        hist_z = torch.cat([hist_z[1:], z_next.unsqueeze(0)], 0)
        is_start = done > 0                                       # next o_t is a start where we reset
        done_envs = np.where(done > 0)[0]
        if len(done_envs):
            for e in done_envs:
                o_e = env.reset_one(int(e))
                if args.no_torque_obs:
                    scrub_torque_obs(o_e, n_dof)
                obs["image"][e] = o_e["image"]
                obs["proprio"][e] = o_e["proprio"]
            z_reset = encode_obs(wm, obs["image"][done_envs], obs["proprio"][done_envs], device)
            z[done_envs] = z_reset
            for j, e in enumerate(done_envs):
                hist_z[:, e] = z_reset[j]
                hist_a[:, e] = 0.0
            if args.goal_explore:                  # draw a fresh goal for each reset env
                has_goal[done_envs] = False
                refresh_goals([int(e) for e in done_envs])
            if run is not None:
                wlog({"episode/return": float(ep_ret[done_envs].mean()),
                      "episode/len": float(ep_len[done_envs].mean())}, step)
            ep_len[done_envs] = 0
            ep_ret[done_envs] = 0.0

        # --- train videos: buffer per-env frames in the window before each save.
        #     wrist (what the policy sees) is free (already rendered as the obs);
        #     overhead is rendered from the training envs only inside the window
        #     (parallel across workers) -> ~video_steps renders per video_every. ---
        if video_on and 0 < step % args.video_every \
                and step % args.video_every >= args.video_every - args.video_steps:
            wrist_buf.append(tile_frames(obs["image"]))
            over_buf.append(tile_frames(env.render_overhead()))

        # --- logging ---
        if step % args.log_every == 0:
            sps = (step + 1) * args.n_envs / (time.time() - t0)
            safe_m, cur_m = np.mean(recent["r_safe"]), np.mean(recent["cur_contrib"])
            d = {"reward/r_cur": np.mean(recent["r_cur"]),
                 "reward/r_safe": safe_m,
                 "reward/cur_contrib": cur_m,                 # lambda_cur * symlog(r_cur)
                 "reward/safe_cur_ratio": abs(safe_m) / max(abs(cur_m), 1e-6),
                 "reward/total": np.mean(recent["ret"]),
                 "interact/contacts_per_step": np.mean(recent["contacts"]),
                 "interact/table_contacts_per_step": np.mean(recent["table_contacts"]),
                 "interact/object_motion": np.mean(recent["motion"]),
                 "interact/frac_touch_block": np.mean(recent["frac_block"]),
                 "interact/frac_touch_table": np.mean(recent["frac_table"]),
                 "smooth/action_rate": np.mean(recent["rate"]),
                 "smooth/action_rate2": np.mean(recent["rate2"]),
                 "smooth/energy": np.mean(recent["energy"]),
                 "smooth/qd_mean": np.mean(recent["qd_mean"]),
                 "smooth/tau_sat_frac": np.mean(recent["tau_sat"]),
                 "smooth/qd_reversal_frac": np.mean(recent["qd_rev"]),
                 "explore/pose_step": np.mean(recent["pose_step"]),     # joint travel/decision; ~0 = parked
                 "buffer/transitions": buf.total, "perf/steps_per_sec": sps}
            # reward-component breakdowns only when that term is enabled (all off in CEM -> omitted)
            if args.lambda_rnd:
                d["reward/r_rnd"] = np.mean(recent["r_rnd"]); d["reward/rnd_contrib"] = np.mean(recent["rnd_contrib"])
            if args.lambda_knn:
                d["reward/r_knn"] = np.mean(recent["r_knn"]); d["reward/knn_contrib"] = np.mean(recent["knn_contrib"])
            if args.w_action_rate:
                d["reward/r_rate"] = np.mean(recent["r_rate"])
            if args.w_energy:
                d["reward/r_energy"] = np.mean(recent["r_energy"])
            if args.h_fwd_max > 1:           # WM-rollout curriculum stage (constant 1 when curriculum off)
                d["wm/h_fwd"] = h_fwd
            if args.cem and cem_diag:        # CEM action-sensitivity probe (computed in cem_plan, iter 0)
                d["cem/cost_cv"] = cem_diag.get("cost_cv", float("nan"))             # ~0 => no optimization signal
                d["cem/z_term_spread"] = cem_diag.get("z_term_spread", float("nan")) # how far actions move the endpoint
                d["cem/reach_gap"] = cem_diag.get("reach_gap", float("nan"))         # current ||z - z*||
                d["cem/finite_frac"] = cem_diag.get("finite_frac", float("nan"))     # frac of CEM candidates rolled out finite
                _rg = cem_diag.get("reach_gap", 0.0)
                d["cem/move_vs_gap"] = (cem_diag["z_term_spread"] / _rg) if _rg > 1e-9 else float("nan")
                d["cem/min_cand_to_goal"] = cem_diag.get("min_cand_to_goal", float("nan"))  # best CONVERGED candidate->goal dist
                d["cem/endpoint_disp"] = cem_diag.get("endpoint_disp", float("nan"))        # mean converged endpoint move from z_now
            if args.goal_explore:
                d["goal/archive_size"] = sum(a.n for a in archives)        # total across buffer(s)
                _ms = [a.mean_score() for a in archives if a.n]
                d["goal/archive_mse_mean"] = float(np.mean(_ms)) if _ms else 0.0
                d["goal/evictions"] = goal_evictions
                if goal_recent["dist"]:
                    d["goal/dist_to_goal"] = float(np.mean(goal_recent["dist"]))
                if goal_recent["dist_env"]:                      # per-env ||z - z*|| (trailing-window mean, comparable to goal/dist_to_goal)
                    de = np.stack(goal_recent["dist_env"])       # (T, n_envs); NaN on steps an env had no real goal
                    cnt = np.sum(np.isfinite(de), axis=0)        # real-goal samples per env in the window (avoids nanmean empty-slice warning)
                    sm = np.nansum(de, axis=0)
                    for i in range(args.n_envs):
                        if cnt[i] > 0:
                            d[f"goal/dist_env/{i:02d}"] = float(sm[i] / cnt[i])
                if goal_recent["reach"]:
                    d["goal/reach_rate"] = float(np.mean(goal_recent["reach"]))
                if args.goal_curriculum:
                    d["goal/curric_k"] = curric_k[0]              # current reachable-radius curriculum offset
                    if args.goal_select == "highmse_under_d":
                        d["goal/curric_d"] = curric_d[0]          # current MSE-buffer latent-distance budget
                if goal_recent["qpos_dist"]:                       # joint-space (physical) ground-truth
                    d["goal/qpos_dist"] = float(np.mean(goal_recent["qpos_dist"]))           # mean max-joint err (rad)
                    d["goal/success_rate_qpos"] = float(np.mean(goal_recent["qpos_reach"]))  # frac within eps on every joint
                if args.goal_reach_eps_start > 0:                   # only varies when annealing is on
                    d["goal/reach_eps"] = float(reach_eps_now(step))
            if recent_qpos:    # how much of joint space the recent window covers (parked -> ~0)
                qarr = np.stack(recent_qpos)                            # (T, n_envs, n_dof)
                d["explore/pose_spread"] = float(qarr.std(0).mean())    # mean temporal std over joints/envs
                d["explore/pose_range"] = float((qarr.max(0) - qarr.min(0)).mean())  # mean per-joint sweep
            if step_jump_recent:    # latent TEMPORAL LOCALITY: mean ||z_t - z_{t+1}|| and its fraction of the
                sj = float(np.mean(step_jump_recent))               # random-pair distance sqrt(2*z_dim)*z_std (z_std~1 post-warmup).
                d["encoder/step_jump"] = sj                         # frac ~1.0 = NO locality (LeWM cube was ~0.09); small = local.
                d["encoder/step_jump_frac_rand"] = sj / (2 * z_dim) ** 0.5
            for key in ("mse_block", "mse_table", "mse_none"):   # curiosity MSE by contact type
                if recent_mse[key]:
                    d[f"wm/{key}"] = float(np.mean(recent_mse[key]))
            # signed contact gap: >0 = empty space is MORE novel than blocks (the inversion that
            # actively steers the policy AWAY from contact). One watchable line vs eyeballing two.
            if "wm/mse_block" in d and "wm/mse_none" in d:
                d["wm/mse_contact_gap"] = d["wm/mse_none"] - d["wm/mse_block"]
            # reward-variance split: how much of the (windowed) reward variance is action-relevant
            # curiosity vs safety. var_cur_frac ->0 = safety owns all the signal SAC can act on
            # (the freeze mechanism), even if r_cur is large in absolute terms.
            cur_v = float(np.var(recent["cur_contrib"]))
            safe_v = float(np.var(np.asarray(recent["r_safe"], np.float64) * args.lambda_safe))
            d.update({"reward/var_cur": cur_v, "reward/var_safe": safe_v,
                      "reward/var_cur_frac": cur_v / max(cur_v + safe_v, 1e-12)})
            if probe_px is not None and step % args.probe_every == 0:   # encoder health on a FIXED probe set
                with torch.no_grad():        # 2048-obs inference encode: MUST NOT build a graph (encode_obs
                                             # lacks @no_grad) and runs only every --probe-every steps -- rank
                                             # moves slowly, and encoding the full probe every log step (with
                                             # grad) cost ~half the SPS.
                    p_std, p_rankme = collapse_metrics(encode_obs(wm, probe_px, probe_prop, device))
                d.update({"encoder/rank_frac_probe": p_rankme / z_dim, "encoder/z_std_probe": p_std})
            if last_wm is not None:
                d.update({"wm/pred_loss": last_wm[0], "wm/sigreg": last_wm[1],
                          "wm/identity_baseline": last_wm[2], "wm/mean_baseline": last_wm[3],
                          "wm/pred_vs_persist": last_wm[0] / max(last_wm[2], 1e-9),   # ~1.0 = persistence-collapse
                          "wm/pred_vs_mean": last_wm[0] / max(last_wm[3], 1e-9)})      # ~1.0 = mean-collapse
            if last_sac is not None:
                d.update({"sac/critic_loss": last_sac[0], "sac/actor_loss": last_sac[1],
                          "smooth/grad_caps": last_sac[2]})
            if last_rnd is not None:
                d["rnd/pred_loss"] = last_rnd                  # RND predictor MSE (should fall as states are visited)
            if last_qh is not None:              # --multihead-q: per-component policy value
                d.update({f"sac/q_{k}": float(v) for k, v in zip(REWARD_COMPONENTS, last_qh)})
            if args.goal_explore:                # no safety in goal-explore -> drop its (zeroed) keys
                for _k in ("reward/r_safe", "reward/safe_cur_ratio", "reward/var_safe",
                           "reward/var_cur_frac"):
                    d.pop(_k, None)
            wlog(d, step)
            with open(out_dir / "metrics.jsonl", "a") as f:    # local metrics record (esp. when --no-wandb)
                f.write(json.dumps({"step": step, **{k: float(v) for k, v in d.items()}}) + "\n")
            if args.goal_explore:           # goal-explore has NO safety: show goal metrics instead
                print(f"[step {step}] cur_contrib={cur_m:.3f} "
                      f"dist_goal={d.get('goal/dist_to_goal', float('nan')):.2f} "
                      f"reach={d.get('goal/reach_rate', 0.0):.2f} "
                      f"archive={sum(a.n for a in archives)} evict={goal_evictions} "
                      f"crit_loss={d.get('sac/critic_loss', float('nan')):.2f} "
                      f"contacts/s={d['interact/contacts_per_step']:.2f} "
                      f"mse[blk/tbl/none]={d.get('wm/mse_block', float('nan')):.2f}/"
                      f"{d.get('wm/mse_table', float('nan')):.2f}/{d.get('wm/mse_none', float('nan')):.2f} "
                      f"pose_step={d['explore/pose_step']:.3f} "
                      f"pose_range={d.get('explore/pose_range', float('nan')):.2f} "
                      f"h_fwd={h_fwd} sps={sps:.1f}", flush=True)
            else:
                print(f"[step {step}] r_safe={safe_m:.3f} cur_contrib={cur_m:.3f} "
                      f"safe:cur={d['reward/safe_cur_ratio']:.2f} "
                      f"contacts/s={d['interact/contacts_per_step']:.2f} "
                      f"mse[blk/tbl/none]={d.get('wm/mse_block', float('nan')):.2f}/"
                      f"{d.get('wm/mse_table', float('nan')):.2f}/{d.get('wm/mse_none', float('nan')):.2f} "
                      f"rate={d['smooth/action_rate']:.2f} pose_step={d['explore/pose_step']:.3f} "
                      f"pose_range={d.get('explore/pose_range', float('nan')):.2f} "
                      f"h_fwd={h_fwd} sps={sps:.1f}", flush=True)

        # --- checkpoint: upload to HF then clear from disk (bounded disk over a long run) ---
        if args.save_every > 0 and step > 0 and step % args.save_every == 0:
            state = {"step": step, "wm": wm.state_dict(), "actor": actor.state_dict(),
                     "critic": critic.state_dict(), "critic_tgt": critic_tgt.state_dict(),
                     "h_fwd": h_fwd, "args": vars(args)}
            if archives is not None:
                state["goal_archive"] = [a.state_dict() for a in archives]
            save_and_upload(state, out_dir, step,
                            args.hf_repo or os.environ.get("HF_UPLOAD_REPO_ID"),
                            run_name, not args.no_hf, args.keep_local_ckpts)

        # --- train videos: save the buffered wrist + overhead clips (every video_every) ---
        if video_on and step > 0 and step % args.video_every == 0:
            roll_dir = out_dir / "rollouts"; roll_dir.mkdir(exist_ok=True)
            for tag, frames in (("wrist", wrist_buf), ("overhead", over_buf)):
                if not frames:
                    continue
                vp = roll_dir / f"train_{tag}_{step:07d}.mp4"
                try:
                    imageio.mimsave(vp, list(frames), fps=args.video_fps)
                    if run is not None:
                        wlog({f"train/{tag}": wandb.Video(str(vp), format="mp4")}, step)
                        vp.unlink(missing_ok=True)   # in W&B now; keep local disk clean
                except Exception as ex:
                    print(f"[video] {tag} failed (non-fatal): {ex}", flush=True)
            wrist_buf.clear(); over_buf.clear()

    # --- final: collected data (frozen / --save-buffer) and/or model checkpoint ---
    if args.frozen_policy or args.save_buffer:
        save_buffer(buf, out_dir)
    if not args.frozen_policy:                    # frozen: weights are unchanged, skip the re-upload
        state = {"step": args.total_steps, "wm": wm.state_dict(), "actor": actor.state_dict(),
                 "critic": critic.state_dict(), "critic_tgt": critic_tgt.state_dict(),
                 "h_fwd": h_fwd, "args": vars(args)}
        if archives is not None:
            state["goal_archive"] = [a.state_dict() for a in archives]
        save_and_upload(state, out_dir, args.total_steps,
                        args.hf_repo or os.environ.get("HF_UPLOAD_REPO_ID"),
                        run_name, not args.no_hf, args.keep_local_ckpts)
    env.close()
    if run is not None:
        run.finish()
    print("[done]", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description="Curious Robot: JEPA+SIGReg WM + SAC curiosity on SO-ARM101")
    # schedule / infra
    p.add_argument("--total-steps", type=int, default=200_000)
    p.add_argument("--start-steps", type=int, default=-1,
                   help="update warmup (decision steps): no WM/SAC gradient steps before this; acting is "
                        "ALWAYS the deterministic policy (the random-action warmup was removed with the "
                        "rest of the policy stochasticity). Default-aware: 0 on hardware, 1000 in sim")
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--env-threads", type=int, default=0,
                   help=">0 steps envs on a thread pool (inproc backend only)")
    p.add_argument("--env-backend", choices=("subproc", "inproc", "hardware"), default="subproc",
                   help="subproc: each env in a CUDA-free worker process, needed on "
                        "GPU+EGL to avoid the MuJoCo-render/CUDA SIGABRT; "
                        "inproc: envs in this process (sequential or --env-threads); "
                        "hardware: one physical SO-ARM101 via env/hardware_env.py (forces n_envs=1)")
    p.add_argument("--frame-skip", type=int, default=6)
    p.add_argument("--fixed-objects", action="store_true",
                   help="place all scene objects at an IDENTICAL deterministic layout for every env and "
                        "every reset (constant-seed positions/sizes/colors/orientations, placed against a "
                        "fixed arm pose). Collapses scene variance to just the arm -> a much easier "
                        "(LeWM-cube-like) target for the encoder/WM. Sim only.")
    p.add_argument("--render-backend", choices=("egl", "osmesa"), default="egl",
                   help="offscreen render backend for the CUDA-free subproc env workers. 'egl' (default) = "
                        "GPU offscreen render, ~0.3ms vs osmesa's ~35ms per 224^2 frame (~100x; overhead-cam "
                        "sps ~2.5 -> ~30+). The worker EGL init is made deterministic in parallel_env.py "
                        "(pinned MUJOCO_EGL_DEVICE_ID + forced NVIDIA ICD so no Mesa-software device is "
                        "enumerated, + a flock serialising the first eglInitialize across workers). 'osmesa' = "
                        "CPU offscreen fallback (no GPU / EGL unavailable). Subproc backend only (inproc "
                        "renders in the CUDA main proc -> osmesa).")
    p.add_argument("--wm-cam", choices=("wrist", "overhead"), default="wrist",
                   help="which camera the encoder/WM sees (obs['image']). 'wrist' (default) = egocentric, "
                        "moves with the arm -> latent jumps ~a full random-pair distance per step (breaks "
                        "latent L2 planning). 'overhead' = fixed third-person worldbody cam (LeWM-style), a "
                        "PROTOTYPE for a temporally-smooth goal-reaching latent. Sim only.")
    p.add_argument("--max-episode-steps", type=int, default=200, help="decision steps before truncation-as-done")
    p.add_argument("--seed", type=int, default=0)
    # resume / warm-start (train.py otherwise never loads a checkpoint)
    p.add_argument("--init-ckpt", default=None,
                   help="local .pt to warm-start wm+actor+critic+critic_tgt+h_fwd before the loop")
    p.add_argument("--resume-name", default=None,
                   help="resume from an HF run name (e.g. safe15) instead of a local --init-ckpt")
    p.add_argument("--resume-step", type=int, default=None,
                   help="checkpoint step for --resume-name (default: latest available)")
    p.add_argument("--name", default="baseline",
                   help="short run keyword; drives the W&B run name, runs/<name>/, and HF "
                        "<name>/ckpt_*.pt. Keep it short and identifiable -- every constant/var "
                        "lives in the W&B config table, not the name.")
    p.add_argument("--out-dir", default=None, help="local dir (default: runs/<name>)")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=1000,
                   help="HF-upload-then-clear-disk period (decision steps)")
    p.add_argument("--keep-local-ckpts", action="store_true",
                   help="keep the local .pt after a successful upload (default: delete to bound disk)")
    p.add_argument("--video-every", type=int, default=1000,
                   help="train-video period (decision steps): save a wrist + overhead clip every N; 0 disables")
    p.add_argument("--video-steps", type=int, default=60,
                   help="frames per train-video clip (window of decision steps before each save)")
    p.add_argument("--video-fps", type=int, default=20)
    p.add_argument("--probe-size", type=int, default=2048,
                   help="size of the fixed probe set for encoder/rank_frac_probe (RankMe) "
                        "(isolates encoder health from behavioral diversity); 0 disables. Must be >> z_dim "
                        "(256): at probe_size~=z_dim even a perfectly isotropic encoder reads ~0.5*D (PR) / "
                        "~0.8*D (RankMe) from finite-sample bias, so 256 (the old default) under-reported "
                        "rank ~2x. 2048 puts the RankMe ceiling at ~99%% of D.")
    p.add_argument("--probe-every", type=int, default=500,
                   help="recompute the probe rank metrics every N decision steps (must be a multiple of "
                        "--log-every). Decoupled from logging because encoding the full --probe-size set "
                        "through the ViT is costly; rank evolves slowly so 500 is plenty. Lower for finer "
                        "resolution at some SPS cost.")
    p.add_argument("--probe-id", default="probe_v1",
                   help="HF probe artifact id (probe/<id>.npz): canonical uniform-pose probe; "
                        "falls back to a warmup-rollout probe if unavailable")
    # action / actuation (README)
    p.add_argument("--action-block", type=int, default=5)
    p.add_argument("--action-max", type=float, default=0.3,
                   help="README dq^max: rad of joint delta per unit tanh action")
    # world model (README; the '?' values below are sweepable, not pinned in README)
    p.add_argument("--no-compile", action="store_true",
                   help="disable torch.compile(wm.predict) (the CEM-rollout speedup); use if compile "
                        "is flaky on this box. CUDA only; compile is on by default there.")
    p.add_argument("--history-size", type=int, default=3, help="H_bwd")
    p.add_argument("--no-proprio", action="store_true",
                   help="LeWM-faithful PIXELS-ONLY state: drop the proprio branch from the encoder so "
                        "z = MLP(ViT_cls) alone (D = vis_dim = 192 = LeWM embed_dim, vs the 256-d "
                        "image+proprio fusion). Proprio is still collected (safety/goals/RND) but NOT fed "
                        "to the WM encoder. Pair with --sigreg-pertimestep for the exact LeWM objective.")
    p.add_argument("--freeze-encoder", action="store_true",
                   help="stop-gradient on the obs->z encoder (StateEncoder): freeze the latent SPACE so "
                        "CEM plans against a STATIONARY geometry (LeWM-style frozen WM; stops the SIGReg "
                        "latent-scale drift that makes ||z-z*|| meaningless). Predictor + action_encoder keep "
                        "training to sharpen rollouts in that fixed latent. Intended WITH --resume-name/"
                        "--init-ckpt (else it freezes a random encoder). Excluded from the WM optimizer and "
                        "kept in eval() so the encoder is fully deterministic.")
    p.add_argument("--h-fwd-start", type=int, default=1)
    p.add_argument("--h-fwd-max", type=int, default=1,
                   help="max forward rollout horizon; ==start (1) pins the WM to 1-step-ahead "
                        "prediction and disables the H_fwd curriculum")
    p.add_argument("--h-fwd-override", type=int, default=0,
                   help="force the WM-rollout-training horizon to this, IGNORING the value pinned by a resumed "
                        "checkpoint (resume normally restores the ckpt's h_fwd stage, so --h-fwd-start is ignored on "
                        "warm-start). Use with --h-fwd-max>=override to train multi-step rollouts on a 1-step ckpt.")
    p.add_argument("--gamma-wm", type=float, default=0.95)
    p.add_argument("--sigreg-weight", type=float, default=0.09,
                   help="beta: SIGReg (isotropic-Gaussian) weight, pinned at 0.09 (LeWM's "
                        "lewm.yaml value; the old 0.3 made beta*sig dominate the WM loss ~5:1 "
                        "over pred_loss)")
    p.add_argument("--wm-batch-size", type=int, default=128)
    p.add_argument("--encoder-thaw-every", type=int, default=0,
                   help="INTERLEAVED freeze/thaw: with --freeze-encoder, periodically UNFREEZE the encoder for "
                        "--encoder-thaw-dur steps every N steps so the representation co-adapts to directed data "
                        "(vs the default continuous freeze). 0 = stay continuously frozen. Watch encoder/step_jump_"
                        "frac_rand: does periodic co-adaptation improve the latent, or does directed motion destroy locality?")
    p.add_argument("--encoder-thaw-dur", type=int, default=100,
                   help="duration (steps) of each thaw window for --encoder-thaw-every.")
    p.add_argument("--encoder-thaw-lr", type=float, default=0.0,
                   help="separate (typically smaller) LR for the encoder during thaw windows, so co-adaptation "
                        "NUDGES the latent instead of lurching it. 0 = use --wm-lr. Try ~1/10 of wm-lr.")
    p.add_argument("--encoder-thaw-beta", type=float, default=-1.0,
                   help="SIGReg weight to use DURING thaw windows (the diagnosed cause of thaw breaking locality is "
                        "SIGReg's isotropy scattering co-adapting frames). -1 = always use --sigreg-weight; set a "
                        "smaller value (or 0) so the encoder learns prediction-locality without isotropy scattering.")
    p.add_argument("--consolidate-every", type=int, default=0,
                   help="multi-epoch CONSOLIDATION (LeWM-regime test): every N collection steps, pause "
                        "stepping and train the WM for --consolidate-epochs full passes over the FROZEN "
                        "buffer (replaces the per-step wm_update_every/wm_grad_steps schedule). Pushes our "
                        "online single-pass regime toward LeWM's 100-epoch offline training -> lets pred_loss "
                        "carve a smooth manifold before SIGReg scatters it. 0 = off (normal online schedule).")
    p.add_argument("--consolidate-epochs", type=int, default=2,
                   help="epochs over the buffer per consolidation burst (--consolidate-every). "
                        "burst grad steps = epochs * (buffer_transitions // wm_batch_size).")
    p.add_argument("--lambda-slow", type=float, default=0.0,
                   help="slowness/temporal-coherence weight: penalize mean sq per-step latent jump "
                        "||z_t - z_{t+1}||^2 on real consecutive frames. Forces temporal locality "
                        "(SIGReg anchors against the dz->0 collapse). 0 = off.")
    p.add_argument("--action-max-start-frac", type=float, default=1.0,
                   help="action_max schedule: initial fraction of action amplitude (scales the normalized "
                        "action). Ramps linearly to --action-max-end-frac over --action-max-warmup-steps. "
                        "1.0 = off (when end-frac is also 1.0).")
    p.add_argument("--action-max-end-frac", type=float, default=1.0,
                   help="action_max schedule: FINAL fraction reached at --action-max-warmup-steps and HELD "
                        "for the rest of the run. <1.0 caps the effective action amplitude permanently (a "
                        "'stricter ending' / lower terminal action_max); 1.0 = ramp to full amplitude (default, "
                        "= old behavior).")
    p.add_argument("--action-max-warmup-steps", type=int, default=0,
                   help="steps to ramp effective action amplitude from --action-max-start-frac to "
                        "--action-max-end-frac, then hold end-frac. 0 = off (no schedule).")
    p.add_argument("--sigreg-pertimestep", action="store_true",
                   help="apply SIGReg per-timestep (T,B,D) like LeWM (isotropize each step's cross-trajectory "
                        "batch) instead of pooling the rollout window into (1,B*T,D). Pooling drags consecutive "
                        "frames into the same isotropy test -> pushes them apart -> kills temporal locality.")
    p.add_argument("--wm-lr", type=float, default=5e-5)
    p.add_argument("--wm-update-every", type=int, default=4)
    p.add_argument("--wm-grad-steps", type=int, default=1,
                   help="WM gradient steps taken per update opportunity (each on a fresh batch). >1 raises "
                        "the replay ratio without changing --wm-update-every. Effective WM updates/decision "
                        "= wm_grad_steps / wm_update_every.")
    p.add_argument("--wm-sample", choices=("uniform", "curiosity", "recency"), default="uniform",
                   help="how the WM batch is drawn from the replay buffer. 'uniform' (default): flat. "
                        "'curiosity': P ~ one-step-MSE^per_alpha (oversample the high-MSE states that "
                        "BECOME goals -> train the WM where it is worst). 'recency': favor freshly-collected "
                        "windows (exp half-life = 25%% of the buffer).")
    p.add_argument("--wm-dropout", type=float, default=0.1)
    # WM predictor sizing — defaults match LeWM (lewm/config/train/model/lewm.yaml). Runs before
    # 2026-06-26 trained a ~half-size predictor (heads 8 / dim-head 32 / mlp-dim 1024); pass those
    # values to reproduce/continue an old checkpoint with its predictor warm-started instead of re-init.
    p.add_argument("--wm-pred-depth", type=int, default=6, help="WM predictor transformer depth (LeWM: 6)")
    p.add_argument("--wm-pred-heads", type=int, default=16, help="WM predictor attention heads (LeWM: 16; was 8)")
    p.add_argument("--wm-pred-dim-head", type=int, default=64, help="WM predictor head dim (LeWM: 64; was 32)")
    p.add_argument("--wm-pred-mlp-dim", type=int, default=2048, help="WM predictor FFN hidden (LeWM: 2048; was 1024)")
    p.add_argument("--wm-grad-checkpoint", action="store_true",
                   help="enable ViT gradient checkpointing in the WM update (default off; trades ~10-15ms "
                        "recompute for memory — only worth it if WM-update activation memory is tight)")
    p.add_argument("--flatline-window", type=int, default=200)
    p.add_argument("--flatline-tol", type=float, default=0.03)
    # reward ('?' values; sweepable)
    p.add_argument("--lambda-safe", type=float, default=None,
                   help="weight on the safety penalty r_safe: r = lambda_safe*r_safe + lambda_cur*symlog(r_cur). "
                        "Default-aware (explicit value always wins): sim backends use 0.1, the hardware backend "
                        "uses 2.2 (2026-06-12 real-arm calibration: under delta=9 benign motion scores exactly 0, "
                        "so lambda_safe scales only genuine events — 2.2 makes one median user-labeled-bad substep "
                        "cancel ~one decision's curiosity term). 0 ablates safety.")
    p.add_argument("--lambda-cur", type=float, default=15.0,
                   help="curiosity weight on symlog(r_cur). r_cur is the per-dim MEAN squared pred error "
                        "(~O(0.1-1)); 15-20 keeps curiosity audible against raw |r_safe|~50 at lambda_safe 0.1. "
                        "Default 15 (2026-06-11; safe15/campaign history ran 20 — old default 1.0 silently "
                        "shrank curiosity 20x and caused one mis-deploy).")
    p.add_argument("--safety-delta", type=float, default=None,
                   help="delta: safety-reward deadband on the per-joint -tau*qddot (N*m*rad/s^2). Default-aware "
                        "(explicit value always wins): sim backends use 15, the hardware backend uses 9. "
                        "9 re-pinned 2026-06-12 from real-arm calibration at P8/D16 (true-dt args: all benign "
                        "motion incl. max-violence reversals <=7.4; user-labeled-bad grabs/blocks/jerks "
                        ">=10.7). SIM uses 15 with lambda_safe 0.1 — measured 2026-06-12 "
                        "(runs/sim_scales/kp499.json): sim's saturated PD torque puts even smooth motion at "
                        "args>9 on 33%% of joint-samples, so the real-arm (9, 2.2) pair freezes sim policies. "
                        "The pre-2026-06 0.05 penalized all motion -> policy froze; the old 15 never fired on hw.")
    # smoothness / transferability experiment knobs (2026-06-12 sim campaign; all default-off)
    p.add_argument("--w-action-rate", type=float, default=0.0,
                   help="weight W on the action-rate penalty -W * mean_dim (a_t - a_{t-1})^2 over consecutive "
                        "sub-actions incl. the block boundary (legged_gym-style; actions already in [-1,1]). "
                        "Episode-start boundary masked. Sim scale ref (kp499.json): dither ~1.42, smooth ~0.09 "
                        "-> W=3 puts dither at -4.3 vs cur_contrib ~+10 while smooth motion pays ~-0.3.")
    p.add_argument("--w-action-rate2", type=float, default=0.0,
                   help="weight on the 2nd-order action-rate term mean_dim (a_t - 2a_{t-1} + a_{t-2})^2 "
                        "(omega^4 rolloff). Off by default; try after the 1st-order term proves out.")
    p.add_argument("--w-energy", type=float, default=0.0,
                   help="weight W on the energy penalty -W * mean_substeps mean_i |tau_i * qd_i| (mechanical "
                        "power; N*m*rad/s). Sim scale ref: dither ~3.3, smooth ~1.4 -> W=1 is a balanced trial. "
                        "Hardware analogue exists since kt=10 current-torque (2026-06-06).")
    # intrinsic exploration reward terms -- COMPOSABLE: each is added to the reward iff its
    # weight != 0, so they stack with curiosity (e.g. --lambda-cur 15 --lambda-rnd 20 = curiosity + RND).
    p.add_argument("--lambda-rnd", type=float, default=0.0,
                   help="weight on the RND novelty reward: lambda_rnd*log1p(rnd_reward_scale*||pred(o)-target(o)||^2); "
                        "0 disables (default). Random-network distillation over the RAW obs o (downsampled wrist "
                        "image + proprio, NOT the co-trained z): predictor chases a FROZEN random net -> high for "
                        "novel obs, ->0 as visited; stable input + stationary target (no noisy-TV, no encoder "
                        "drift). Add to the curiosity reward by setting lambda_rnd>0 alongside lambda_cur.")
    p.add_argument("--rnd-out-dim", type=int, default=128, help="RND target/predictor output dim")
    p.add_argument("--rnd-hidden", type=int, default=256, help="RND proprio-MLP / head hidden dim")
    p.add_argument("--rnd-lr", type=float, default=5e-5,
                   help="RND predictor Adam lr (lower than a latent-RND default: the obs target is fixed, so the "
                        "predictor should chase it gently to keep the novelty signal from collapsing too fast)")
    p.add_argument("--rnd-loss-clip", type=float, default=10.0,
                   help="clamp on the per-sample RND training-loss weight err/MSE (novelty-prioritized predictor "
                        "training): caps how much one high-error sample can dominate the batch. Large -> pure "
                        "err/MSE weighting; ->1 approaches a uniform (standard RND) loss.")
    p.add_argument("--rnd-reward-scale", type=float, default=200.0,
                   help="multiply the raw RND error before log1p so it lands in symlog's active range (obs-RND "
                        "error ~5e-3 is in log1p's linear dead-zone). Default 200 -> typical log1p(~1.0)~0.69, so "
                        "lambda_rnd~20 gives an O(10) bonus comparable to curiosity. Scales the REWARD only; "
                        "predictor training (err/MSE-weighted + Adam) is scale-invariant and untouched.")
    p.add_argument("--rnd-train-every", type=int, default=1,
                   help="train the RND predictor once every N SAC updates (default 1 = every update). >1 slows the "
                        "predictor so obs novelty doesn't collapse to ~0 within ~250 steps on the low-diversity "
                        "visited obs -> the novelty bonus persists through the exploration window.")
    p.add_argument("--lambda-knn", type=float, default=0.0,
                   help="weight on the k-NN coverage reward log(1+mean kNN latent dist); 0 disables (default; alt to RND)")
    p.add_argument("--knn-k", type=int, default=12, help="k for the k-NN state-entropy estimate")
    p.add_argument("--knn-buffer", type=int, default=4096,
                   help="size of the recent-latent ring buffer the k-NN reward measures novelty against")
    p.add_argument("--no-torque-obs", action="store_true",
                   help="zero the u^app slice of proprio (obs -> [q, qd, 0]); shapes unchanged so old ckpts "
                        "load. Removes the obs channel that is ~96%% saturated sign-bit on hw and the main "
                        "sim->real proprio mismatch.")
    p.add_argument("--multihead-q", action="store_true",
                   help="critic outputs one Q head per reward component (cur/safe/rate/energy), each trained "
                        "on its own TD target; the actor maximizes the sum (same optimum as the scalar critic). "
                        "Logs sac/q_<comp> for interpretability.")
    p.add_argument("--actor-rate-reg", type=float, default=0.0,
                   help="action-rate as an ACTOR-LOSS regularizer (vs --w-action-rate's reward term): "
                        "+W * mean (pi(z) sub-action diffs)^2 added to actor_loss. Keeps smoothness "
                        "pressure out of r and Q — the A/B for whether the reward-term variant "
                        "pollutes the curiosity balance.")
    p.add_argument("--lambda-temp", type=float, default=0.0,
                   help="Grad-CAPS temporal-smoothness weight lambda_T on the ACTOR loss: "
                        "+lambda_T * mean_k ||s_{k-1}-2s_k+s_{k+1}|| * tanh(1/(||s_{k+1}-s_{k-1}||+eps)) "
                        "over the deterministic-MEAN applied sub-action path across the t->t+1 boundary "
                        "[pi_mean(z_t)|pi_mean(z_{t+1})]. Unlike --actor-rate-reg / --w-action-rate (squared "
                        "velocity -> penalizes ALL motion -> parks), this pays only low-travel "
                        "curvature (in-place zigzag), leaving smooth wide ramps free. Q/r untouched; "
                        "0.0 = off (no extra actor-loss term, default code path unchanged).")
    p.add_argument("--grad-caps-eps", type=float, default=1e-2,
                   help="epsilon in the Grad-CAPS 1/(displacement+eps) factor (caps the in-place "
                        "blow-up before tanh; smaller eps = sharper jitter penalty). Only read when "
                        "--lambda-temp != 0.")
    p.add_argument("--warmup-random", action="store_true",
                   help="act with uniform random actions during start_steps (restores the pre-2026-06-12 "
                        "warmup as an opt-in): extra buffer diversity for from-scratch sim runs; acting "
                        "samples ~pi after warmup in sim (deterministic mean on hardware).")
    p.add_argument("--explore-noise", type=float, default=0.0,
                   help="extra Gaussian noise std added on TOP of the action during COLLECTION only (clamped "
                        "to [-1,1]); eval/hardware act with the deterministic mean. Mostly redundant now that "
                        "sim training samples ~pi (2026-06-14); kept as an extra sim-pretrain knob. Not for "
                        "hardware.")
    p.add_argument("--collect-smooth", action="store_true",
                   help="SCRIPTED SMOOTH COLLECTOR (data-hypothesis test): replace the policy/CEM with a "
                        "sub-action-level OU correlated random walk (a_k = beta*a_{k-1} + (1-beta)*N(0,1), "
                        "clamped, persisted across decisions so block boundaries stay smooth). Generates "
                        "GUARANTEED-smooth joint trajectories with NO curiosity/CEM in the loop -> isolates "
                        "'smooth data -> local latent?'. Trains WM-only (skips sac_update, like --cem), so the "
                        "encoder is shaped purely by the JEPA+SIGReg loss on smooth data (mirrors LeWM).")
    p.add_argument("--smooth-beta", type=float, default=0.9,
                   help="OU smoothness for --collect-smooth: higher = smoother (slower walk). 0.9 ~ LeWM-ish "
                        "per-step action delta; tune via smooth/action_rate (LeWM ref ~0.09).")
    # actor-critic (README). SAC: stochastic train + entropy alpha, deterministic-mean deploy (2026-06-14).
    p.add_argument("--alpha", type=float, default=0.2,
                   help="fixed SAC entropy temperature: actor maximizes Q + alpha*H (re-added 2026-06-14). "
                        "The entropy bonus keeps the policy off the freeze attractor the 2026-06-12 "
                        "deterministic actor parked on. Sim training samples ~pi; eval/hardware act with the "
                        "deterministic mean tanh(MLP(z)). safe15 ran 0.2.")
    p.add_argument("--deterministic-act", action="store_true",
                   help="force the deterministic mean even during sim-training collection (default: sample ~pi "
                        "in sim, mean on hardware/frozen). Reproduces the 2026-06-12 deterministic acting or a "
                        "clean eval rollout.")
    # --- goal-conditioned Go-Explore (gated; default OFF -> the default path stays byte-identical) ---
    p.add_argument("--goal-explore", action="store_true",
                   help="goal-conditioned Go-Explore: a DETERMINISTIC actor pi(a|z,z*) + Q(z,a,z*) chases goals "
                        "z* drawn (P(k)~MSE_k) from an archive of the highest one-step-WM-MSE states, with a "
                        "dense reach reward -||z-z*|| + HER, while r_cur stays live ('return, then explore'). "
                        "Forces deterministic acting (alpha=0, no explore-noise), drops r_safe (lambda_safe=0) "
                        "and the dq_max action scaling (action_max=1.0), and disables --multihead-q. Sim only.")
    p.add_argument("--goal-archive-size", type=int, default=64,
                   help="K: max goals kept in the archive (top-K by capture-time r_cur).")
    p.add_argument("--per-env-archive", action="store_true",
                   help="give EACH env its own goal archive (K each) instead of one global pool shared by "
                        "all envs. Each env then explores + targets only the high-MSE states IT visited -> "
                        "goals nearer the env's own trajectory (typically more reachable).")
    p.add_argument("--goal-update-every", type=int, default=0,
                   help="THE decisive knob: resample each env's goal every N decision steps (and on reset). "
                        "Want it slower than policy convergence, faster than full WM mastery. 0 -> one goal per "
                        "episode (= --max-episode-steps).")
    p.add_argument("--goal-rescore-every", type=int, default=0,
                   help="re-measure every archived goal with the CURRENT WM every N steps and evict any whose "
                        "MSE dropped below --goal-drop-frac of its capture score (the 're-measured -> evict "
                        "mastered' rule). 0 -> = --goal-update-every.")
    p.add_argument("--goal-drop-frac", type=float, default=0.5,
                   help="evict an archived goal when its re-measured one-step MSE falls below this fraction of "
                        "its capture score (the WM has learned it).")
    p.add_argument("--goal-temp", type=float, default=1.0,
                   help="softmax temperature for goal sampling P(k) ~ exp(score_k/temp) (NOT argmax).")
    p.add_argument("--goal-select", choices=("mse", "near", "future", "recent", "highmse_under_d"), default="mse",
                   help="goal source. 'mse' (default) = archive P(k)~softmax(score/temp), highest-WM-MSE "
                        "(most novel -> FARTHEST in latent). 'near' = archive P(k)~softmax(-||z_e-z_k||/temp), "
                        "NEAREST the env's current latent. 'future' = NOT the archive: the obs the agent "
                        "ACHIEVED --goal-future-k forward steps after a recent point in its OWN current "
                        "episode (HER 'future' for the controller) -- reachable by forward dynamics and "
                        "on-manifold, mirroring LeWM's goal_offset_steps. 'future' is the decisive test of "
                        "whether CEM can close a genuinely reachable goal. 'highmse_under_d' = MSE-BUFFER "
                        "curriculum: among buffer states within latent distance d (--goal-curric-d-*) of "
                        "z_now, pursue the HIGHEST current-WM-MSE one; grow d as latent reach is earned.")
    p.add_argument("--goal-future-k", type=int, default=10,
                   help="--goal-select future/recent: step offset of the goal (future=ahead of a random anchor, "
                        "recent=decisions AGO). Also the constant offset when the curriculum is OFF.")
    p.add_argument("--goal-curriculum", action="store_true",
                   help="reachable-radius curriculum: a pure goal-RANGE schedule (NO loss/architecture change; "
                        "objective stays latent goal/reach_rate, NEVER qpos). With --goal-select recent, start at "
                        "--goal-curric-start k and grow k by 1 (goal further back => larger reach_gap) every "
                        "--goal-curric-patience steps once windowed latent reach_rate >= --goal-curric-thresh, "
                        "capped at --goal-curric-max-k. Extends the planner's reachable radius as it earns it.")
    p.add_argument("--goal-curric-start", type=int, default=1, help="curriculum starting goal offset k (inside the closing radius).")
    p.add_argument("--goal-curric-max-k", type=int, default=20, help="curriculum cap on the goal offset k.")
    p.add_argument("--goal-curric-thresh", type=float, default=0.25,
                   help="grow the radius when the windowed LATENT reach_rate reaches this (qpos NOT involved).")
    p.add_argument("--goal-curric-patience", type=int, default=200, help="min steps between curriculum radius bumps.")
    p.add_argument("--goal-curric-d-start", type=float, default=6.0,
                   help="--goal-select highmse_under_d: starting latent-distance budget d (admit candidate goals "
                        "with ||z_cand - z_now|| < d). Grown like the k-curriculum as latent reach is earned.")
    p.add_argument("--goal-curric-d-step", type=float, default=1.0, help="grow d by this each curriculum advance (highmse_under_d).")
    p.add_argument("--goal-curric-d-max", type=float, default=22.0, help="cap on the distance budget d (~the random-pair latent ceiling).")
    p.add_argument("--goal-cand-n", type=int, default=256,
                   help="--goal-select highmse_under_d: # buffer transitions sampled + re-scored (current-WM MSE) per "
                        "goal refresh; the highest-MSE candidate within d is pursued.")
    p.add_argument("--her-frac", type=float, default=0.5,
                   help="fraction of SAC transitions whose goal is HER-relabeled to an achieved future obs from "
                        "the same episode (densifies the reach reward).")
    p.add_argument("--lambda-reach", type=float, default=1.0,
                   help="weight on the dense reach reward -||z_{t+1}-z*|| (curiosity keeps its own --lambda-cur). "
                        "The natural z-scale makes reach dominate far from the goal and r_cur dominate at it.")
    p.add_argument("--goal-reach-eps", type=float, default=2.0,
                   help="z-space distance below which a goal counts as REACHED (logged as goal/reach_rate). "
                        "This is the FINAL/floor value when --goal-reach-eps-start anneals down to it.")
    p.add_argument("--goal-reach-eps-start", type=float, default=0.0,
                   help="if > 0, linearly anneal the reach threshold from this (lenient) value DOWN to "
                        "--goal-reach-eps over --goal-reach-eps-anneal-frac of training. Lets early "
                        "'approximate reaches' register (the WM's 1-step precision floor is ~7-9 z-units, "
                        "well above the 2.0 floor) and tightens over time. 0 = constant (default, unchanged). "
                        "NOTE: reach_rate is a DIAGNOSTIC only -- it does not affect collection or planning.")
    p.add_argument("--goal-reach-eps-anneal-frac", type=float, default=0.5,
                   help="fraction of --total-steps over which --goal-reach-eps-start anneals to --goal-reach-eps.")
    p.add_argument("--goal-success-qpos-eps", type=float, default=0.05,
                   help="LeWM-reacher-style PHYSICAL success threshold (radians): a goal counts as reached "
                        "when EVERY joint is within this many rad of the goal's stored qpos "
                        "(logged as goal/success_rate_qpos + goal/qpos_dist). LeWM's reacher uses 0.05. "
                        "Joint-space analogue of the latent goal/reach_rate; DIAGNOSTIC only (does not "
                        "affect planning/collection).")
    p.add_argument("--td3-target-noise", type=float, default=0.0,
                   help="optional TD3 target-policy smoothing: std of clipped noise added to the TARGET action in "
                        "the critic update only (not a collection action). 0 = pure deterministic (DDPG).")
    p.add_argument("--td3-noise-clip", type=float, default=0.5, help="clip for --td3-target-noise.")
    # --- CEM-MPC controller in LATENT space (--cem): goal-reaching by PLANNING, not a learned policy.
    #     Reward stays MSE-only; SAC is disabled; the WM provides the latent dynamics. Implies
    #     --goal-explore (high-MSE goal-latent archive + per-episode goal sampling). ---
    p.add_argument("--cem", action="store_true",
                   help="latent CEM-MPC controller: each step, optimize the block action to minimize "
                        "||zhat_{t+1} - z*|| (z* = goal latent from the archive) via WM rollout. No learned "
                        "policy, no reach reward (reward stays MSE-only); SAC off. Implies --goal-explore.")
    # CEM solver defaults match LeWM's cube config (stable_worldmodel solver/cem.yaml + cube.yaml:
    # 300 samples / 30 iters / 30 elites / var_scale=1.0; horizon=receding_horizon=5 -> open-loop).
    p.add_argument("--cem-samples", type=int, default=300, help="CEM action samples per iter (per env); LeWM=300")
    p.add_argument("--cem-iters", type=int, default=30, help="CEM refit iterations (LeWM cem.yaml n_steps=30)")
    p.add_argument("--cem-elites", type=int, default=30, help="CEM elite count (top-k lowest cost); LeWM topk=30")
    p.add_argument("--cem-init-std", type=float, default=1.0, help="CEM initial action std; LeWM var_scale=1.0")
    p.add_argument("--cem-min-std", type=float, default=0.0,
                   help="elite-refit std FLOOR: clamp the per-step action std to at least this after each refit "
                        "(0 = LeWM-faithful, no floor). >0 curbs the std-collapse that makes 30-iter CEM zoom "
                        "into the model's single most-optimistic (least-reliable) plan -- i.e. less model exploitation.")
    p.add_argument("--cem-horizon", type=int, default=5,
                   help="CEM planning horizon H (block-decisions): minimize terminal ||zhat_H - z*||^2 via "
                        "autoregressive WM rollout. Executed OPEN-LOOP, LeWM-style (receding_horizon == H): "
                        "plan H, execute all H, re-plan when the buffer empties / env resets / goals refresh. "
                        "Set H=1 for replan-every-step. Longer H = more lookahead but more rollout bias "
                        "(this WM is 1-step trained).")
    p.add_argument("--cem-replan-every", type=int, default=0,
                   help="replan STRIDE (decisions): re-plan and execute only this many of the H-step plan before "
                        "re-planning, decoupling execution from the H-step lookahead. 1 = true receding-horizon MPC "
                        "(plan H, execute 1, replan -- fresh goal feedback every step). 0 (default) = LeWM open-loop "
                        "(stride == --cem-horizon: execute all H before replanning). Clamped to [1, cem_horizon].")
    p.add_argument("--cem-gamma", type=float, default=0.0,
                   help="running/shaped-cost discount: cost = sum_h gamma^(H-1-h) ||z_h - z*||^2 over the rollout "
                        "(terminal weight 1). 0 (default) = terminal-only, the exact LeWM objective. >0 rewards "
                        "INTERMEDIATE progress toward the goal so CEM has a gradient even when the H-step endpoint is "
                        "out of single-plan reach (raises cost_cv); 0.7-0.9 is a reasonable shaping range.")
    p.add_argument("--cem-mppi-temp", type=float, default=0.0,
                   help="MPPI-style SOFT update temperature: if >0, refit the plan as the exp(-cost/temp)-weighted "
                        "mean over ALL candidates (information-theoretic / path-integral update) instead of CEM's "
                        "hard top-k elites. Lower temp -> greedier (more model exploitation); higher -> softer. "
                        "0 (default) = LeWM hard-elite CEM. Tests whether soft selection exploits WM error less.")
    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument("--tau", type=float, default=0.005, help="Polyak rate (SAC-style target critic)")
    p.add_argument("--actor-lr", type=float, default=3e-4)
    p.add_argument("--critic-lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--updates-per-step", type=int, default=1)
    p.add_argument("--per-alpha", type=float, default=0.6)
    p.add_argument("--per-beta-start", type=float, default=0.4)
    p.add_argument("--per-priority", choices=["curiosity", "td"], default="curiosity",
                   help="SAC replay priority. 'curiosity' (default) = 1-step pred error r_cur; "
                        "'td' (ablation) = |TD error|, sign-agnostic so it also replays unsafe "
                        "transitions the critic mispredicts. See results.md.")
    p.add_argument("--buffer-frac", type=float, default=0.1, help="cap = clip(frac*total, 1e3, 5e4)")
    # logging backends (keys from .env)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--no-hf", action="store_true", help="disable HF checkpoint upload")
    p.add_argument("--frozen-policy", action="store_true",
                   help="data-collection/eval: act with the loaded policy, NO gradient updates "
                        "(much higher cadence, learner removed). Refuses on hardware without a loaded policy.")
    p.add_argument("--save-buffer", action="store_true",
                   help="dump the replay buffer to out_dir/buffer_<N>.npz on exit (implied by "
                        "--frozen-policy); enables graceful Ctrl-C save.")
    p.add_argument("--hf-repo", default=None, help="HF repo id (else $HF_UPLOAD_REPO_ID)")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
