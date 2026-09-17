"""RP1-style planning for the wr stack (three-arm experiment, 2026-09-17).

Arms (flags in src/train.py):
  --plan-cost latent   CEM scores candidates by ||z_hat - z*||^2            (campaign baseline)
  --plan-cost value    CEM scores candidates by V(z_hat, z*): a learned goal-conditioned
                       quasimetric critic = temporal cost-to-go in DECISIONS (RP1 App. B.1)
  --planner rp1        the RP1 refiner replaces the CEM search: K residual rounds
                       a <- clip(a + f_theta(a, v, dv/da)) through the frozen WM, trained pathwise
                       against the critic (RP1 App. B.2). 9 WM rollouts per decision.

Nothing external: critic and refiner are fitted on the replay buffer's OWN latents under the
frozen encoder (hindsight goals along the buffer's episodes + cross-episode goals), re-fitted
periodically as data arrives and fully after every encoder sleep (latents move). The critic /
refiner code is the reproduction in rp1/ (rp1.critic, rp1.planner), re-used as a library.

Conventions match cem_plan: the WM is rolled from the last Hb REAL latents + actions, the plan is
in the same units CEM plans in (executed after the loop's amplitude scaling), T = --cem-horizon.
"""
from __future__ import annotations

import contextlib
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(_REPO, "rp1") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO, "rp1"))
from rp1.critic import make_critic, TDSampler, CriticTrainer, c_gamma      # noqa: E402
from rp1.planner import Refiner, cosine_lr                                  # noqa: E402


# ----------------------------------------------------------------------------- cache
class BufferLatentCache:
    """rp1.cache.LatentCache-compatible view of the replay buffer: episodes are the contiguous
    time-ordered segments of each env ring split at is_start; time unit = one decision (= one
    5-step action block = RP1's 'block'). Holds z (rows, D), the executed actions a (rows, A)
    and the episode table (ep_len / ep_off) the rp1 samplers index with."""

    def __init__(self, buf, encode_rows, device, Hb, min_len=4):
        self.device, self.Hb, self.block = device, Hb, 1
        # 1. make every stored row's frozen-encoder latent available (z_cache is filled live while
        #    the encoder is frozen and invalidated on every thaw -> encode the misses once here)
        es, ss = [], []
        for e in range(buf.n_envs):
            n = int(buf.count[e])
            if n == 0:
                continue
            slots = np.arange(n) if n < buf.C else (np.arange(buf.C) + int(buf.head[e])) % buf.C
            es.append(np.full(n, e)); ss.append(slots)
        if not es:
            raise RuntimeError("empty replay buffer")
        es, ss = np.concatenate(es), np.concatenate(ss)
        miss = ~buf.z_valid[es, ss]
        self.n_encoded = int(miss.sum())
        if self.n_encoded:
            for j in range(0, self.n_encoded, 512):
                me, ms = es[miss][j:j + 512], ss[miss][j:j + 512]
                buf.z_put(me, ms, encode_rows(buf.pixels[me, ms], buf.proprio[me, ms]))
        # 2. episodes: per env ring in time order, split at is_start (and at the ring seam)
        zs, acts, ep_len = [], [], []
        for e in range(buf.n_envs):
            n = int(buf.count[e])
            if n == 0:
                continue
            slots = np.arange(n) if n < buf.C else (np.arange(buf.C) + int(buf.head[e])) % buf.C
            starts = np.flatnonzero(buf.is_start[e, slots])
            bounds = np.unique(np.concatenate([[0], starts, [n]]))
            for lo, hi in zip(bounds[:-1], bounds[1:]):
                if hi - lo < min_len:
                    continue
                sl = slots[lo:hi]
                zs.append(buf.z_cache[e, sl]); acts.append(buf.action[e, sl]); ep_len.append(hi - lo)
        self.ep_len = np.asarray(ep_len, np.int64)
        self.ep_off = np.concatenate([[0], np.cumsum(self.ep_len)[:-1]]).astype(np.int64)
        self.n_ep = len(self.ep_len)
        self.z = torch.as_tensor(np.concatenate(zs), device=device)          # (rows, D) raw latents
        self.a = torch.as_tensor(np.concatenate(acts), device=device)        # (rows, A) executed actions
        self.mean = self.z.mean(0, keepdim=True)
        self.std = self.z.std(0, keepdim=True) + 1e-6
        self.extras = {}

    @property
    def rows(self):
        return int(self.z.shape[0])

    def history(self, rows):
        """(Hb, B, D) latents z_{t-Hb+1..t} and (Hb, B, A) actions a_{t-Hb..t-1} for anchor rows,
        clipped at the episode start exactly as the live loop seeds a reset (z repeated, a = 0)."""
        rows = np.asarray(rows, np.int64)
        ep = np.searchsorted(self.ep_off, rows, side="right") - 1
        off = self.ep_off[ep]
        hz, ha = [], []
        for k in range(self.Hb - 1, -1, -1):                 # z rows t-k
            rr = np.maximum(rows - k, off)
            hz.append(self.z[torch.as_tensor(rr, device=self.device)])
        for k in range(self.Hb, 0, -1):                      # a rows t-k (zeros before the episode)
            rr = rows - k
            ok = rr >= off
            a = self.a[torch.as_tensor(np.where(ok, rr, off), device=self.device)]
            ha.append(a * torch.as_tensor(ok, device=self.device).float().unsqueeze(1))
        return torch.stack(hz, 0), torch.stack(ha, 0)

    def sample_actor(self, B, max_delta, cross_prob, rng):
        """Anchor rows + hindsight goals: the state delta in [1, max_delta] decisions ahead in the
        same episode, or (cross_prob) a random row of another episode (RP1 ActorSampler)."""
        e = rng.integers(0, self.n_ep, B)
        L = self.ep_len[e]
        t = (rng.random(B) * (L - 1)).astype(np.int64)                      # anchor with >= 1 future step
        room = np.maximum(L - 1 - t, 1)
        delta = (rng.random(B) * np.minimum(room, max_delta)).astype(np.int64) + 1
        rows0 = self.ep_off[e] + t
        rowsg = self.ep_off[e] + np.minimum(t + delta, L - 1)
        cross = rng.random(B) < cross_prob
        e2 = rng.integers(0, self.n_ep, B)
        t2 = (rng.random(B) * self.ep_len[e2]).astype(np.int64)
        rowsg = np.where(cross, self.ep_off[e2] + t2, rowsg)
        hz, ha = self.history(rows0)
        return hz, ha, self.z[torch.as_tensor(rowsg, device=self.device)]


# --------------------------------------------------------------------------- WM rollout
def wm_rollout(wm, hist_z, hist_a, plan, Hb, act_scale=1.0):
    """Autoregressive rollout of a (B, T, A) plan from the last Hb real latents/actions, exactly the
    cem_plan recipe (eager predict, fp32, differentiable w.r.t. plan). hist_z (Hb,B,D), hist_a (Hb,B,A).
    Returns the terminal latent (B, D)."""
    predict = getattr(wm, "predict_eager", wm.predict)
    z_seq = hist_z.transpose(0, 1)                       # (B, Hb, D)
    a_seq = hist_a.transpose(0, 1)                       # (B, Hb, A)
    for h in range(plan.shape[1]):
        a_seq = torch.cat([a_seq, plan[:, h:h + 1] * act_scale], dim=1)   # act_scale: executed units (--plan-act-scale)
        znext = predict(z_seq[:, -Hb:], wm.action_encoder(a_seq[:, -Hb:]))[:, -1:]
        z_seq = torch.cat([z_seq, znext], dim=1)
    return z_seq[:, -1]


@contextlib.contextmanager
def wm_frozen_eval(wm):
    """No dropout, no parameter grads (only the plan / refiner receive gradients)."""
    was_training = wm.training
    flags = [(p, p.requires_grad) for p in wm.parameters()]
    for p, _ in flags:
        p.requires_grad_(False)
    wm.eval()
    try:
        yield
    finally:
        for p, f in flags:
            p.requires_grad_(f)
        if was_training:
            wm.train()


# ------------------------------------------------------------------------ the models
class PlannerModels:
    """Value critic (+ EMA teacher used for planning) and the RP1 refiner, fitted on the buffer cache."""

    def __init__(self, args, D, A, T, Hb, device):
        self.args, self.D, self.A, self.T, self.Hb, self.device = args, D, A, T, Hb, device
        self.critic = make_critic(D, 256, 128, 2).to(device)
        self.trainer = None                                  # CriticTrainer (holds the EMA target) after the first fit
        self.refiner = Refiner(T * A, 512).to(device) if args.planner == "rp1" else None
        self.ropt = torch.optim.Adam(self.refiner.parameters(), lr=args.rp1_lr) if self.refiner is not None else None
        self.rng = np.random.default_rng(args.seed + 7)
        self.n_fits, self.last_fit_step, self.stats = 0, -1, {}
        self.act_scale = 1.0        # --plan-act-scale: the WM sees plan * amax_frac (what is executed); set by the loop

    # planning-time critic = the EMA teacher (RP1: the actor's teacher)
    @property
    def V(self):
        return self.trainer.target if self.trainer is not None else self.critic

    def value_cost(self):
        """cost_fn(z_hat (N,D), z_goal (N,D)) -> (N,) for cem_plan(--plan-cost value)."""
        V = self.V

        def cost(zh, zg):
            with torch.no_grad():
                return V(zh.float(), zg.float())
        return cost

    # ---- fitting --------------------------------------------------------------------------------
    def fit(self, buf, wm, encode_rows, step, full):
        a = self.args
        t0 = time.time()
        cache = BufferLatentCache(buf, encode_rows, self.device, self.Hb)
        n_c = a.vcritic_fit_steps if full else a.vcritic_refit_steps
        n_r = (a.rp1_fit_steps if full else a.rp1_refit_steps) if self.refiner is not None else 0
        sampler = TDSampler(cache, range(cache.n_ep), n_step=a.vcritic_nstep, gamma=a.vcritic_gamma,
                            cross_prob=a.vcritic_cross_prob, seed=int(self.rng.integers(1 << 30)))
        if self.trainer is None:
            self.trainer = CriticTrainer(self.critic, sampler, gamma=a.vcritic_gamma, tau=a.vcritic_expectile,
                                         lr=a.vcritic_lr, polyak=0.005, total_steps=max(n_c, 1), device=self.device)
        else:
            self.trainer.sampler = sampler                   # continue training (moments kept) on the fresh cache
            self.trainer.step_i, self.trainer.total = 0, max(n_c, 1)
        with wm_frozen_eval(wm):
            ci = {}
            for _ in range(n_c):
                ci = self.trainer.step(a.vcritic_batch)
            ri = self._fit_refiner(cache, wm, n_r) if n_r > 0 else {}
        self.stats = {"fit/cache_rows": cache.rows, "fit/cache_eps": cache.n_ep, "fit/encoded": cache.n_encoded,
                      "fit/critic_steps": n_c, "fit/refiner_steps": n_r, "fit/seconds": time.time() - t0,
                      **{f"vcritic/{k}": v for k, v in ci.items()}, **{f"rp1/{k}": v for k, v in ri.items()}}
        self.n_fits += 1; self.last_fit_step = step
        print(f"[planner-fit #{self.n_fits}] step={step} {'FULL' if full else 'refit'}: cache {cache.rows} rows / "
              f"{cache.n_ep} eps ({cache.n_encoded} encoded); critic {n_c} steps"
              + (f" loss {ci['loss']:.3f} v {ci['v_mean']:.2f} y {ci['y_mean']:.2f}" if ci else "")
              + (f"; refiner {n_r} steps J {ri['J']:.3f} v0 {ri['v0']:.2f} -> vK {ri['vK']:.2f}" if ri else "")
              + f"; {time.time() - t0:.0f}s", flush=True)
        return self.stats

    def _plan_rounds(self, wm, hist_z, hist_a, zg, train):
        """K refinement rounds (RP1Planner.plan). Returns plans [a_0..a_K], values [v_0..v_K]."""
        a = self.args
        B = zg.shape[0]
        V = self.V
        x = torch.zeros(B, self.T, self.A, device=self.device, requires_grad=True)
        plans, values = [x], []
        for k in range(a.rp1_K):
            with torch.enable_grad():
                v = V(wm_rollout(wm, hist_z, hist_a, x, self.Hb, self.act_scale), zg)
                (g,) = torch.autograd.grad(v.sum(), x, retain_graph=train)
            values.append(v if train else v.detach())
            delta = self.refiner(x.flatten(1), v.detach(), g.detach().flatten(1))
            xn = (x + delta.view(B, self.T, self.A)).clamp(-a.rp1_amax, a.rp1_amax)
            x = xn if train else xn.detach().requires_grad_(True)
            plans.append(x)
        with (torch.enable_grad() if train else torch.no_grad()):
            vK = V(wm_rollout(wm, hist_z, hist_a, x, self.Hb, self.act_scale), zg)
        values.append(vK)
        return plans, values

    def _fit_refiner(self, cache, wm, n_steps):
        a = self.args
        info = {}
        for i in range(n_steps):
            lr = cosine_lr(i, n_steps, a.rp1_lr, a.rp1_lr * 0.1)
            for g in self.ropt.param_groups:
                g["lr"] = lr
            hz, ha, zg = cache.sample_actor(a.rp1_batch, a.vcritic_max_delta, a.vcritic_cross_prob, self.rng)
            plans, values = self._plan_rounds(wm, hz, ha, zg, train=True)
            vals = torch.stack(values[1:], 0)                                # v_1..v_K
            loss = values[-1].mean() + a.rp1_lambda_mean * vals.mean()      # J = v_K + lam * mean_k v_k
            self.ropt.zero_grad(set_to_none=True)
            loss.backward()
            gn = nn.utils.clip_grad_norm_(self.refiner.parameters(), 10.0)
            self.ropt.step()
            if a.rp1_live_critic:                                            # critic co-training (EMA teacher)
                self.trainer.step(a.vcritic_batch)
            info = dict(J=loss.item(), v0=values[0].mean().item(), vK=values[-1].mean().item(), gnorm=float(gn),
                        sat=(plans[-1].abs() >= a.rp1_amax - 1e-6).float().mean().item())
        return info

    # ---- acting -----------------------------------------------------------------------------------
    def plan(self, wm, hist_z, hist_a, z_goal, diag=None):
        """(Hb,n,D), (Hb,n,A), (n,D) -> (n, T, A) refined plan (K rounds, 9 WM rollouts)."""
        with wm_frozen_eval(wm), torch.enable_grad():
            plans, values = self._plan_rounds(wm, hist_z.float(), hist_a.float(), z_goal.float(), train=False)
        if diag is not None:
            diag["v0"] = float(values[0].mean()); diag["vK"] = float(values[-1].mean())
            diag["sat"] = float((plans[-1].abs() >= self.args.rp1_amax - 1e-6).float().mean())
        return plans[-1].detach()

    # ---- checkpointing --------------------------------------------------------------------------------
    def state_dict(self):
        sd = {"critic": self.critic.state_dict(), "n_fits": self.n_fits}
        if self.trainer is not None:
            sd["critic_teacher"] = self.trainer.target.state_dict()
        if self.refiner is not None:
            sd["refiner"] = self.refiner.state_dict()
        return sd
