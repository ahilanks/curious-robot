"""PPO on an intrinsic reward in the point-push toy or Push-T: what does each curiosity signal make the agent do?

    python toy/run_curiosity.py --signal {none,pred,lp,ln,count} [--tv] [--seed 0] [--iters 300]
    python toy/run_curiosity.py --env pusht --start {fixed,random} --signal pred --save-replay 32 --wm-replay 5000000

One PPO iteration = one synchronous episode in every env from the start distribution (point-push:
the fixed start, agent bottom-left, block in the centre; Push-T: --start). No task reward and no entropy bonus: the intrinsic reward is the only
drive. It is divided by a running std of its discounted return (RND-style), so the signals are
compared on where they pay, not on their scale.

Ground-truth read-outs per iteration (the agent never sees them): agent / block coverage of a
10x10 grid within an episode, block displacement, contact rate, time on the noisy TV, and where
the reward pays out (contact / wall / TV / free-space transitions).
Writes toy/runs/<name>/{config.json, metrics.jsonl, final.pt} (final.pt carries the agent's world
model for pred / lp); --save-replay K adds replay.pt = every iteration's states and applied actions
of the first K envs (the data a fresh model is trained on in toy/pusht_eval.py).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch import nn

from point_push import PointPush
from signals import make_signal


def make_env(args, n_envs, seed, device):
    if args.env == "pusht":
        from push_t import PushT
        return PushT(n_envs, device=device, ep_len=args.ep_len, start=args.start,
                     action_scale=args.action_scale, workers=args.workers, seed=seed, tv=args.tv, physics=args.physics)
    return PointPush(n_envs, device=device, ep_len=args.ep_len, seed=seed, tv=args.tv)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128, log_std=0.0):
        super().__init__()
        self.pi = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden),
                                nn.Tanh(), nn.Linear(hidden, act_dim))
        self.v = nn.Sequential(nn.Linear(obs_dim + 1, hidden), nn.Tanh(), nn.Linear(hidden, hidden),
                               nn.Tanh(), nn.Linear(hidden, 1))   # +1 = episode-time feature
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std)))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, math.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.pi[-1].weight, 0.01)
        nn.init.orthogonal_(self.v[-1].weight, 1.0)

    def dist(self, obs):
        return torch.distributions.Normal(self.pi(obs), self.log_std.exp())

    def value(self, obs, tfrac):
        return self.v(torch.cat([obs, tfrac], -1)).squeeze(-1)


class RunningVar:
    """Running variance of the discounted intrinsic return (RND's reward normaliser)."""

    def __init__(self):
        self.mean, self.var, self.n = 0.0, 1.0, 1e-4

    def update(self, x):
        bm, bv, bn = x.mean().item(), x.var(unbiased=False).item(), x.numel()
        d, tot = bm - self.mean, self.n + bn
        self.mean += d * bn / tot
        self.var = (self.var * self.n + bv * bn + d * d * self.n * bn / tot) / tot
        self.n = tot


@torch.no_grad()
def rollout(env, ac, T):
    obs = env.reset()
    ro = {k: [] for k in ("obs", "act", "u", "logp", "val", "state", "contact", "block_move", "in_tv", "wall")}
    ro["obs"].append(obs)
    ro["state"].append(env.state())
    for t in range(T):
        d = ac.dist(obs)
        u = d.sample()
        ro["u"].append(u)
        ro["logp"].append(d.log_prob(u).sum(-1))
        ro["val"].append(ac.value(obs, torch.full((env.E, 1), t / T, device=obs.device)))
        a = u.clamp(-1, 1)
        obs, info = env.step(a)
        ro["obs"].append(obs)
        ro["act"].append(info.get("act_applied", a))            # Push-T: after the arena clamp
        ro["state"].append(env.state())
        for k in ("contact", "block_move", "in_tv", "wall"):
            ro[k].append(info[k])
    return {k: torch.stack(v) for k, v in ro.items()}


def ppo_update(ac, opt, ro, rew, args):
    T, E = rew.shape
    v = ro["val"]
    adv, last = torch.zeros_like(rew), torch.zeros(E, device=rew.device)
    for t in reversed(range(T)):                                 # finite horizon: no bootstrap past T
        nv = v[t + 1] if t + 1 < T else torch.zeros_like(last)
        last = rew[t] + args.gamma * nv - v[t] + args.gamma * args.lam * last
        adv[t] = last
    ret = adv + v
    D = ro["obs"].shape[-1]
    b_obs, b_u = ro["obs"][:-1].reshape(-1, D), ro["u"].reshape(-1, ro["u"].shape[-1])
    b_tf = (torch.arange(T, device=rew.device).float() / T).repeat_interleave(E).unsqueeze(-1)
    b_logp, b_adv, b_ret = ro["logp"].reshape(-1), adv.reshape(-1), ret.reshape(-1)
    n = b_obs.shape[0]
    mb, stats = n // args.minibatches, []
    for _ in range(args.ppo_epochs):
        perm = torch.randperm(n, device=rew.device)
        for i in range(0, n, mb):
            idx = perm[i:i + mb]
            d = ac.dist(b_obs[idx])
            ratio = (d.log_prob(b_u[idx]).sum(-1) - b_logp[idx]).exp()
            a_ = b_adv[idx]
            a_ = (a_ - a_.mean()) / (a_.std() + 1e-8)
            pg = -torch.min(ratio * a_, ratio.clamp(1 - args.clip, 1 + args.clip) * a_).mean()
            vf = 0.5 * (ac.value(b_obs[idx], b_tf[idx]) - b_ret[idx]).pow(2).mean()
            ent = d.entropy().sum(-1).mean()
            loss = pg + args.vf_coef * vf - args.ent_coef * ent
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(ac.parameters(), args.max_grad_norm)
            opt.step()
            stats.append(torch.stack([pg.detach(), vf.detach(), ent.detach(),
                                      ((ratio - 1).abs() > args.clip).float().mean()]))
    s = torch.stack(stats).mean(0).tolist()
    return {"pg_loss": s[0], "vf_loss": s[1], "entropy": s[2], "clipfrac": s[3]}


# ------------------------------------------------------------------ ground-truth read-outs
def grid_cells(pos, lo, hi, bins=10):
    idx = ((pos - lo) / (hi - lo) * bins).long().clamp(0, bins - 1)
    return idx[..., 0] * bins + idx[..., 1]


def episode_coverage(cells, n_cells):
    """Mean over envs of the fraction of grid cells visited within the episode. cells: (T+1, E)."""
    vis = torch.zeros(cells.shape[1], n_cells, dtype=torch.bool, device=cells.device)
    vis.scatter_(1, cells.T.contiguous(), True)
    return vis.float().mean(1).mean().item()


def hist2d(pos, bins=40):
    """Visit histogram over the unit square; h[ix, iy]."""
    idx = (pos.reshape(-1, 2) * bins).long().clamp(0, bins - 1)
    h = torch.zeros(bins * bins, device=pos.device)
    h.index_add_(0, idx[:, 0] * bins + idx[:, 1], torch.ones(idx.shape[0], device=pos.device))
    return h.view(bins, bins)


def masked_mean(x, m):
    return x[m].mean().item() if m.any() else float("nan")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--signal", required=True, choices=("none", "pred", "lp", "lps", "ln", "count"))
    p.add_argument("--env", default="point", choices=("point", "pusht"))
    p.add_argument("--start", default="fixed", choices=("fixed", "random"),
                   help="Push-T reset: fixed (agent (100,100), T centred at angle 0) or gym-pusht's random")
    p.add_argument("--action-scale", type=float, default=100.0, help="Push-T: px of PD-target offset per unit action")
    p.add_argument("--workers", type=int, default=12, help="Push-T: simulator processes")
    p.add_argument("--physics", default="safe", choices=("safe", "gym"),
                   help="Push-T: safe (default; a wall-pinned T cannot be squeezed, see push_t.py) or exact gym-pusht")
    p.add_argument("--save-replay", type=int, default=0, metavar="K",
                   help="save every iteration's states + applied actions of the first K envs to replay.pt")
    p.add_argument("--tv", action="store_true", help="add the noisy TV (unlearnable noise channels)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--envs", type=int, default=256)
    p.add_argument("--ep-len", type=int, default=200)
    p.add_argument("--name", default=None, help="run dir name (default <signal>[_tv]_s<seed>)")
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "runs"))
    # PPO
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--minibatches", type=int, default=8)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--log-std-init", type=float, default=0.0)
    # world model (pred / lp)
    p.add_argument("--wm-lr", type=float, default=1e-3)
    p.add_argument("--wm-batch", type=int, default=1024)
    p.add_argument("--wm-epochs", type=int, default=1)
    p.add_argument("--lp-ema", type=float, default=0.01, help="EMA rate of the lagged WM copy (per WM step)")
    p.add_argument("--wm-replay", type=int, default=0,
                   help="WM trains on each rollout mixed 1:1 with a FIFO replay of this many transitions (0 = rollout only)")
    # learnable novelty (their RL settings)
    p.add_argument("--ln-hidden", type=int, default=32)
    p.add_argument("--ln-ridge", type=float, default=0.3)
    p.add_argument("--ln-eta", type=float, default=1.0)
    # logging
    p.add_argument("--snap-every", type=int, default=25)
    p.add_argument("--log-every", type=int, default=10)
    args = p.parse_args()
    if args.env == "point":
        for k, default in (("start", "fixed"), ("action_scale", 100.0), ("workers", 12), ("physics", "safe")):
            if getattr(args, k) != default:
                p.error(f"--{k.replace('_', '-')} is Push-T only (point-push always starts fixed)")

    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prefix = f"pt{args.start[0]}_" if args.env == "pusht" else ""
    name = args.name or f"{prefix}{args.signal}{'_tv' if args.tv else ''}_s{args.seed}"
    out = Path(args.out) / name
    out.mkdir(parents=True, exist_ok=True)

    env = make_env(args, args.envs, args.seed, dev)
    sig = make_signal(args.signal, env, dev, args)
    extra = {}
    if args.signal == "ln":                                      # frozen stats from a base-seed env
        cal_env = make_env(args, 64, 1000, dev)
        extra = sig.calibrate(cal_env)
        cal_env.close()
        print(f"[ln] calibrated: {extra}", flush=True)
    json.dump({**vars(args), **extra, "obs_dim": env.obs_dim, "device": str(dev)},
              open(out / "config.json", "w"), indent=2)

    ac = ActorCritic(env.obs_dim, env.act_dim, log_std=args.log_std_init).to(dev)
    opt = torch.optim.Adam(ac.parameters(), lr=args.lr, eps=1e-5)
    rv = RunningVar()
    # cells_cum: point-push (agent cell, block cell) on 10x10 grids; Push-T the count oracle's
    # 80k (agent, T position, T angle) cells -- a 10x10x10x10 grid saturates under a random walk
    visited = torch.zeros(env.n_count_cells if args.env == "pusht" else 100 * 100, dtype=torch.bool, device=dev)
    hist_agent = torch.zeros(40, 40, device=dev)
    hist_block = torch.zeros(40, 40, device=dev)
    snaps, replay = [], {"state": [], "act": []}
    tail_from = int(0.9 * args.iters)
    T, E = args.ep_len, args.envs
    t0 = time.time()
    with open(out / "metrics.jsonl", "w") as mf:
        for it in range(args.iters):
            ro = rollout(env, ac, T)
            r_int, aux = sig.reward(ro)
            upd = sig.update(ro)
            m = {}
            if args.signal != "none":
                disc, R = torch.zeros(E, device=dev), []
                for t in range(T):
                    disc = args.gamma * disc + r_int[t]
                    R.append(disc)
                rv.update(torch.stack(R))
                m.update(ppo_update(ac, opt, ro, r_int / math.sqrt(rv.var + 1e-8), args))

            st = ro["state"]
            if args.save_replay:
                replay["state"].append(st[:, :args.save_replay].cpu())
                replay["act"].append(ro["act"][:, :args.save_replay].cpu())
            ag, bl = env.agent_xy(st), env.block_xy(st)
            ca = grid_cells(ag, env.agent_lo, env.agent_hi)
            cb = grid_cells(bl, env.block_lo, env.block_hi)
            visited[(env.count_cell(st) if args.env == "pusht" else ca * 100 + cb).reshape(-1)] = True
            contact, in_tv = ro["contact"], ro["in_tv"]
            at_wall = ro["wall"]                                   # point-push: agent pinned; Push-T: T on a wall
            free = ~contact & ~in_tv & ~at_wall
            wall_only = at_wall & ~contact & ~in_tv                # the TV disk touches the bottom wall
            disp = (bl[-1] - bl[0]).norm(dim=-1)
            m.update({
                "iter": it, "env_steps": (it + 1) * T * E, "time": time.time() - t0,
                "agent_cov": episode_coverage(ca, 100), "block_cov": episode_coverage(cb, 100),
                "block_disp": disp.mean().item(), "block_moved_frac": (disp > 0.05).float().mean().item(),
                "block_path": ro["block_move"].sum(0).mean().item(),
                "contact_rate": contact.float().mean().item(), "wall_rate": at_wall.float().mean().item(),
                "tv_frac": in_tv.float().mean().item(), "cells_cum": visited.float().mean().item(),
                "r_int": r_int.mean().item(), "r_contact": masked_mean(r_int, contact),
                "r_wall": masked_mean(r_int, wall_only), "r_tv": masked_mean(r_int, in_tv),
                "r_free": masked_mean(r_int, free), "policy_std": ac.log_std.exp().mean().item(), **upd,
            })
            if "err" in aux:
                e = aux["err"]
                m.update({"err_contact": masked_mean(e, contact), "err_wall": masked_mean(e, wall_only),
                          "err_tv": masked_mean(e, in_tv), "err_free": masked_mean(e, free)})
            if "S_episode" in aux:
                m["S_episode"] = aux["S_episode"].mean().item()
            mf.write(json.dumps(m) + "\n")
            mf.flush()

            if it >= tail_from:
                hist_agent += hist2d(ag)
                hist_block += hist2d(bl)
            if it % args.snap_every == 0 or it == args.iters - 1:
                snaps.append({"iter": it, "agent": hist2d(ag).cpu(), "block": hist2d(bl).cpu()})
            if it % args.log_every == 0 or it == args.iters - 1:
                print(f"[{name}] it {it:4d}  agent_cov {m['agent_cov']:.2f}  block_cov {m['block_cov']:.3f}  "
                      f"disp {m['block_disp']:.3f}  contact {m['contact_rate']:.3f}  wall {m['wall_rate']:.2f}  "
                      f"tv {m['tv_frac']:.3f}  cells {m['cells_cum']:.3f}  std {m['policy_std']:.2f}  "
                      f"r c/w/tv/f {m['r_contact']:.3g}/{m['r_wall']:.3g}/{m['r_tv']:.3g}/{m['r_free']:.3g}  "
                      f"{m['time']:.0f}s", flush=True)

    torch.save({"args": vars(args), "extra": extra, "ac": ac.state_dict(),
                "wm": sig.model.state_dict() if hasattr(sig, "model") else None,
                "hist_agent": hist_agent.cpu(), "hist_block": hist_block.cpu(), "snaps": snaps,
                "traj": ro["state"][:, :8].cpu(), "visited": visited.cpu()}, out / "final.pt")
    if args.save_replay:
        torch.save({"state": torch.stack(replay["state"]), "act": torch.stack(replay["act"]),
                    "env": args.env, "start": args.start}, out / "replay.pt")
    env.close()
    print(f"[{name}] done in {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
