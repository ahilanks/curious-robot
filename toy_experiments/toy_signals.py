"""1-D toy of the curiosity-signal comparison: mse vs rnd vs lp (learning progress).

Target f : fixed random MLP on [-1, 1] (bigger, wiggly).
Learner g: smaller MLP that can never fit f exactly; ONE Adam step on each newly visited point.
Explorer : SAC policy pi(delta | x), |delta| <= amax, x' = clip(x + delta, -1, 1); the reward is
           the chosen signal evaluated at x' (computed at collection time, stored in the replay
           buffer, as in the main trainer).
Signals  : mse = g's squared error at x' before g's step on it
           rnd = predictor-vs-fixed-random-target error at x' (predictor: one Adam step per visit)
           lp  = |phi - EMA phi| of the visited-point pool re-scored under the current g
                 (src/train.py --goal-score lp convention, absolute value, EMA alpha 0.3), averaged
                 over pool points within a Gaussian kernel of x'. A pool point's EMA starts at its
                 first phi, so a brand-new region scores 0 (never re-scored = 0 in the main trainer).
Baselines: uniform (x' ~ U[-1, 1], no locality) and walk (delta ~ U[-amax, amax]).
Metric   : g's MSE against f on a dense grid vs samples taken, and where the samples went.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

METHODS = ("mse", "rnd", "lp", "uniform", "walk")


# ----------------------------------------------------------------------------- nets
def mlp(sizes, act=nn.Tanh):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)


def random_target(hidden: int, depth: int, gain_in: float, gain_h: float, gen: torch.Generator,
                  act: str = "sin") -> nn.Module:
    """A random MLP 1 -> hidden x depth -> 1. First layer scaled by gain_in (tanh: kink density;
    sin: SIREN-style omega_0, i.e. frequency content), hidden layers by gain_h. Output standardised
    on a grid to mean 0 / std 1. tanh random nets average out to a few smooth bumps that an 8x2
    learner fits to <2% of the variance; sin nets keep fine structure the learner cannot fit."""
    net = mlp([1] + [hidden] * depth + [1], act=Sin if act == "sin" else nn.Tanh)
    with torch.no_grad():
        first = True
        for m in net:
            if isinstance(m, nn.Linear):
                fan_in = m.weight.shape[1]
                if first:
                    m.weight.normal_(0.0, gain_in, generator=gen)
                    m.bias.uniform_(-gain_in, gain_in, generator=gen)
                    first = False
                elif act == "sin":                       # SIREN hidden init, scaled by gain_h
                    b = math.sqrt(6.0 / fan_in) * gain_h
                    m.weight.uniform_(-b, b, generator=gen)
                    m.bias.uniform_(-math.pi, math.pi, generator=gen)
                else:
                    m.weight.normal_(0.0, gain_h / math.sqrt(fan_in), generator=gen)
                    m.bias.normal_(0.0, 0.1, generator=gen)
        grid = torch.linspace(-1, 1, 2048).unsqueeze(1)
        y = net(grid)
        mu, sd = y.mean(), y.std()
        last = [m for m in net if isinstance(m, nn.Linear)][-1]
        last.weight.div_(sd)
        last.bias.sub_(mu).div_(sd)
    for p in net.parameters():
        p.requires_grad_(False)
    return net.eval()


def init_learner(hidden: int, depth: int, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    return mlp([1] + [hidden] * depth + [1])


# ----------------------------------------------------------------------------- SAC (1-D state, 1-D action)
class SAC:
    def __init__(self, amax, hidden=64, gamma=0.9, tau=0.005, lr=3e-4, target_entropy=-1.0):
        self.amax, self.gamma, self.tau = amax, gamma, tau
        self.actor = mlp([1, hidden, hidden, 2], act=nn.ReLU)
        self.q1 = mlp([2, hidden, hidden, 1], act=nn.ReLU)
        self.q2 = mlp([2, hidden, hidden, 1], act=nn.ReLU)
        self.q1_t, self.q2_t = copy.deepcopy(self.q1), copy.deepcopy(self.q2)
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.target_entropy = target_entropy
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_q = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)
        self.opt_al = torch.optim.Adam([self.log_alpha], lr=lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def _dist(self, x):
        out = self.actor(x)
        mu, log_std = out[:, :1], out[:, 1:].clamp(-5.0, 2.0)
        return mu, log_std.exp()

    def sample(self, x, deterministic=False):
        """Returns the action in [-amax, amax] and its log-prob (tanh-squashed Gaussian)."""
        mu, std = self._dist(x)
        u = mu if deterministic else mu + std * torch.randn_like(mu)
        a = torch.tanh(u)
        logp = (-0.5 * ((u - mu) / std) ** 2 - std.log() - 0.5 * math.log(2 * math.pi)).sum(1, keepdim=True)
        logp = logp - torch.log(1 - a.pow(2) + 1e-6).sum(1, keepdim=True)
        return a * self.amax, logp

    def q(self, net, x, a):
        return net(torch.cat([x, a / self.amax], 1))

    def update(self, x, a, r, x2):
        with torch.no_grad():
            a2, logp2 = self.sample(x2)
            qt = torch.min(self.q(self.q1_t, x2, a2), self.q(self.q2_t, x2, a2)) - self.alpha * logp2
            y = r + self.gamma * qt                                   # continuing task: no terminals
        lq = F.mse_loss(self.q(self.q1, x, a), y) + F.mse_loss(self.q(self.q2, x, a), y)
        self.opt_q.zero_grad(); lq.backward(); self.opt_q.step()

        a_new, logp = self.sample(x)
        q_new = torch.min(self.q(self.q1, x, a_new), self.q(self.q2, x, a_new))
        la = (self.alpha.detach() * logp - q_new).mean()
        self.opt_a.zero_grad(); la.backward(); self.opt_a.step()

        lal = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.opt_al.zero_grad(); lal.backward(); self.opt_al.step()

        with torch.no_grad():
            for t, s in ((self.q1_t, self.q1), (self.q2_t, self.q2)):
                for pt, ps in zip(t.parameters(), s.parameters()):
                    pt.mul_(1 - self.tau).add_(self.tau * ps)
        return float(lq), float(la), float(self.alpha)


class RunningStd:
    """Welford running variance; rewards are divided by the running std (RND convention)."""
    def __init__(self):
        self.n, self.mean, self.m2 = 0, 0.0, 0.0

    def update(self, v: float):
        self.n += 1
        d = v - self.mean
        self.mean += d / self.n
        self.m2 += d * (v - self.mean)

    @property
    def std(self):
        return math.sqrt(self.m2 / (self.n - 1)) if self.n > 1 else 1.0


# ----------------------------------------------------------------------------- oracle floor
def oracle_floor(f, hidden, depth, seed, grid, steps=4000, lr=1e-2, restarts=3):
    """Best grid MSE a learner of this size reaches with FULL access to f (full-batch Adam).
    A reference for the capacity-limited error, not a tight bound."""
    with torch.no_grad():
        y = f(grid)
    best = float("inf")
    for k in range(restarts):
        g = init_learner(hidden, depth, seed * 1000 + k)
        opt = torch.optim.Adam(g.parameters(), lr=lr)
        for _ in range(steps):
            loss = F.mse_loss(g(grid), y)
            opt.zero_grad(); loss.backward(); opt.step()
            best = min(best, float(loss.detach()))
    return best


# ----------------------------------------------------------------------------- one run
def run(args) -> dict:
    torch.set_num_threads(1)
    seed = args.seed
    gen = torch.Generator().manual_seed(seed)                # target f depends only on the seed
    f = random_target(args.f_hidden, args.f_depth, args.f_gain_in, args.f_gain_h, gen, args.f_act)
    grid = torch.linspace(-1, 1, args.grid_n).unsqueeze(1)
    with torch.no_grad():
        f_grid = f(grid).squeeze(1)
    oracle = oracle_floor(f, args.g_hidden, args.g_depth, seed, grid) if args.oracle else float("nan")

    mseed = seed * 7919 + METHODS.index(method := args.method)   # str hash is per-process random
    torch.manual_seed(mseed)
    rng = np.random.default_rng(mseed)
    g = init_learner(args.g_hidden, args.g_depth, seed)      # same init for every method at a seed
    opt_g = (torch.optim.Adam(g.parameters(), lr=args.g_lr) if args.g_opt == "adam"
             else torch.optim.SGD(g.parameters(), lr=args.g_lr))

    amax = args.amax
    learned = method in ("mse", "rnd", "lp")
    sac = SAC(amax, args.sac_hidden, args.gamma, args.tau, args.sac_lr) if learned else None

    # RND nets: fixed random target with a moderate input gain so novelty is local, predictor default init.
    if method == "rnd":
        rnd_t = mlp([1, args.rnd_hidden, args.rnd_hidden, args.rnd_out])
        with torch.no_grad():
            first = [m for m in rnd_t if isinstance(m, nn.Linear)][0]
            first.weight.normal_(0.0, args.rnd_gain_in, generator=gen)
            first.bias.uniform_(-args.rnd_gain_in, args.rnd_gain_in, generator=gen)
        for p in rnd_t.parameters():
            p.requires_grad_(False)
        rnd_p = mlp([1, args.rnd_hidden, args.rnd_hidden, args.rnd_out])
        opt_rnd = torch.optim.Adam(rnd_p.parameters(), lr=args.rnd_lr)

    # LP pool: every visited point, its f value, and the EMA of its re-scored error phi.
    N = args.steps
    pool_x = torch.zeros(N, 1)
    pool_f = torch.zeros(N, 1)
    pool_ema = torch.zeros(N, 1)
    n_pool = 0

    # SAC replay: (x, a, r, x')
    buf_x, buf_a, buf_r, buf_x2 = (torch.zeros(N, 1) for _ in range(4))
    rstd = RunningStd()

    xs = np.zeros(N, np.float32)          # visited points
    r_raw = np.zeros(N, np.float32)       # raw signal at the visited point
    r_norm = np.zeros(N, np.float32)
    eval_steps, eval_mse, g_snaps = [], [], []
    logs = {"lq": [], "la": [], "alpha": []}

    def evaluate(step):
        with torch.no_grad():
            gg = g(grid).squeeze(1)
        eval_steps.append(step)
        eval_mse.append(float(((gg - f_grid) ** 2).mean()))
        g_snaps.append(gg.numpy().astype(np.float32))

    x = float(rng.uniform(-1, 1))
    evaluate(0)
    t0 = time.time()
    for step in range(N):
        # ---- choose the next point
        if method == "uniform":
            x2 = float(rng.uniform(-1, 1)); a = x2 - x
        elif method == "walk":
            a = float(rng.uniform(-amax, amax)); x2 = float(np.clip(x + a, -1, 1))
        else:
            if step < args.warmup:
                a = float(rng.uniform(-amax, amax))
            else:
                with torch.no_grad():
                    a = float(sac.sample(torch.tensor([[x]]))[0])
            x2 = float(np.clip(x + a, -1, 1))
        xt = torch.tensor([[x2]])
        with torch.no_grad():
            ft = f(xt)

        # ---- the signal at x' (before g / the RND predictor take their step on it)
        if method == "mse" or method == "uniform" or method == "walk":
            with torch.no_grad():
                r = float((g(xt) - ft).pow(2))
        elif method == "rnd":
            with torch.no_grad():
                r = float((rnd_p(xt) - rnd_t(xt)).pow(2).mean())
        elif method == "lp":
            with torch.no_grad():
                if n_pool > 0 and step % args.lp_rescore_every == 0:
                    phi = (g(pool_x[:n_pool]) - pool_f[:n_pool]).pow(2)
                    lp = (phi - pool_ema[:n_pool]).abs()
                    pool_ema[:n_pool] += args.lp_alpha * (phi - pool_ema[:n_pool])
                    w = torch.exp(-0.5 * ((pool_x[:n_pool] - xt) / args.lp_bw) ** 2)
                    ws = float(w.sum())
                    r = float((w * lp).sum() / ws) if ws > 1e-6 else 0.0
                else:
                    r = 0.0
                phi_new = float((g(xt) - ft).pow(2))
            pool_x[n_pool] = x2; pool_f[n_pool] = ft; pool_ema[n_pool] = phi_new; n_pool += 1

        # ---- learner g: ONE step on the new point only (default), or on the new point plus
        #      --g-replay random past points (the control that removes forgetting)
        if args.g_replay and step > 0:
            ridx = torch.from_numpy(rng.integers(0, step, args.g_replay))
            bx = torch.cat([xt, torch.from_numpy(xs[ridx.numpy()]).unsqueeze(1)])
            with torch.no_grad():
                bf = f(bx)
            loss_g = (g(bx) - bf).pow(2).mean()
        else:
            loss_g = (g(xt) - ft).pow(2).mean()
        opt_g.zero_grad(); loss_g.backward(); opt_g.step()
        if method == "rnd":
            loss_r = (rnd_p(xt) - rnd_t(xt)).pow(2).mean()
            opt_rnd.zero_grad(); loss_r.backward(); opt_rnd.step()

        # ---- store, update the policy
        rstd.update(r)
        rn = r / (rstd.std + 1e-8) if args.reward_norm else r
        xs[step], r_raw[step], r_norm[step] = x2, r, rn
        if learned:
            buf_x[step] = x; buf_a[step] = a; buf_r[step] = rn; buf_x2[step] = x2
            if step >= args.warmup:
                for _ in range(args.updates_per_step):
                    idx = torch.from_numpy(rng.integers(0, step + 1, args.batch))
                    lq, la, al = sac.update(buf_x[idx], buf_a[idx], buf_r[idx], buf_x2[idx])
                if step % args.eval_every == 0:
                    logs["lq"].append(lq); logs["la"].append(la); logs["alpha"].append(al)
        x = x2
        if args.episode_len and (step + 1) % args.episode_len == 0:
            x = float(rng.uniform(-1, 1))
        if (step + 1) % args.eval_every == 0:
            evaluate(step + 1)

    # final policy snapshot: mean action over the grid (learned methods only)
    if learned:
        with torch.no_grad():
            pi_grid = sac.sample(grid, deterministic=True)[0].squeeze(1).numpy()
    else:
        pi_grid = np.full(args.grid_n, np.nan, np.float32)
    return dict(
        method=method, seed=seed, wall=time.time() - t0, oracle=oracle,
        grid=grid.squeeze(1).numpy(), f_grid=f_grid.numpy(), xs=xs, r_raw=r_raw, r_norm=r_norm,
        eval_steps=np.array(eval_steps), eval_mse=np.array(eval_mse), g_snaps=np.stack(g_snaps),
        pi_grid=pi_grid, args=json.dumps(vars(args)), **{f"log_{k}": np.array(v) for k, v in logs.items()},
    )


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", choices=METHODS, default="mse")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs"))
    # target f
    p.add_argument("--f-act", choices=("sin", "tanh"), default="sin")
    p.add_argument("--f-hidden", type=int, default=64)
    p.add_argument("--f-depth", type=int, default=2)
    p.add_argument("--f-gain-in", type=float, default=8.0, help="first-layer scale: sin omega_0 (frequency), tanh kink density")
    p.add_argument("--f-gain-h", type=float, default=1.0)
    # learner g
    p.add_argument("--g-hidden", type=int, default=8)
    p.add_argument("--g-depth", type=int, default=2)
    p.add_argument("--g-lr", type=float, default=1e-2)
    p.add_argument("--g-opt", choices=("adam", "sgd"), default="adam")
    p.add_argument("--g-replay", type=int, default=0,
                   help="also include this many random PAST points in g's single step (0 = new point only)")
    p.add_argument("--oracle", action=argparse.BooleanOptionalAction, default=True,
                   help="fit a same-size learner on the full grid for the capacity floor")
    # explorer
    p.add_argument("--amax", type=float, default=0.1, help="|delta| bound")
    p.add_argument("--episode-len", type=int, default=0, help="reset x ~ U[-1,1] every n steps (0 = never)")
    p.add_argument("--warmup", type=int, default=200, help="random-delta steps before SAC acts/updates")
    p.add_argument("--sac-hidden", type=int, default=64)
    p.add_argument("--sac-lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--updates-per-step", type=int, default=1)
    p.add_argument("--reward-norm", action=argparse.BooleanOptionalAction, default=True,
                   help="divide the reward by its running std (puts the three signals on one scale)")
    # rnd
    p.add_argument("--rnd-hidden", type=int, default=32)
    p.add_argument("--rnd-out", type=int, default=8)
    p.add_argument("--rnd-gain-in", type=float, default=4.0)
    p.add_argument("--rnd-lr", type=float, default=1e-3)
    # lp
    p.add_argument("--lp-alpha", type=float, default=0.3, help="EMA rate per re-scoring (main trainer: 0.3)")
    p.add_argument("--lp-rescore-every", type=int, default=1)
    p.add_argument("--lp-bw", type=float, default=0.05, help="Gaussian kernel width over pool points around x'")
    # eval
    p.add_argument("--grid-n", type=int, default=512)
    p.add_argument("--eval-every", type=int, default=50)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    res = run(args)
    path = os.path.join(args.out, f"{args.method}_s{args.seed}.npz")
    np.savez_compressed(path, **res)
    print(f"{args.method:8s} seed {args.seed}  final mse {res['eval_mse'][-1]:.4f}  "
          f"min {res['eval_mse'].min():.4f}  oracle {res['oracle']:.4f}  ({res['wall']:.1f}s)  -> {path}")


if __name__ == "__main__":
    main()
