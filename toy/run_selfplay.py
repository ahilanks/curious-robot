"""Self-play with a learning-progress reward on Push-T: the policy is the generator, a latent dynamics model the learner.

A port of Cowsik et al. 2026, "Self-Play Pretraining with Zero Data" (arXiv 2609.30063, Sec. 2.2), core objective only
(no expert iteration, mutation or replay bank) and their schedule: one learner step and one generator step per round.

    python toy/run_selfplay.py --chunk 0  --seed 0      # program = one whole episode  -> toy/runs/pusht/ptf_spep_s0
    python toy/run_selfplay.py --chunk 20 --seed 0      # program = a 20-step segment  -> toy/runs/pusht/ptf_spch20_s0
    python toy/run_selfplay.py --frozen-policy --seed 0 # the learner on the untrained policy -> toy/runs/pusht/ptf_spuni_s0

Round e (E closed-loop episodes of T steps from the fixed start; pi(a|s) is a tanh-squashed Gaussian):
  learner    z = enc(o), z' ~ z + pred(z, a): a from-scratch JEPA on the 8-D observation, trained like LeWM
             (lewm/train.py: MSE to the un-detached target embedding + lambda SIGReg on the embeddings, AdamW, clip 1).
             L(y; theta) = the mean latent prediction MSE over one program's transitions.
  reward     r_i = |<grad_theta L(y_i; theta_e), P_e * (theta_{floor(e/2)} - theta_e)>|   (Eq. 2)
             P_e = lr / (sqrt(v_hat) + eps) from the learner's AdamW state, scored at theta_e before this round's
             learner step. It is L's directional derivative along one vector, so one forward-mode JVP scores the pool.
  generator  A_i = (r_i - mean r) / (std r + eps) - beta (log pi(x_i) - log g0(x_i)), mean / std over the round's pool,
             g0 = uniform on [-1, 1]^2, log pi(x) = the sum of the program's per-step log-probs; one Adam step on
             -mean_i log pi(x_i) A_i (Eq. 4; every sample is fresh and on-policy, so rho_i = 1). The policy starts
             near g0: log_std -0.15 minimises KL(tanh N(0, sigma^2) || U[-1, 1]) (0.0097 nats / dim).
  learner    one AdamW step on the whole pool: mean L + lambda SIGReg(z)   (Eq. 1).

The default budget is the 09-25 Push-T sweep's (toy/run_pusht_sweep.sh): 64 envs x 1200 rounds x 200 steps = 15.36M
env steps, and --save-replay 8 keeps 8 x 1200 episodes = 1.92M transitions, the E1 data size of every sweep arm.
Outputs follow run_curiosity.py (config.json, metrics.jsonl, final.pt, replay.pt), so pusht_eval.py score and
pusht_report.py read them; final.pt["wm"] is None (the learner is latent, so there is no E2).

The paper's pool extras (Sec. 2.2, App. G), off by default; with --chunk 0 only:
  --mut-envs M      MAP-Elites mutation: the last M envs of a round replay an archived program's action sequence with one
                    window of 5-40 steps substituted / inserted / deleted (fresh near-uniform actions), open loop from the
                    fixed start. Archive: 6 x 6 niches (T displacement x T rotation over the episode), top 8 per niche
                    by stored reward, stored rewards x0.97 per round, only positive rewards admitted, parents drawn
                    uniformly across occupied niches. Mutated rows are excluded from the policy gradient (as in Eq. 4).
  --replay-progs R  replay bank: R programs from earlier rounds join each round's pool. With the fixed start and
                    deterministic physics re-executing an action sequence reproduces it, so the stored trajectory is
                    used. They are re-scored, normalise the advantage and train the learner, but are excluded from the
                    policy gradient: the paper's sequence ratio over 400 continuous action dims would sit at its
                    e^+-20 clip.
  --ei W            expert iteration: + W * -sum_i w_i log pi(x_i), w_i = r_i / sum_j r_j over the whole pool (lambda_EI).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.func import functional_call, jvp

from push_t import T_COG, PushT, angle_diff
from run_curiosity import episode_coverage, grid_cells, hist2d, masked_mean

LOG2 = math.log(2.0)


class SIGReg(nn.Module):
    """LeWM's Sketch Isotropic Gaussian Regulariser, verbatim from lewm/module.py (toy/ stays standalone)."""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """proj: (T, B, D)"""
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


def mlp(din, dout, hidden):
    # LayerNorm, not LeWM's BatchNorm: the reward needs per-sample losses that do not see the rest of the batch
    return nn.Sequential(nn.Linear(din, hidden), nn.LayerNorm(hidden), nn.GELU(),
                         nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, dout))


class LatentWM(nn.Module):
    def __init__(self, obs_dim, act_dim, z_dim=32, hidden=256):
        super().__init__()
        self.enc, self.pred = mlp(obs_dim, z_dim, hidden), mlp(z_dim + act_dim, z_dim, hidden)

    def forward(self, obs, act):
        """obs (T+1, E, D), act (T, E, A) -> per-transition prediction MSE (T, E), embeddings (T+1, E, Z)."""
        z = self.enc(obs)
        zh = z[:-1] + self.pred(torch.cat([z[:-1], act], -1))
        return (zh - z[1:]).pow(2).mean(-1), z


class Learner:
    def __init__(self, obs_dim, act_dim, device, z_dim, hidden, lr, wd, sigreg_w, clip):
        self.model = LatentWM(obs_dim, act_dim, z_dim, hidden).to(device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        self.sigreg, self.lam, self.clip = SIGReg().to(device), sigreg_w, clip
        self.snaps = {}                                           # round -> theta at its start, back to floor(e/2)

    def snapshot(self, e):
        self.snaps[e] = {k: p.detach().clone() for k, p in self.model.named_parameters()}
        for k in [k for k in self.snaps if k < e // 2]:
            del self.snaps[k]

    def direction(self, e):
        """P_e * (theta_{floor(e/2)} - theta_e); None before the learner's first step."""
        g = self.opt.param_groups[0]
        lr, b2, eps = g["lr"], g["betas"][1], g["eps"]
        past, v = self.snaps[e // 2], {}
        for k, p in self.model.named_parameters():
            st = self.opt.state.get(p)
            if not st:
                return None
            vhat = st["exp_avg_sq"] / (1 - b2 ** float(st["step"]))
            v[k] = lr / (vhat.sqrt() + eps) * (past[k] - p.detach())
        return v

    @torch.no_grad()
    def progress(self, e, obs, act):
        """Per-transition prediction error and its directional derivative along direction(e), both (T, E)."""
        v = self.direction(e)
        if v is None:
            err = self.model(obs, act)[0]
            return err, torch.zeros_like(err), 0.0
        params = {k: p.detach() for k, p in self.model.named_parameters()}
        err, dl = jvp(lambda p: functional_call(self.model, p, (obs, act))[0], (params,), (v,))
        return err, dl, math.sqrt(sum(x.pow(2).sum().item() for x in v.values()))

    def step(self, obs, act):
        err, z = self.model(obs, act)
        pred, sig = err.mean(), self.sigreg(z)
        self.opt.zero_grad(set_to_none=True)
        (pred + self.lam * sig).backward()
        if self.clip:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.opt.step()
        return {"wm_pred": pred.item(), "wm_sigreg": sig.item()}

    def step_flat(self, o, a, o2):
        """One AdamW step on a minibatch of transitions; SIGReg over groups of 64 samples (the pool's E), so lambda
        means the same as in step()."""
        err, z = self.model(torch.stack([o, o2]), a[None])
        pred, sig = err.mean(), self.sigreg(z.reshape(-1, 64, z.shape[-1]))
        self.opt.zero_grad(set_to_none=True)
        (pred + self.lam * sig).backward()
        if self.clip:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.opt.step()
        return {"wm_pred": pred.item(), "wm_sigreg": sig.item()}


class Replay:
    """FIFO buffer of past rounds' transitions (o, a, o2) on the device."""

    def __init__(self, cap, obs_dim, act_dim, device):
        self.x, self.cap, self.n, self.i = torch.zeros(cap, 2 * obs_dim + act_dim, device=device), cap, 0, 0
        self.D = obs_dim

    def add(self, o, a, o2):
        x = torch.cat([o, a, o2], -1)[-self.cap:]
        self.x[(self.i + torch.arange(len(x), device=x.device)) % self.cap] = x
        self.i, self.n = (self.i + len(x)) % self.cap, min(self.n + len(x), self.cap)

    def sample(self, k):
        x = self.x[torch.randint(0, self.n, (k,), device=self.x.device)]
        return x[:, :self.D], x[:, self.D:-self.D], x[:, -self.D:]


class Policy(nn.Module):
    """pi(a|s): a = tanh(u), u ~ N(mu(o), sigma) with a state-independent sigma (run_curiosity's actor, squashed)."""

    def __init__(self, obs_dim, act_dim, hidden=128, log_std=-0.15):
        super().__init__()
        self.mu = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                                nn.Linear(hidden, act_dim))
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std)))
        for m in self.mu:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, math.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.mu[-1].weight, 0.01)

    def dist(self, obs):
        return torch.distributions.Normal(self.mu(obs), self.log_std.exp())

    def log_prob(self, obs, u):
        """log pi(tanh(u) | o), summed over action dims."""
        return (self.dist(obs).log_prob(u) - 2 * (LOG2 - u - nn.functional.softplus(-2 * u))).sum(-1)


class ProgramBank:
    """Every executed program (observations, applied and pre-tanh actions), for the replay pool."""

    def __init__(self, cap, T, obs_dim, act_dim, device):
        self.obs = torch.zeros(cap, T + 1, obs_dim, device=device)
        self.act = torch.zeros(cap, T, act_dim, device=device)
        self.u = torch.zeros(cap, T, act_dim, device=device)
        self.cap, self.n, self.i = cap, 0, 0

    def add(self, obs, act, u):
        idx = (self.i + torch.arange(obs.shape[1], device=obs.device)) % self.cap
        self.obs[idx], self.act[idx], self.u[idx] = obs.transpose(0, 1), act.transpose(0, 1), u.transpose(0, 1)
        self.i, self.n = int((self.i + obs.shape[1]) % self.cap), min(self.n + obs.shape[1], self.cap)

    def sample(self, k):
        idx = torch.randperm(self.n, device=self.obs.device)[:k]                  # uniform, without replacement
        return self.obs[idx].transpose(0, 1), self.act[idx].transpose(0, 1), self.u[idx].transpose(0, 1)


class Archive:
    """MAP-Elites archive of action sequences, niched by the T's displacement and rotation over the episode."""
    DISP = torch.tensor([5.0, 20.0, 50.0, 100.0, 200.0])       # px bin edges -> 6 bins
    ROT = torch.tensor([5.0, 15.0, 45.0, 90.0, 135.0])         # degree bin edges -> 6 bins

    def __init__(self, top=8, decay=0.97):
        self.cells, self.top, self.decay = {}, top, decay

    def step(self):
        for elites in self.cells.values():
            for x in elites:
                x[0] *= self.decay

    def add(self, u, r, state):
        """u (T, B, 2) pre-tanh actions, r (B,) rewards, state (T+1, B, 7)."""
        def cog(s):
            c, sn = torch.cos(s[..., 6]), torch.sin(s[..., 6])
            return torch.stack([s[..., 4] + c * T_COG[0] - sn * T_COG[1], s[..., 5] + sn * T_COG[0] + c * T_COG[1]], -1)
        disp = (cog(state[-1]) - cog(state[0])).norm(dim=-1).cpu()
        rot = angle_diff(state[-1, :, 6], state[0, :, 6]).abs().rad2deg().cpu()
        nd, nr = torch.bucketize(disp, self.DISP), torch.bucketize(rot, self.ROT)
        u, r = u.transpose(0, 1).cpu(), r.cpu()
        for b in torch.nonzero(r > 0).squeeze(1).tolist():
            elites = self.cells.setdefault((int(nd[b]), int(nr[b])), [])
            elites.append([float(r[b]), u[b].clone()])
            elites.sort(key=lambda x: -x[0])
            del elites[self.top:]

    def parents(self, k):
        keys = list(self.cells)
        return [self.cells[keys[i]][torch.randint(len(self.cells[keys[i]]), ()).item()][1]
                for i in torch.randint(len(keys), (k,)).tolist()]


def mutate(u, sigma):
    """Substitute, insert or delete one window of 5-40 steps of a (T, 2) pre-tanh action sequence (fresh ~N(0, sigma))."""
    T = len(u)
    L = int(torch.randint(5, 41, ()))
    t0 = int(torch.randint(0, T - L + 1, ()))
    new = torch.randn(L, u.shape[1]) * sigma
    op = int(torch.randint(3, ()))
    if op == 0:
        u = u.clone()
        u[t0:t0 + L] = new
    elif op == 1:
        u = torch.cat([u[:t0], new, u[t0:]])[:T]
    else:
        u = torch.cat([u[:t0], u[t0 + L:], new])
    return u


@torch.no_grad()
def rollout(env, pi, T, fixed_u=None):
    """fixed_u (T, M, 2): the last M envs replay these pre-tanh actions open loop instead of sampling pi."""
    obs = env.reset()
    ro = {k: [] for k in ("obs", "u", "act", "state", "contact", "block_move", "wall")}
    ro["obs"].append(obs)
    ro["state"].append(env.state())
    for _ in range(T):
        u = pi.dist(obs).sample()
        if fixed_u is not None:
            u[-fixed_u.shape[1]:] = fixed_u[len(ro["u"])]
        obs, info = env.step(torch.tanh(u))
        ro["u"].append(u)
        ro["act"].append(info["act_applied"])                    # after the arena clamp: what the learner models
        ro["obs"].append(obs)
        ro["state"].append(env.state())
        for k in ("contact", "block_move", "wall"):
            ro[k].append(info[k])
    return {k: torch.stack(v) for k, v in ro.items()}


def per_program(x, chunk):
    """(T, E) per-transition values -> (P,) per-program sums; a program is a chunk-step segment (0 = the episode)."""
    T, E = x.shape
    c = chunk or T
    return x.reshape(T // c, c, E).sum(1).reshape(-1)


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = a.norm() * b.norm()
    return (a @ b / d).item() if d > 0 else float("nan")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chunk", type=int, default=0, help="steps per program (0 = the whole episode)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rounds", type=int, default=1200)
    p.add_argument("--envs", type=int, default=64, help="episodes per round")
    p.add_argument("--ep-len", type=int, default=200)
    p.add_argument("--start", default="fixed", choices=("fixed", "random"))
    p.add_argument("--physics", default="safe", choices=("safe", "gym"))
    p.add_argument("--action-scale", type=float, default=100.0)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--save-replay", type=int, default=8, metavar="K",
                   help="save every round's states + applied actions of the first K envs to replay.pt")
    p.add_argument("--name", default=None, help="run dir name (default pt<start>_sp<ep|ch<chunk>>_s<seed>)")
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "runs" / "pusht"))
    # generator
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=0.01, help="KL(pi || uniform) coefficient in the advantage")
    p.add_argument("--adv-eps", type=float, default=1e-8)
    p.add_argument("--log-std-init", type=float, default=-0.15, help="-0.15 = the tanh-Gaussian nearest uniform")
    p.add_argument("--mut-envs", type=int, default=0, help="MAP-Elites mutated programs per round (of --envs)")
    p.add_argument("--replay-progs", type=int, default=0, help="programs replayed from the bank per round")
    p.add_argument("--ei", type=float, default=0.0, help="expert-iteration weight lambda_EI (the paper uses 1)")
    p.add_argument("--frozen-policy", action="store_true",
                   help="never update the generator: the paper's 'uniform' ablation, i.e. the learner's matched baseline")
    # learner
    p.add_argument("--wm-lr", type=float, default=1e-3)
    p.add_argument("--wm-wd", type=float, default=1e-3)
    p.add_argument("--wm-clip", type=float, default=1.0)
    p.add_argument("--z-dim", type=int, default=32)
    p.add_argument("--wm-hidden", type=int, default=256)
    p.add_argument("--sigreg", type=float, default=0.09, help="SIGReg weight lambda (LeWM's default)")
    p.add_argument("--wm-steps", type=int, default=1,
                   help="learner steps per round; 1 (with --wm-replay 0) = the paper's one step on the whole pool, "
                        "> 1 = minibatch steps on the pool mixed 1:1 with a replay of past rounds")
    p.add_argument("--wm-batch", type=int, default=2048, help="transitions per minibatch step (--wm-steps > 1)")
    p.add_argument("--wm-pool-frac", type=float, default=0.5,
                   help="share of each minibatch from the current round, the rest from the replay; 0 = replay only "
                        "(the round is added first, so every step samples uniformly from everything seen so far)")
    p.add_argument("--wm-replay", type=int, default=5_000_000, help="replay capacity in transitions (--wm-steps > 1)")
    # logging
    p.add_argument("--snap-every", type=int, default=100)
    p.add_argument("--log-every", type=int, default=25)
    args = p.parse_args()
    T, E, c = args.ep_len, args.envs, args.chunk
    if c and T % c:
        p.error(f"--chunk {c} must divide --ep-len {T}")
    paper_pool = args.mut_envs or args.replay_progs or args.ei
    if paper_pool and c and c < T:
        p.error("--mut-envs / --replay-progs / --ei need whole-episode programs (--chunk 0)")

    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arm = ("uni" if args.frozen_policy else (f"ch{c}" if c and c < T else "ep")) + \
        (f"k{args.wm_steps}" if args.wm_steps > 1 else "") + \
        (f"p{round(100 * args.wm_pool_frac)}" if args.wm_steps > 1 and args.wm_pool_frac != 0.5 else "") + \
        ("pool" if paper_pool else "")
    name = args.name or f"pt{args.start[0]}_sp{arm}_s{args.seed}"
    out = Path(args.out) / name
    out.mkdir(parents=True, exist_ok=True)
    env = PushT(E, device=dev, ep_len=T, start=args.start, action_scale=args.action_scale, workers=args.workers,
                seed=args.seed, physics=args.physics)
    json.dump({**vars(args), "signal": "sp", "env": "pusht", "tv": False, "obs_dim": env.obs_dim,
               "device": str(dev)}, open(out / "config.json", "w"), indent=2)

    pi = Policy(env.obs_dim, env.act_dim, log_std=args.log_std_init).to(dev)
    opt = torch.optim.Adam(pi.parameters(), lr=args.lr)
    learner = Learner(env.obs_dim, env.act_dim, dev, args.z_dim, args.wm_hidden, args.wm_lr, args.wm_wd,
                      args.sigreg, args.wm_clip)
    visited = torch.zeros(env.n_count_cells, dtype=torch.bool, device=dev)
    hist_agent, hist_block = torch.zeros(40, 40, device=dev), torch.zeros(40, 40, device=dev)
    snaps, replay = [], {"state": [], "act": []}
    wm_buf = Replay(args.wm_replay, env.obs_dim, env.act_dim, dev) if args.wm_steps > 1 else None
    bank = ProgramBank(args.rounds * E, T, env.obs_dim, env.act_dim, dev) if args.replay_progs else None
    archive = Archive() if args.mut_envs else None
    sigma0 = math.exp(args.log_std_init)
    tail_from = int(0.9 * args.rounds)
    t0 = time.time()
    with open(out / "metrics.jsonl", "w") as mf:
        for e in range(args.rounds):
            n_mut = args.mut_envs if archive is not None and archive.cells else 0
            fixed = torch.stack([mutate(u, sigma0) for u in archive.parents(n_mut)], 1).to(dev) if n_mut else None
            ro = rollout(env, pi, T, fixed)
            kind = torch.zeros(E, dtype=torch.long, device=dev)               # 0 fresh, 1 mutated, 2 replayed
            kind[E - n_mut:] = 1
            obs, act, u_all = ro["obs"], ro["act"], ro["u"]                     # the round's pool
            if bank is not None and bank.n >= args.replay_progs:
                ob_r, ac_r, u_r = bank.sample(args.replay_progs)
                obs, act, u_all = torch.cat([obs, ob_r], 1), torch.cat([act, ac_r], 1), torch.cat([u_all, u_r], 1)
                kind = torch.cat([kind, torch.full((args.replay_progs,), 2, device=dev)])

            # reward at theta_e (Eq. 2): |d/dt L(y_i; theta_e + t v)|, L = the program's mean prediction error
            learner.snapshot(e)
            err, dl, dir_norm = learner.progress(e, obs, act)
            n_per = c or T
            r = (per_program(dl, c) / n_per).abs()

            # generator (Eq. 4): pool-normalised reward minus beta * the sampled log-ratio to the uniform prior
            logp = pi.log_prob(obs[:-1], u_all)
            log_ratio = logp.detach() + env.act_dim * LOG2           # log pi(a|s) - log g0(a), per step
            kl = per_program(log_ratio, c)
            adv = (r - r.mean()) / (r.std() + args.adv_eps) - args.beta * kl
            lp_prog = per_program(logp, c)
            fresh = kind.repeat(len(r) // len(kind)) == 0
            pg = -(lp_prog[fresh] * adv[fresh]).mean()
            ei = -(r / r.sum().clamp_min(1e-12) * lp_prog).sum() if args.ei else torch.zeros((), device=dev)
            if not args.frozen_policy:
                opt.zero_grad(set_to_none=True)
                (pg + args.ei * ei).backward()
                opt.step()
            if archive is not None:
                archive.step()
                archive.add(ro["u"], r[:E], ro["state"])
            if bank is not None:
                bank.add(ro["obs"], ro["act"], ro["u"])

            if wm_buf is None:                                   # learner (Eq. 1), after the reward was scored
                m = learner.step(obs, act)
            else:
                po, pa, po2 = obs[:-1].reshape(-1, env.obs_dim), act.reshape(-1, env.act_dim), obs[1:].reshape(-1, env.obs_dim)
                ex = (ro["obs"][:-1].reshape(-1, env.obs_dim), ro["act"].reshape(-1, env.act_dim),
                      ro["obs"][1:].reshape(-1, env.obs_dim))              # executed only: replayed rows are already in
                n_pool = round(args.wm_pool_frac * args.wm_batch)
                if not n_pool:
                    wm_buf.add(*ex)
                ms = []
                for _ in range(args.wm_steps):
                    i = torch.randint(0, len(po), (n_pool,), device=dev)
                    batch = [(po[i], pa[i], po2[i])]
                    if args.wm_batch - n_pool and wm_buf.n:              # round 0 of pool mixing: pool only
                        batch.append(wm_buf.sample(args.wm_batch - n_pool))
                    ms.append(learner.step_flat(*(torch.cat(x) for x in zip(*batch))))
                if n_pool:
                    wm_buf.add(*ex)
                m = {k: sum(x[k] for x in ms) / len(ms) for k in ms[0]}

            r_pool, r = r, r[:E]                                   # read-outs below: the executed envs only
            err, dl = err[:, :E], dl[:, :E]
            st = ro["state"]
            if args.save_replay:
                replay["state"].append(st[:, :args.save_replay].cpu())
                replay["act"].append(ro["act"][:, :args.save_replay].cpu())
            ag, bl = env.agent_xy(st), env.block_xy(st)
            ca, cb = grid_cells(ag, env.agent_lo, env.agent_hi), grid_cells(bl, env.block_lo, env.block_hi)
            visited[env.count_cell(st).reshape(-1)] = True
            contact, at_wall = ro["contact"], ro["wall"]
            free = ~contact & ~at_wall
            disp = (bl[-1] - bl[0]).norm(dim=-1)
            m.update({
                "iter": e, "env_steps": (e + 1) * T * E, "time": time.time() - t0,
                "agent_cov": episode_coverage(ca, 100), "block_cov": episode_coverage(cb, 100),
                "block_disp": disp.mean().item(), "block_moved_frac": (disp > 0.05).float().mean().item(),
                "block_path": ro["block_move"].sum(0).mean().item(),
                "contact_rate": contact.float().mean().item(), "wall_rate": at_wall.float().mean().item(),
                "cells_cum": visited.float().mean().item(),
                "r_int": r.mean().item(), "r_std": r.std().item(), "r_max": r.max().item(),
                "r_corr_contact": corr(r, per_program(contact.float(), c)),
                "r_corr_tpath": corr(r, per_program(ro["block_move"], c)),
                "r_corr_err": corr(r, per_program(err, c)),
                "dl_contact": masked_mean(dl, contact), "dl_wall": masked_mean(dl, at_wall & ~contact),
                "dl_free": masked_mean(dl, free), "dir_norm": dir_norm,
                "err_contact": masked_mean(err, contact), "err_wall": masked_mean(err, at_wall & ~contact),
                "err_free": masked_mean(err, free),
                "pg_loss": pg.item(), "kl_step": log_ratio.mean().item(), "adv_kl": (args.beta * kl).mean().item(),
                "policy_std": pi.log_std.exp().mean().item(),
            })
            if paper_pool:
                kp = kind.repeat(len(r_pool) // len(kind))
                m.update({"n_mut": n_mut, "n_replay": int((kind == 2).sum()), "ei_loss": ei.item(),
                          "niches": len(archive.cells) if archive is not None else 0,
                          **{f"r_{k}": masked_mean(r_pool, kp == j) for j, k in enumerate(("fresh", "mut", "replay"))}})
            mf.write(json.dumps(m) + "\n")
            mf.flush()

            if e >= tail_from:
                hist_agent += hist2d(ag)
                hist_block += hist2d(bl)
            if e % args.snap_every == 0 or e == args.rounds - 1:
                snaps.append({"iter": e, "agent": hist2d(ag).cpu(), "block": hist2d(bl).cpu()})
            if e % args.log_every == 0 or e == args.rounds - 1:
                print(f"[{name}] round {e:4d}  agent_cov {m['agent_cov']:.2f}  block_cov {m['block_cov']:.3f}  "
                      f"disp {m['block_disp']:.3f}  contact {m['contact_rate']:.3f}  wall {m['wall_rate']:.2f}  "
                      f"cells {m['cells_cum']:.3f}  std {m['policy_std']:.2f}  kl/step {m['kl_step']:.3f}  "
                      f"r {m['r_int']:.3g}±{m['r_std']:.2g}  corr c/T/err {m['r_corr_contact']:+.2f}/"
                      f"{m['r_corr_tpath']:+.2f}/{m['r_corr_err']:+.2f}  wm {m['wm_pred']:.4f}/{m['wm_sigreg']:.2f}  "
                      f"{m['time']:.0f}s", flush=True)

    torch.save({"args": vars(args), "policy": pi.state_dict(), "learner": learner.model.state_dict(), "wm": None,
                "hist_agent": hist_agent.cpu(), "hist_block": hist_block.cpu(), "snaps": snaps,
                "traj": ro["state"][:, :8].cpu(), "visited": visited.cpu()}, out / "final.pt")
    if args.save_replay:
        torch.save({"state": torch.stack(replay["state"]), "act": torch.stack(replay["act"]),
                    "env": "pusht", "start": args.start}, out / "replay.pt")
    env.close()
    print(f"[{name}] done in {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
