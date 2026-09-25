"""Does the agent understand Push-T? Open-loop prediction of held-out trajectories.

Protocol (pre-registered 2026-09-24, before any curiosity arm ran on Push-T):

  Question  Given the start state s_0 and the action sequence a_0..a_{h-1}, predict s_1..s_h by
            feeding the model its own predictions. Score the T (the object): the mean distance over
            its 8 corners, in px.
  Reference "The agent follows its command, the T stays still" -- the exact PD map for the agent,
            no physics for the T.
  Skill     mean over h in HS of log(MSE_ref(h) / MSE_model(h)); > 0 = better than the reference.
            Short (h <= 5) and long (h >= 10) reported separately; per-h skill is unbounded both
            ways, the log keeps catastrophic failures visible without clipping.
  Test sets (frozen; `build`)
      DEMO   all 206 human demos (pusht_data.py), a window from every 2nd frame, windows starting
             on an agent-T overlap (> 1 px; 0.25% of frames, not representable) dropped. PRIMARY.
             Shuffled-action control: the same windows with another window's actions; the
             action gain = skill(true) - skill(shuffled) must be > 0 for a model that uses the
             actions (a "slide the T to the goal" cheat scores well on the demos with gain 0).
      FREE   the agent moves and never touches the T: predicted T motion is invented motion.
      NEAR   the agent passes within 10 px of the T without touching it.
      PUSH   the agent pushes the T; the T never touches a wall (so pushes end by sliding off:
             T motion is over by step ~10 in most windows; h >= 25 mostly tests "the T stays at
             rest after contact").
      WALL   the agent pushes the T into a wall (windows where a T corner goes > 5 px past a
             wall's face -- the kinematic agent can squeeze the T through the thin wall in gym-pusht --
             are excluded).
      Sim sets: 1024 sequences x 50 steps, actions inside the agent's action box (|a| <= 1 per
      axis at scale 100, targets clamped to the arena), starts with the T fully inside the walls.
  Models
      E1 (primary) "data value": a fresh model trained from scratch on the arm's saved replay
             (replay.pt: 32 envs x every iteration = 1.92M transitions), 3 training seeds as an
             ensemble, standardised targets (plain MSE fits the agent's velocity -- 90% of the
             one-step variance -- and ignores the T). The standardisation constants are FROZEN with
             the test sets (from a fixed, seeded scripted-pusher dataset), so every fit optimises
             the same objective and only the data differs between arms. Learner "abs" (the observation) is primary;
             "tframe" (+ the agent's position, velocity and action in the T's frame) secondary.
             MSEs are averaged over the 3 seeds before the log.
      E2     the agent's own world model (pred / lp) at the end of training.
  Guards (reported; tolerances are set from the calibration, not in advance)
      invented motion: fraction of FREE / NEAR windows whose predicted T corners move > 5 px from
      s_0 by h = 10 (and 50); PUSH hit rate: fraction of PUSH windows with true motion > 5 px at
      h = 10 where the prediction also moves > 5 px. The invented rate grows with the learner's
      free-space drift, which grows with how much T motion its data holds -- read it next to the
      FREE T drift in px and the calibration ceiling's value, not as evidence on its own.
  Calibration (`calibrate`; must pass before any arm is read): references -- T static (0 by
      construction), the true simulator (the ceiling; restarted from s7 it is ~0.4 px/step off in
      contact, see push_t), a simulator with a 1.25x agent radius (geometry sensitivity: true sim
      >> wrong-radius sim is the metric resolving near-correct physics) -- and physics-free cheats,
      each fitted on DEMO itself: goal drift, a drag rule, a push-projection rule, push-projection +
      goal drift. Per start mode, E1 on a scripted pusher's data (ceiling) and on the none arm's
      data (floor), same N = 1.92M.
      PASS = E1 on the ceiling data beats the best physics-free cheat on the DEMO skill, action gain > 0.
      (Revised 2026-09-24 after the first calibration and a code review, before any curiosity arm
      ran: the first panel also required beating the 1.25x-radius simulator, which no learned model
      can -- it is the exact physics with a 3.75 px geometry error -- and its drag rule was unfitted;
      the first scripted pusher squeezed the T through walls in 23-29% of episodes and never roamed
      free space, so it was rewritten, and the random start was corrected to gym-pusht's.)
  Start mode (decided by `calibrate`, rule fixed before the floors were run): headroom(mode) =
      DEMO skill(E1 abs, ceiling data) - DEMO skill(E1 abs, none data); the curiosity arms run
      under the mode with the larger headroom, "random" (gym-pusht's reset) if within 0.05.
  Physics (amended 2026-09-25, after the first sweep): push_t physics="safe" -- thick walls, and an
      agent that cannot squeeze a wall-pinned T -- because under exact gym-pusht physics the pred / lp
      agents learned to jam the T into and through the thin walls (a chaotic simulator glitch acting
      as a noisy TV; toy/runs/pusht_v0_thinwalls). "safe" is bit-identical to gym-pusht whenever the T
      is not pinned against a wall, including on all 206 demos. Test sets, calibration and all arms
      are rebuilt under it; the start mode (fixed) decided under gym physics is kept -- the change
      touches wall contact only, 0.2% of the random agent's steps.
  Arms: none, pred, lp, count, ln; 3 seeds, then n from the observed variance. Each arm vs none:
      Welch t on the primary score, Holm-corrected over the 4 comparisons; "tframe" and E2 are
      secondary and must not contradict a claimed difference.

    python toy/pusht_eval.py build                              # freeze the test sets
    python toy/pusht_eval.py calibrate --start fixed --floor-run toy/runs/pusht/ptf_none_s0
    python toy/pusht_eval.py score toy/runs/pusht/ptf_pred_s0 [...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
from torch import nn

from push_t import (AGENT_R, ARENA, FIXED_START, WALL_HI, WALL_LO, PushT, Sim, T_RECTS, angle_diff, dist_to_t,
                    keypoints, pd_matrices, sample_starts, t_in_arena, to_t_frame)

HS = (1, 5, 10, 25, 50)
H = 50
SCALE = 100.0
TESTS = Path(__file__).resolve().parent / "data" / "pusht_tests.pt"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MOVE_TOL = 5.0                                                   # px of mean T-corner motion
GOAL = np.array([256.0, 256.0, math.pi / 4])
POOL = 12


# ================================================================== test sets
def _boundary_point(rng):
    """Uniform point on the T outline (body frame) and its outward normal."""
    segs = [((-60, 0), (60, 0), (0, -1)), ((60, 0), (60, 30), (1, 0)), ((60, 30), (15, 30), (0, 1)),
            ((15, 30), (15, 120), (1, 0)), ((15, 120), (-15, 120), (0, 1)), ((-15, 120), (-15, 30), (-1, 0)),
            ((-15, 30), (-60, 30), (0, 1)), ((-60, 30), (-60, 0), (-1, 0))]
    L = np.array([math.dist(a, b) for a, b, _ in segs])
    a, b, n = segs[rng.choice(len(segs), p=L / L.sum())]
    return np.array(a) + rng.uniform() * (np.array(b) - np.array(a)), np.array(n, dtype=float)


def _rot(v, th):
    c, s = math.cos(th), math.sin(th)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def _smooth(rng, th0, mag, jitter):
    th = th0 + np.cumsum(rng.normal(0, jitter, H))
    m = rng.uniform(*mag, H)
    return np.stack([m * np.cos(th), m * np.sin(th)], -1)


def _run(sim, st, deltas):
    """Execute relative PD-target offsets (px) as the agent would: per-axis clip to the action box,
    target clamped to the arena. Returns states (H+1, 7), applied actions (H, 2), flags (H, 2)."""
    sim.set(st)
    S, A, F = [sim.state()], [], []
    for d in deltas:
        x = S[-1][:2]
        tgt = np.clip(x + np.clip(d, -SCALE, SCALE), *ARENA)
        F.append(sim.step(tgt))
        A.append((tgt - x) / SCALE)
        S.append(sim.state())
    return np.array(S), np.array(A), np.array(F, dtype=bool)


def _t_moved(S):
    return np.linalg.norm(keypoints(S[1:, 4:7]) - keypoints(S[:-1, 4:7]), axis=-1).mean(-1) > 1e-3


def _stratum_worker(args):
    kind, seed, n = args
    rng, sim, out, tries = np.random.default_rng(seed), Sim(), [], 0
    while len(out) < n and tries < 500 * n:
        tries += 1
        if kind == "FREE":
            st = sample_starts(rng, 1)[0]
            st[2:4] = rng.normal(0, 60, 2)
            deltas = _smooth(rng, rng.uniform(0, 2 * math.pi), (5, 90), 0.3)
        else:
            tpos = rng.uniform(150, 362, 2)
            if kind == "WALL":                                   # T centre 65-110 px from one wall
                ax, side = rng.integers(2), rng.integers(2)
                tpos[ax] = rng.uniform(65, 110) if side == 0 else 512 - rng.uniform(65, 110)
            th = rng.uniform(-math.pi, math.pi)
            pose = np.array([*tpos, th])
            if not t_in_arena(pose, margin=1.0):
                continue
            p, nrm = _boundary_point(rng)
            pw, nw = tpos + _rot(p, th), _rot(nrm, th)
            if kind == "WALL":                                   # push from the side facing away from the wall
                wall_dir = np.zeros(2)
                wall_dir[ax] = -1.0 if side == 0 else 1.0
                if np.dot(nw, wall_dir) > -0.3:
                    continue
            gap = rng.uniform(3, 20) if kind == "NEAR" else rng.uniform(1, 25)
            ag = pw + nw * (AGENT_R + gap)
            if not ((ag > ARENA[0]).all() and (ag < ARENA[1]).all()):
                continue
            st = np.array([*ag, 0.0, 0.0, *pose])
            if dist_to_t(st[:2], st[4:7]) <= AGENT_R:
                continue
            if kind == "NEAR":                                   # slide along the outline, not into it
                tang = np.array([-nw[1], nw[0]]) * (1 if rng.uniform() < 0.5 else -1)
                deltas = _smooth(rng, math.atan2(tang[1], tang[0]) + rng.uniform(-0.3, 0.3), (5, 30), 0.15)
            else:
                th0 = math.atan2(-nw[1], -nw[0]) + rng.uniform(-math.pi / 4, math.pi / 4)
                deltas = _smooth(rng, th0, (10, 60), 0.1)
        S, A, F = _run(sim, st, deltas)
        moved, con, wall = _t_moved(S), F[:, 0], F[:, 1]
        if kind == "FREE":
            ok = not moved.any() and not con.any()
        elif kind == "NEAR":
            gaps = dist_to_t(S[:, :2], S[:, 4:7]) - AGENT_R
            ok = not moved.any() and not con.any() and gaps.min() < 10
        elif kind == "PUSH":
            ok = moved.any() and not wall.any()
        else:
            ok = wall.any() and bool(t_in_arena(S[:, 4:7], margin=-5.0).all())
        if ok:
            out.append((S, A, con, wall))
    return out


def _demo_sets():
    from pusht_data import load_demos
    eps, sets = load_demos(), {}
    for h in HS:
        s0, act, S, sup, ep_id = [], [], [], [], []
        for e, ep in enumerate(eps):
            s7, tg = ep["s7"], ep["target"]
            for t0 in range(0, len(s7) - h, 2):
                if dist_to_t(s7[t0, :2], s7[t0, 4:7]) < AGENT_R - 1.0:
                    continue
                rel = tg[t0:t0 + h] - s7[t0:t0 + h, :2]
                s0.append(s7[t0])
                act.append(rel / SCALE)
                S.append(s7[t0:t0 + h + 1])
                sup.append(bool((np.abs(rel) <= SCALE).all() and (tg[t0:t0 + h] >= ARENA[0]).all()
                                and (tg[t0:t0 + h] <= ARENA[1]).all()))
                ep_id.append(e)
        sets[h] = {"S": torch.tensor(np.array(S), dtype=torch.float32),
                   "act": torch.tensor(np.array(act), dtype=torch.float32),
                   "support": torch.tensor(sup), "episode": torch.tensor(ep_id)}
    return sets


def build(n_per=1024):
    t0 = time.time()
    tests = {"DEMO": _demo_sets()}
    with get_context("fork").Pool(POOL) as pool:
        for k, kind in enumerate(("FREE", "NEAR", "PUSH", "WALL")):
            per = -(-n_per // POOL)
            chunks = pool.map(_stratum_worker, [(kind, 1_000_000 * (k + 1) + i, per) for i in range(POOL)])
            items = [x for c in chunks for x in c][:n_per]
            tests[kind] = {"S": torch.tensor(np.stack([x[0] for x in items]), dtype=torch.float32),
                           "act": torch.tensor(np.stack([x[1] for x in items]), dtype=torch.float32),
                           "contact": torch.tensor(np.stack([x[2] for x in items])),
                           "wall": torch.tensor(np.stack([x[3] for x in items]))}
    s, a, s2, _ = scripted_data("fixed", n=480_000, seed=12345)
    d = PushT.obs_from_state(s2) - PushT.obs_from_state(s)
    tests["norm"] = {"mu": d.mean(0), "sd": d.std(0).clamp_min(1e-6)}
    TESTS.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tests, TESTS)
    print(f"[build] test sets in {time.time() - t0:.0f}s -> {TESTS} (sha1 {_sha1(TESTS)})")
    describe(tests)


def _sha1(p):
    return hashlib.sha1(Path(p).read_bytes()).hexdigest()[:12]


def load_tests():
    if not TESTS.exists():
        build()
    return torch.load(TESTS)


def describe(tests):
    for name, d in tests.items():
        if name == "norm":
            print("  E1 standardisation (obs-delta sd): " + " ".join(f"{v:.4f}" for v in d["sd"].tolist()))
            continue
        if name == "DEMO":
            for h in HS:
                S = d[h]["S"].double().numpy()
                ref = np.linalg.norm(keypoints(S[:, h, 4:7]) - keypoints(S[:, 0, 4:7]), axis=-1).mean(-1)
                print(f"  DEMO h={h:2d}: {len(S):6d} windows ({d[h]['support'].float().mean():.1%} inside the action box); "
                      f"T-static RMS {np.sqrt((ref ** 2).mean()):6.2f} px, T moved > {MOVE_TOL:.0f} px in {np.mean(ref > MOVE_TOL):.1%}")
        else:
            S = d["S"].double().numpy()
            moved = np.linalg.norm(keypoints(S[:, 1:, 4:7]) - keypoints(S[:, :-1, 4:7]), axis=-1).mean(-1) > 1e-3
            ref = [np.sqrt((np.linalg.norm(keypoints(S[:, h, 4:7]) - keypoints(S[:, 0, 4:7]), axis=-1).mean(-1) ** 2).mean())
                   for h in HS]
            print(f"  {name:4s}: n={len(S)}; T moves in {moved.mean():.1%} of steps, contact {d['contact'].float().mean():.1%}, "
                  f"T-wall {d['wall'].float().mean():.1%}; T-static RMS at h={HS}: " + " ".join(f"{r:.1f}" for r in ref))


# ================================================================== predictors
# A predictor maps s0 (N, 7) float32 on DEV and actions (N, h, 2) (applied PD-target offset / 100)
# to predicted states (N, h + 1, 7).
_M, _N = pd_matrices()


def _pd_agent(s0, act):
    """Exact agent trajectory (N, h + 1, 4): position, velocity."""
    x, v = s0[:, 0:2].double(), s0[:, 2:4].double()
    out = [torch.cat([x, v], -1)]
    for t in range(act.shape[1]):
        tgt = x + SCALE * act[:, t].double()
        x, v = _M[0, 0] * x + _M[0, 1] * v + _N[0] * tgt, _M[1, 0] * x + _M[1, 1] * v + _N[1] * tgt
        out.append(torch.cat([x, v], -1))
    return torch.stack(out, 1).float()


def static_predictor(s0, act):
    ag = _pd_agent(s0, act)
    return torch.cat([ag, s0[:, None, 4:7].expand(-1, act.shape[1] + 1, -1)], -1)


def goal_drift_predictor(c):
    """Cheat: ignore the actions, slide the T toward the demos' goal pose at rate c per step."""
    def f(s0, act):
        ag = _pd_agent(s0, act)
        p, th = s0[:, 4:6].clone(), s0[:, 6].clone()
        g = torch.as_tensor(GOAL, dtype=torch.float32, device=s0.device)
        out = [torch.cat([p, th[:, None]], -1)]
        for _ in range(act.shape[1]):
            p = p + c * (g[:2] - p)
            th = th + c * torch.remainder(g[2] - th + math.pi, 2 * math.pi).sub(math.pi)
            out.append(torch.cat([p, th[:, None]], -1))
        return torch.cat([ag, torch.stack(out, 1)], -1)
    return f


def drag_predictor(k=0.8, reach=2.0):
    """Cheat: when the agent touches the T, the T translates by k x the agent's displacement."""
    def f(s0, act):
        ag = _pd_agent(s0, act)
        pose = s0[:, 4:7].clone()
        out = [pose]
        for t in range(act.shape[1]):
            touch = dist_to_t(ag[:, t, :2], pose) < AGENT_R + reach
            d = ag[:, t + 1, :2] - ag[:, t, :2]
            pose = torch.cat([pose[:, :2] + k * d * touch[:, None], pose[:, 2:]], -1)
            out.append(pose)
        return torch.cat([ag, torch.stack(out, 1)], -1)
    return f


def _nearest_t_normal(p, pose):
    """Unit vector from the nearest point of the T to world points p (world frame), and the gap."""
    q = to_t_frame(p, pose)
    best_d, best_n = None, None
    for cx, cy, hx, hy in T_RECTS:
        nx = q[..., 0] - (q[..., 0].clamp(cx - hx, cx + hx))
        ny = q[..., 1] - (q[..., 1].clamp(cy - hy, cy + hy))
        d = torch.sqrt(nx * nx + ny * ny)
        if best_d is None:
            best_d, best_n = d, torch.stack([nx, ny], -1)
        else:
            m = d < best_d
            best_d = torch.where(m, d, best_d)
            best_n = torch.where(m[..., None], torch.stack([nx, ny], -1), best_n)
    n = best_n / best_d[..., None].clamp_min(1e-6)
    c, s = torch.cos(pose[..., 2]), torch.sin(pose[..., 2])
    return torch.stack([c * n[..., 0] - s * n[..., 1], s * n[..., 0] + c * n[..., 1]], -1), best_d


def push_predictor(k=0.8, reach=5.0, c=0.0):
    """Cheat: within `reach` px of contact, the T translates by k x the part of the agent's
    displacement that points into the T (no rotation, no walls, no T geometry beyond a distance
    test); optionally plus goal drift at rate c."""
    def f(s0, act):
        ag = _pd_agent(s0, act)
        pose = s0[:, 4:7].clone()
        g = torch.as_tensor(GOAL, dtype=torch.float32, device=s0.device)
        out = [pose]
        for t in range(act.shape[1]):
            n, gap = _nearest_t_normal(ag[:, t, :2], pose)
            d = ag[:, t + 1, :2] - ag[:, t, :2]
            into = (-(d * n).sum(-1)).clamp_min(0) * (gap < AGENT_R + reach)
            p = pose[:, :2] - k * into[:, None] * n
            th = pose[:, 2]
            if c:
                p = p + c * (g[:2] - p)
                th = th + c * torch.remainder(g[2] - th + math.pi, 2 * math.pi).sub(math.pi)
            pose = torch.cat([p, th[:, None]], -1)
            out.append(pose)
        return torch.cat([ag, torch.stack(out, 1)], -1)
    return f


def _sim_worker(args):
    s0, act, agent_r = args
    sim, out = Sim(agent_r), []
    for s, a in zip(s0, act):
        sim.set(s)
        tr = [sim.state()]
        for u in a:
            sim.step(tr[-1][:2] + SCALE * u)                      # no clamp: reproduce the recorded targets
            tr.append(sim.state())
        out.append(np.array(tr))
    return out


def sim_predictor(agent_r=AGENT_R):
    """The simulator itself (agent_r = 15: the physics ceiling; != 15: a geometry-sensitivity reference)."""
    def f(s0, act):
        s0n, an = s0.double().cpu().numpy(), act.double().cpu().numpy()
        idx = np.array_split(np.arange(len(s0n)), POOL * 4)
        with get_context("fork").Pool(POOL) as pool:
            parts = pool.map(_sim_worker, [(s0n[i], an[i], agent_r) for i in idx])
        return torch.tensor(np.concatenate([np.array(p).reshape(-1, act.shape[1] + 1, 7) for p in parts if len(p)]),
                            dtype=torch.float32, device=s0.device)
    return f


# ------------------------------------------------------------------ learned models
def tframe_feats(s, a):
    """Agent position / velocity and the action in the T's body frame."""
    pose = s[..., 4:7]
    p = to_t_frame(s[..., 0:2], pose) / 128
    c, sn = torch.cos(pose[..., 2]), torch.sin(pose[..., 2])
    rot = lambda v: torch.stack([c * v[..., 0] + sn * v[..., 1], -sn * v[..., 0] + c * v[..., 1]], -1)
    return torch.cat([p, rot(s[..., 2:4]) / 500, rot(a)], -1)


def features(s, a, kind):
    x = torch.cat([PushT.obs_from_state(s), a], -1)
    return torch.cat([x, tframe_feats(s, a)], -1) if kind == "tframe" else x


class EnsembleMLP(nn.Module):
    """S independent MLPs evaluated in one batched matmul (x: (S, B, din))."""

    def __init__(self, S, din, dout, hidden=512, layers=3):
        super().__init__()
        dims = [din] + [hidden] * layers + [dout]
        self.W, self.b = nn.ParameterList(), nn.ParameterList()
        for i, o in zip(dims[:-1], dims[1:]):
            bound = 1 / math.sqrt(i)
            self.W.append(nn.Parameter(torch.empty(S, i, o).uniform_(-bound, bound)))
            self.b.append(nn.Parameter(torch.empty(S, 1, o).uniform_(-bound, bound)))

    def forward(self, x):
        for k, (W, b) in enumerate(zip(self.W, self.b)):
            x = torch.baddbmm(b, x, W)
            if k < len(self.W) - 1:
                x = nn.functional.elu(x)
        return x


class Learner:
    """E1: fresh ensemble, standardised one-step obs delta, early-stopped per member on held-out episodes."""

    def __init__(self, norm, kind="abs", seeds=3, steps=40_000, batch=2048, lr=1e-3):
        self.kind, self.S, self.steps, self.batch, self.lr = kind, seeds, steps, batch, lr
        self.mu, self.sd = norm["mu"].to(DEV), norm["sd"].to(DEV)

    def fit(self, s, a, s2, val_frac=0.05, episode=None, seed=0, log=None):
        g = torch.Generator(device=DEV).manual_seed(seed)
        torch.manual_seed(seed)
        s, a, s2 = s.to(DEV), a.to(DEV), s2.to(DEV)
        n = s.shape[0]
        if episode is None:
            episode = torch.arange(n, device=DEV) // 200
        episode = episode.to(DEV)
        ueps = torch.unique(episode)
        val_eps = ueps[torch.randperm(len(ueps), device=DEV, generator=g)[:max(1, int(val_frac * len(ueps)))]]
        is_val = torch.isin(episode, val_eps)
        tr, va = torch.nonzero(~is_val).squeeze(1), torch.nonzero(is_val).squeeze(1)[:100_000]
        d = PushT.obs_from_state(s2) - PushT.obs_from_state(s)
        din = features(s[:1], a[:1], self.kind).shape[-1]
        self.net = EnsembleMLP(self.S, din, 8).to(DEV)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.steps, eta_min=self.lr * 0.01)
        tgt = (d - self.mu) / self.sd
        xv, yv = features(s[va], a[va], self.kind), tgt[va]
        best = torch.full((self.S,), float("inf"), device=DEV)
        best_state = [p.detach().clone() for p in self.net.parameters()]
        self.curve = []
        for it in range(1, self.steps + 1):
            idx = tr[torch.randint(0, len(tr), (self.S, self.batch), device=DEV, generator=g)]
            x = features(s[idx], a[idx], self.kind)
            loss = (self.net(x) - tgt[idx]).pow(2).mean((1, 2))
            opt.zero_grad(set_to_none=True)
            loss.sum().backward()
            opt.step()
            sched.step()
            if it % 2000 == 0 or it == self.steps:
                with torch.no_grad():
                    vl = (self.net(xv.expand(self.S, -1, -1)) - yv).pow(2).mean((1, 2))
                better = vl < best
                best = torch.where(better, vl, best)
                for p, bp in zip(self.net.parameters(), best_state):
                    bp[better] = p.detach()[better]
                self.curve.append((it, loss.detach().cpu().tolist(), vl.cpu().tolist()))
                if log:
                    log(f"    step {it:6d}  train {loss.mean().item():.4f}  val {vl.mean().item():.4f}")
        with torch.no_grad():
            for p, bp in zip(self.net.parameters(), best_state):
                p.copy_(bp)
        self.best_val = best.cpu().tolist()
        return self

    @torch.no_grad()
    def rollout(self, s0, act):
        """Per-member predictions (S, N, h + 1, 7)."""
        s = s0.to(DEV).unsqueeze(0).expand(self.S, -1, -1)
        out = [s]
        for t in range(act.shape[1]):
            a = act[:, t].to(DEV).unsqueeze(0).expand(self.S, -1, -1)
            o = PushT.obs_from_state(s) + self.net(features(s, a, self.kind)) * self.sd + self.mu
            o = torch.cat([o[..., :6], o[..., 6:8] / o[..., 6:8].norm(dim=-1, keepdim=True).clamp_min(1e-6)], -1)
            s = PushT.state_from_obs(o)
            out.append(s)
        return torch.stack(out, 2)


class AgentWM:
    """E2: the agent's own residual dynamics model (signals.Dynamics) on the Push-T observation."""

    def __init__(self, state_dict):
        from signals import Dynamics
        self.m = Dynamics(PushT.obs_dim, PushT.act_dim).to(DEV)
        self.m.load_state_dict(state_dict)
        self.m.eval()

    @torch.no_grad()
    def __call__(self, s0, act):
        o = PushT.obs_from_state(s0.to(DEV))
        out = [s0.to(DEV)]
        for t in range(act.shape[1]):
            o = self.m(o, act[:, t].to(DEV))
            o = torch.cat([o[..., :6], o[..., 6:8] / o[..., 6:8].norm(dim=-1, keepdim=True).clamp_min(1e-6)], -1)
            out.append(PushT.state_from_obs(o))
        return torch.stack(out, 1)


# ================================================================== scoring
def _kp_err(P, S):
    """Mean T-corner distance (..., ) between predicted and true poses (..., 3)."""
    return (keypoints(P.double()) - keypoints(S.double())).norm(dim=-1).mean(-1)


@torch.no_grad()
def evaluate(predict, tests, members=False, shuffle_seed=0):
    """Score a predictor on every test set. `members`: predict returns (S, N, h+1, 7) -- per-member
    MSEs are averaged before the log. Returns nested dict of floats."""
    def run(s0, act):
        P = predict(s0.to(DEV), act.to(DEV))
        return P if members else P.unsqueeze(0)

    res = {}
    for name, d in tests.items():
        if name == "norm":
            continue
        r = {}
        for h in HS:
            if name == "DEMO":
                S, act, sup = d[h]["S"].to(DEV), d[h]["act"].to(DEV), d[h]["support"].to(DEV)
            else:
                S, act, sup = d["S"][:, :h + 1].to(DEV), d["act"][:, :h].to(DEV), None
            P = run(S[:, 0], act)                                                    # (S, N, h+1, 7)
            e = _kp_err(P[:, :, h, 4:7], S[None, :, h, 4:7])                         # (S, N)
            ref = _kp_err(S[:, 0, 4:7], S[:, h, 4:7])                                # (N,)
            pred_move = _kp_err(P[:, :, h, 4:7], S[None, :, 0, 4:7])                 # (S, N)
            mse, mse_ref = (e ** 2).mean(1).mean().item(), (ref ** 2).mean().item()
            q = {"mse": mse, "mse_ref": mse_ref,                         # FREE / NEAR: the reference is exact
                 "skill": math.log(mse_ref / max(mse, 1e-9)) if mse_ref > 1e-6 else float("nan"),
                 "agent_rmse": (P[:, :, h, :2] - S[None, :, h, :2]).norm(dim=-1).pow(2).mean().sqrt().item(),
                 "pos_rmse": (P[:, :, h, 4:6] - S[None, :, h, 4:6]).norm(dim=-1).pow(2).mean().sqrt().item(),
                 "ang_rmse_deg": math.degrees(angle_diff(P[:, :, h, 6], S[None, :, h, 6]).pow(2).mean().sqrt().item()),
                 "invented": (pred_move > MOVE_TOL).float().mean().item()}
            true_move = ref > MOVE_TOL
            q["hit"] = (pred_move[:, true_move] > MOVE_TOL).float().mean().item() if true_move.any() else float("nan")
            if sup is not None:
                es = (e[:, sup] ** 2).mean(1).mean().item()
                q["skill_support"] = math.log(max((ref[sup] ** 2).mean().item(), 1e-9) / max(es, 1e-9))
                perm = torch.randperm(len(act), generator=torch.Generator().manual_seed(shuffle_seed + h)).to(DEV)
                Ps = run(S[:, 0], act[perm])
                es = (_kp_err(Ps[:, :, h, 4:7], S[None, :, h, 4:7]) ** 2).mean(1).mean().item()
                q["skill_shuffled"] = math.log(max(mse_ref, 1e-9) / max(es, 1e-9))
            r[h] = q
        summ = {"skill": float(np.mean([r[h]["skill"] for h in HS])),                # nan for FREE / NEAR
                "skill_short": float(np.mean([r[h]["skill"] for h in HS if h <= 5])),
                "skill_long": float(np.mean([r[h]["skill"] for h in HS if h >= 10]))}
        if name == "DEMO":
            summ["skill_shuffled"] = float(np.mean([r[h]["skill_shuffled"] for h in HS]))
            summ["action_gain"] = summ["skill"] - summ["skill_shuffled"]
            summ["skill_support"] = float(np.mean([r[h]["skill_support"] for h in HS]))
        res[name] = {"per_h": r, **summ}
    return res


def headline(res):
    """One-line summary of an evaluate() result."""
    D = res["DEMO"]
    return (f"DEMO skill {D['skill']:+.3f} (short {D['skill_short']:+.3f} long {D['skill_long']:+.3f}, "
            f"action gain {D['action_gain']:+.3f}) | PUSH {res['PUSH']['skill']:+.3f} WALL {res['WALL']['skill']:+.3f} | "
            f"invented FREE/NEAR h10 {res['FREE']['per_h'][10]['invented']:.3f}/{res['NEAR']['per_h'][10]['invented']:.3f} | "
            f"FREE T drift h10/h50 {res['FREE']['per_h'][10]['pos_rmse']:.1f}/{res['FREE']['per_h'][50]['pos_rmse']:.1f} px | "
            f"PUSH hit h10 {res['PUSH']['per_h'][10]['hit']:.3f} | agent RMSE DEMO h50 {D['per_h'][50]['agent_rmse']:.1f} px")


# ================================================================== training data
def load_replay(run_dir):
    """(s, a, s2, episode) from a run's replay.pt: (iters, T+1, K, 7) states, (iters, T, K, 2) actions."""
    r = torch.load(Path(run_dir) / "replay.pt")
    S, A = r["state"].float(), r["act"].float()
    I, T1, K, _ = S.shape
    s = S[:, :-1].permute(0, 2, 1, 3).reshape(-1, 7)
    s2 = S[:, 1:].permute(0, 2, 1, 3).reshape(-1, 7)
    a = A.permute(0, 2, 1, 3).reshape(-1, 2)
    ep = torch.arange(I * K).repeat_interleave(T1 - 1)
    return s, a, s2, ep


def _clearance(pose, direction):
    """Free px between the T and the walls ahead of it along `direction` (axes with |component| > 0.3)."""
    k, c = keypoints(pose), np.inf
    if direction[0] > 0.3:
        c = min(c, WALL_HI - k[:, 0].max())
    if direction[0] < -0.3:
        c = min(c, k[:, 0].min() - WALL_LO)
    if direction[1] > 0.3:
        c = min(c, WALL_HI - k[:, 1].max())
    if direction[1] < -0.3:
        c = min(c, k[:, 1].min() - WALL_LO)
    return c


def _scripted_worker(args):
    """Scripted pusher (the calibration ceiling's data). Segments alternate free roaming (35%: a
    smooth random walk) with pushes: walk to a random point on the T outline, push inward for 3-12
    steps. Near a wall (T centre of gravity within 100 px) a push must head back toward the arena
    centre, never toward a wall closer than 40 px, and it is abandoned when the T comes within 10 px
    of one. An episode is cut when a T corner goes > 2 px into a wall (the kinematic agent can
    squeeze the T through gym-pusht's thin walls), so the data never contains tunnelling.
    Returns flat transitions (s, a, s2, episode id) totalling n_steps."""
    start, seed, n_steps, T = args
    rng, sim = np.random.default_rng(seed), Sim()
    s_all, a_all, s2_all, ep_all, ep = [], [], [], [], 0
    centre = np.array([256.0, 256.0])
    while len(s_all) < n_steps:
        st = np.array(FIXED_START) if start == "fixed" else sample_starts(rng, 1)[0]
        sim.set(st)
        x = sim.state()
        roam = rng.uniform() < 0.35
        phase, left, goal_pt, heading = ("roam" if roam else "approach"), int(rng.integers(5, 21)), None, rng.uniform(0, 2 * math.pi)
        for _ in range(T):
            pose = x[4:7]
            if phase == "roam":
                heading += rng.normal(0, 0.3)
                d = rng.uniform(10, 90) * np.array([math.cos(heading), math.sin(heading)])
                left -= 1
                if left <= 0:
                    phase, goal_pt = "approach", None
            elif phase == "approach":
                if goal_pt is None:
                    cog = pose[:2] + _rot(np.array([0.0, 45.0]), pose[2])
                    near_wall = min(cog.min() - WALL_LO, WALL_HI - cog.max()) < 100
                    for _ in range(30):
                        p, nrm = _boundary_point(rng)
                        pw, nw = pose[:2] + _rot(p, pose[2]), _rot(nrm, pose[2])
                        gp = pw + nw * (AGENT_R + rng.uniform(3, 12))
                        pdir = _rot(-nw, rng.uniform(-math.pi / 4, math.pi / 4))
                        if ((gp > ARENA[0]).all() and (gp < ARENA[1]).all() and _clearance(pose, pdir) > 40
                                and (not near_wall or np.dot(pdir, centre - cog) > 0)):
                            goal_pt, push_dir = gp, pdir
                            break
                    if goal_pt is None:                          # no safe push: roam
                        phase, left, heading = "roam", int(rng.integers(5, 21)), rng.uniform(0, 2 * math.pi)
                        d = np.zeros(2)
                    else:
                        left = 12
                if goal_pt is not None:
                    d = goal_pt - x[:2]
                    left -= 1
                    if np.linalg.norm(d) < 5 or left <= 0:
                        phase, left, mag = "push", int(rng.integers(3, 13)), rng.uniform(15, 70)
            else:
                if _clearance(pose, push_dir) < 10:
                    d, left = -push_dir * 30, 0                  # back off instead of squeezing
                else:
                    d = push_dir * mag
                    left -= 1
                if left <= 0:
                    if rng.uniform() < 0.35:
                        phase, left, heading = "roam", int(rng.integers(5, 21)), rng.uniform(0, 2 * math.pi)
                    else:
                        phase, goal_pt = "approach", None
            d = d + rng.normal(0, 5, 2)
            tgt = np.clip(x[:2] + np.clip(d, -SCALE, SCALE), *ARENA)
            sim.step(tgt)
            x2 = sim.state()
            if not t_in_arena(x2[4:7], margin=-2.0):
                break
            s_all.append(x)
            a_all.append((tgt - x[:2]) / SCALE)
            s2_all.append(x2)
            ep_all.append(ep)
            x = x2
        ep += 1
    return np.array(s_all[:n_steps]), np.array(a_all[:n_steps]), np.array(s2_all[:n_steps]), np.array(ep_all[:n_steps])


def scripted_data(start, n=1_920_000, T=200, seed=0):
    with get_context("fork").Pool(POOL) as pool:
        per = -(-n // POOL)
        parts = pool.map(_scripted_worker, [(start, 10_000 + 97 * seed + i, per, T) for i in range(POOL)])
    f = lambda j: np.concatenate([p[j] for p in parts])[:n]
    ep = np.concatenate([p[3] + 1_000_000 * i for i, p in enumerate(parts)])[:n]
    return (torch.tensor(f(0), dtype=torch.float32), torch.tensor(f(1), dtype=torch.float32),
            torch.tensor(f(2), dtype=torch.float32), torch.tensor(ep))


def moved_frac(s, s2):
    return ((keypoints(s2[:, 4:7].double()) - keypoints(s[:, 4:7].double())).norm(dim=-1).mean(-1) > 1.0).float().mean().item()


def e1(s, a, s2, ep, kinds=("abs", "tframe"), steps=40_000, seed=0, tests=None, log=print):
    tests = tests or load_tests()
    out = {}
    for kind in kinds:
        t0 = time.time()
        L = Learner(tests["norm"], kind, steps=steps).fit(s, a, s2, episode=ep, seed=seed)
        out[kind] = evaluate(L.rollout, tests, members=True)
        out[kind]["best_val"] = L.best_val
        out[kind]["curve"] = L.curve
        log(f"  E1[{kind}] ({time.time() - t0:.0f}s): {headline(out[kind])}")
    return out


def _strip(res):
    """JSON-safe copy (int keys -> str)."""
    if isinstance(res, dict):
        return {str(k): _strip(v) for k, v in res.items()}
    return res


# ================================================================== CLI
def _fit_cheat(name, make, grid, tests, log):
    """Fit a physics-free cheat's parameters on DEMO itself (the most favourable case for it)."""
    best = None
    for params in grid:
        r = evaluate(make(*params), {"DEMO": tests["DEMO"]})["DEMO"]
        if best is None or r["skill"] > best[1]:
            best = (params, r["skill"])
    res = evaluate(make(*best[0]), tests)
    res["params"] = list(best[0])
    log(f"  {name:12s} {best[0]}: {headline(res)}")
    return res


def cmd_calibrate(args):
    tests = load_tests()
    res, log = {"tests_sha1": _sha1(TESTS)}, lambda m: print(m, flush=True)
    log("[calibrate] references (T static; the simulator; the simulator with a 1.25x agent = geometry sensitivity)")
    for name, f in (("static", static_predictor), ("sim", sim_predictor()), ("sim_r1.25", sim_predictor(AGENT_R * 1.25))):
        t0 = time.time()
        res[name] = evaluate(f, tests)
        log(f"  {name:12s} ({time.time() - t0:.0f}s): {headline(res[name])}")
    log("[calibrate] physics-free cheats, each fitted on DEMO")
    cheats = {
        "goal_drift": _fit_cheat("goal_drift", goal_drift_predictor, [(c,) for c in (0.005, 0.01, 0.02, 0.03, 0.05)], tests, log),
        "drag": _fit_cheat("drag", drag_predictor, [(k, r) for k in (0.3, 0.5, 0.8, 1.0) for r in (0.0, 2.0, 5.0)], tests, log),
        "push": _fit_cheat("push", push_predictor, [(k, r) for k in (0.5, 0.8, 1.0) for r in (2.0, 5.0, 10.0)], tests, log),
    }
    k, r = cheats["push"]["params"]
    cheats["push_goal"] = _fit_cheat("push+goal", push_predictor, [(k, r, c) for c in (0.005, 0.01, 0.02)], tests, log)
    res["cheats"] = cheats
    res["best_cheat"] = max(cheats, key=lambda n: cheats[n]["DEMO"]["skill"])
    log(f"  best physics-free cheat: {res['best_cheat']} DEMO skill {cheats[res['best_cheat']]['DEMO']['skill']:+.3f}")
    for start in args.start:
        log(f"[calibrate] start={start}: ceiling = scripted pusher data, floor = the none arm's replay")
        s, a, s2, ep = scripted_data(start)
        gap = dist_to_t(s[:, :2].double(), s[:, 4:7].double()) - AGENT_R
        log(f"  scripted {start}: {len(s)} transitions, T moves (>1 px) in {moved_frac(s, s2):.1%}, "
            f"agent > 30 px from the T in {(gap > 30).float().mean():.1%}, T inside the walls in "
            f"{t_in_arena(s[:, 4:7].double()).float().mean():.1%}")
        res[f"ceiling_{start}"] = e1(s, a, s2, ep, steps=args.steps, tests=tests, log=log)
        floor_run = Path(args.runs) / f"pt{start[0]}_none_s0"
        if (floor_run / "replay.pt").exists():
            s, a, s2, ep = load_replay(floor_run)
            log(f"  none {start}: {len(s)} transitions, T moves (>1 px) in {moved_frac(s, s2):.1%}")
            res[f"floor_{start}"] = e1(s, a, s2, ep, steps=args.steps, tests=tests, log=log)
        else:
            log(f"  (no {floor_run}/replay.pt -- floor skipped)")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(_strip(res), open(out, "w"), indent=1)
    log(f"[calibrate] -> {out}")


def cmd_decide(args):
    """PASS per start mode and the pre-registered start-mode rule, from the calibration files."""
    rows = {}
    for f in args.files:
        c = json.load(open(f))
        best = c["cheats"][c["best_cheat"]]["DEMO"]["skill"]
        for key in c:
            if key.startswith("ceiling_"):
                mode = key.split("_", 1)[1]
                ceil, floor = c[key]["abs"]["DEMO"], c.get(f"floor_{mode}", {}).get("abs", {}).get("DEMO")
                rows[mode] = {"ceiling": ceil["skill"], "gain": ceil["action_gain"], "best_cheat": best,
                              "cheat": c["best_cheat"], "floor": floor["skill"] if floor else float("nan")}
    for mode, r in rows.items():
        r["pass"] = r["ceiling"] > r["best_cheat"] and r["gain"] > 0
        r["headroom"] = r["ceiling"] - r["floor"]
        print(f"  {mode:6s}: ceiling {r['ceiling']:+.3f} (gain {r['gain']:+.3f}) vs best cheat {r['cheat']} "
              f"{r['best_cheat']:+.3f} -> {'PASS' if r['pass'] else 'FAIL'}; floor {r['floor']:+.3f}; headroom {r['headroom']:+.3f}")
    if {"fixed", "random"} <= set(rows):
        f, r = rows["fixed"]["headroom"], rows["random"]["headroom"]
        mode = "random" if abs(f - r) <= 0.05 else ("fixed" if f > r else "random")
        print(f"  start-mode rule (larger headroom; random if within 0.05): {mode}")


def cmd_score(args):
    tests = load_tests()
    for run in args.runs:
        run = Path(run)
        out = run / "eval.json"
        if out.exists() and not args.force:
            print(f"[score] {run.name}: exists, skipped")
            continue
        t0 = time.time()
        s, a, s2, ep = load_replay(run)
        res = {"tests_sha1": _sha1(TESTS), "n": len(s), "t_moved_frac": moved_frac(s, s2)}
        print(f"[score] {run.name}: {len(s)} transitions, T moves (>1 px) in {res['t_moved_frac']:.2%}", flush=True)
        res["E1"] = e1(s, a, s2, ep, steps=args.steps, tests=tests, log=lambda m: print(m, flush=True))
        fin = torch.load(run / "final.pt", weights_only=False)
        if fin.get("wm") is not None:
            res["E2"] = evaluate(AgentWM(fin["wm"]), tests)
            print(f"  E2 (agent's own model): {headline(res['E2'])}", flush=True)
        json.dump(_strip(res), open(out, "w"), indent=1)
        print(f"[score] {run.name} done in {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    c = sub.add_parser("calibrate")
    c.add_argument("--start", nargs="+", default=["fixed", "random"])
    c.add_argument("--runs", default=str(Path(__file__).resolve().parent / "runs" / "pusht"))
    c.add_argument("--steps", type=int, default=40_000)
    c.add_argument("--out", default=str(Path(__file__).resolve().parent / "runs" / "pusht" / "calibration.json"))
    d = sub.add_parser("decide")
    d.add_argument("files", nargs="+")
    s = sub.add_parser("score")
    s.add_argument("runs", nargs="+")
    s.add_argument("--steps", type=int, default=40_000)
    s.add_argument("--force", action="store_true")
    a = p.parse_args()
    {"build": lambda: build(), "calibrate": lambda: cmd_calibrate(a), "decide": lambda: cmd_decide(a),
     "score": lambda: cmd_score(a)}[a.cmd]()
