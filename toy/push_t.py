"""Push-T: a disk agent pushing a T-shaped block -- gym-pusht's physics, batched for PPO.

`Sim` rebuilds gym-pusht 0.1.6's pymunk scene (PushTEnv._setup, pusht.py:428-452: same bodies,
add order, the doubled-inertia and friction-attribute quirks) and steps it with the env's control loop (pusht.py:242-253: 10 substeps of 0.01 s, PD on a KINEMATIC agent, k_p 100,
k_v 20) -- without gym, pygame or the per-step coverage computation. Needs pymunk 6
(gym-pusht's `add_collision_handler` is gone in pymunk 7).

physics="safe" (default) vs "gym": gym-pusht's agent is kinematic -- infinite force, nothing stops
it -- and its walls are 2 px-radius segments, so the agent can squeeze a T that touches a wall into
it, the T is ejected violently or tunnels through. The first Push-T sweep
(toy/runs/pusht_v0_thinwalls, 2026-09-25) found the prediction-error agents exploiting exactly that:
by the end of training pred / lp kept the T jammed into or through a wall 22-28% of the time (the
random agent 0.2%) -- a chaotic glitch acting as a noisy TV. "safe" makes two changes that only act
when the T touches a wall: (1) each wall is a thick capsule (radius WALL_R) whose inner face is
where gym-pusht's is (x / y = 7 and 504) -- the T cannot tunnel; (2) while the agent touches the T
and the T touches a wall, the agent cannot move further along its push direction when that pushes the T
into the wall (its velocity component along the agent-to-T contact normal is removed): a pinned T is
not squeezed. Everywhere else the two are bit-identical; physics="gym" rebuilds gym-pusht exactly
(bit-identical to gym-pusht 0.1.6 over 20k steps incl. contacts).

State (7): agent x, y, agent vx, vy, T x, y (body origin = the bar's bottom-centre), T angle; px,
px/s, rad. The agent's velocity is part of the state: the 5-D gym-pusht state is not Markov
(the kinematic agent carries velocity between control steps). The T keeps a little hidden state
during contact: pymunk moves it with the velocity left from the previous substep before damping 0
zeroes it, so restarting a scene from s7 mid-push lands ~0.4 px off after one step (vs ~10 px of
true T motion; copying the T's velocity removes 97% of that). Outside contact s7 is exact.
Set a state angle-BEFORE-position: gym-pusht's `reset_to_state` does the opposite and lands the T
~60 px away (setting the angle rotates the body about its centre of gravity).

Action (2) in [-1, 1]: the PD target is agent + scale * a (stable-worldmodel's relative
convention, scale 100), clamped to the arena so the kinematic agent -- which walls do not stop --
stays inside. The action actually applied, (target - agent) / scale, is returned as
info["act_applied"].

Observation (8) = the state rescaled to O(1): (agent - 256) / 256, v / 500, (T - 256) / 256,
cos(angle), sin(angle).

`PushT` steps E scenes split over worker processes (pymunk is CPU-only); one tensor in, one out.
"""
from __future__ import annotations

import math
import multiprocessing as mp

import numpy as np
import pymunk
import torch

DT, CONTROL_HZ, K_P, K_V = 0.01, 10, 100.0, 20.0
N_SUB = int(1 / (DT * CONTROL_HZ))
AGENT_R = 15.0
ARENA = (20.0, 492.0)                         # PD-target clamp: the agent's centre stays inside the walls
WALL_LO, WALL_HI = 7.0, 504.0                 # inner faces of the wall segments (x/y = 5 or 506, radius 2)
WALL_R = 100.0                                # thick-wall capsule radius (inner faces unchanged)
FIXED_START = (100.0, 100.0, 0.0, 0.0, 256.0, 256.0, 0.0)
V_SCALE = 500.0
# T outline in body-local coordinates: bar [-60, 60] x [0, 30], stem [-15, 15] x [30, 120]
T_RECTS = np.array([[0.0, 15.0, 60.0, 15.0], [0.0, 75.0, 15.0, 45.0]])     # centre x, y, half-w, half-h
T_KEYPOINTS = np.array([(-60, 30), (60, 30), (60, 0), (-60, 0), (-15, 30), (-15, 120), (15, 120), (15, 30)],
                       dtype=np.float64)


# ------------------------------------------------------------------ geometry (numpy or torch)
def _lib(x):
    return torch if isinstance(x, torch.Tensor) else np


def keypoints(pose):
    """T corner positions (..., 8, 2) from a pose (..., 3) = T x, y, angle."""
    L = _lib(pose)
    kp = T_KEYPOINTS if L is np else torch.as_tensor(T_KEYPOINTS, dtype=pose.dtype, device=pose.device)
    c, s = L.cos(pose[..., 2:3]), L.sin(pose[..., 2:3])
    x = pose[..., 0:1] + c * kp[:, 0] - s * kp[:, 1]
    y = pose[..., 1:2] + s * kp[:, 0] + c * kp[:, 1]
    return L.stack([x, y], -1)


def to_t_frame(p, pose):
    """World points p (..., 2) in the T's body frame."""
    L = _lib(p)
    d0, d1 = p[..., 0] - pose[..., 0], p[..., 1] - pose[..., 1]
    c, s = L.cos(pose[..., 2]), L.sin(pose[..., 2])
    return L.stack([c * d0 + s * d1, -s * d0 + c * d1], -1)


def dist_to_t(p, pose):
    """Distance from world points p (..., 2) to the T (0 inside); a disk of radius r touches it iff < r."""
    L = _lib(p)
    q = to_t_frame(p, pose)
    d = []
    for cx, cy, hx, hy in T_RECTS:
        dx = (L.abs(q[..., 0] - cx) - hx).clip(0, None)
        dy = (L.abs(q[..., 1] - cy) - hy).clip(0, None)
        d.append(L.sqrt(dx * dx + dy * dy))
    return L.minimum(d[0], d[1])


def t_in_arena(pose, margin=0.0):
    """All T corners inside the walls' inner faces."""
    k = keypoints(pose)
    return ((k >= WALL_LO + margin) & (k <= WALL_HI - margin)).all(-1).all(-1)


def pd_matrices():
    """The kinematic agent's exact per-control-step map, per axis: [x', v'] = M [x, v] + N target."""
    A = np.array([[1 - K_P * DT * DT, DT * (1 - K_V * DT)], [-K_P * DT, 1 - K_V * DT]])
    B = np.array([K_P * DT * DT, K_P * DT])
    M, N = np.eye(2), np.zeros(2)
    for _ in range(N_SUB):
        M, N = A @ M, A @ N + B
    return M, N


def angle_diff(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


# ------------------------------------------------------------------ one scene
class Sim:
    """One Push-T scene. `agent_r` != 15 gives a wrong-geometry world (the calibration's geometry-sensitivity reference)."""

    def __init__(self, agent_r: float = AGENT_R, physics: str = "safe"):
        assert physics in ("safe", "gym")
        self.agent_r, self.physics = agent_r, physics
        self.set(np.array(FIXED_START))

    def _build(self):
        sp = pymunk.Space()
        sp.gravity = 0, 0
        sp.damping = 0.0
        if self.physics == "gym":
            sp.add(*[pymunk.Segment(sp.static_body, a, b, 2) for a, b in
                     (((5, 506), (5, 5)), ((5, 5), (506, 5)), ((506, 5), (506, 506)), ((5, 506), (506, 506)))])
        else:
            lo, hi = WALL_LO - WALL_R, WALL_HI + WALL_R           # capsule centre-lines; faces at 7 / 504
            sp.add(*[pymunk.Segment(sp.static_body, a, b, WALL_R) for a, b in
                     (((lo, hi), (lo, lo)), ((lo, lo), (hi, lo)), ((hi, lo), (hi, hi)), ((lo, hi), (hi, hi)))])
        agent = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        agent.position = (256, 400)
        self.agent_shape = pymunk.Circle(agent, self.agent_r)
        sp.add(agent, self.agent_shape)
        v1 = [(-60, 30), (60, 30), (60, 0), (-60, 0)]
        v2 = [(-15, 30), (-15, 120), (15, 120), (15, 30)]
        inertia = pymunk.moment_for_poly(1, vertices=v1)
        block = pymunk.Body(1, inertia + inertia)                # sic: gym-pusht uses the bar's moment twice
        s1, s2 = pymunk.Poly(block, v1), pymunk.Poly(block, v2)
        block.center_of_gravity = (s1.center_of_gravity + s2.center_of_gravity) / 2
        block.angle = 0
        block.position = (256, 300)
        sp.add(block, s1, s2)
        h = sp.add_collision_handler(0, 0)
        h.post_solve = self._on_contact
        self.space, self.agent, self.block = sp, agent, block
        self.flags = [False, False]                              # agent-T contact, T-wall contact
        self.n_agent, self.n_walls = None, []                    # last substep's contact normals (into the T)

    def _on_contact(self, arbiter, space, data):
        a, b = arbiter.shapes
        n = arbiter.contact_point_set.normal                     # points from shape a to shape b
        if a is self.agent_shape or b is self.agent_shape:
            self.flags[0] = True
            self.n_agent = n if a is self.agent_shape else -n    # agent -> T
        elif isinstance(a, pymunk.Segment) or isinstance(b, pymunk.Segment):
            self.flags[1] = True
            self.n_walls.append(n if isinstance(a, pymunk.Segment) else -n)   # wall -> T

    def set(self, s7):
        """Fresh scene at exactly s7 (angle before position); T velocity 0."""
        self._build()
        self.agent.position = (float(s7[0]), float(s7[1]))
        self.agent.velocity = (float(s7[2]), float(s7[3]))
        self.block.angle = float(s7[6])
        self.block.position = (float(s7[4]), float(s7[5]))
        self.space.reindex_shapes_for_body(self.block)

    def step(self, target):
        """One control step towards an absolute PD target. Returns (agent-T contact, T-wall contact)."""
        self.flags = [False, False]
        tx, ty = float(target[0]), float(target[1])
        ag = self.agent
        for _ in range(N_SUB):
            px, py = ag.position
            vx, vy = ag.velocity
            vx, vy = vx + (K_P * (tx - px) - K_V * vx) * DT, vy + (K_P * (ty - py) - K_V * vy) * DT
            na = self.n_agent
            if self.physics == "safe" and na is not None and any(na.dot(nw) < -0.3 for nw in self.n_walls):
                into = vx * na.x + vy * na.y                     # pushing a wall-pinned T into the wall
                if into > 0:
                    vx, vy = vx - into * na.x, vy - into * na.y
            ag.velocity = (vx, vy)
            self.n_agent, self.n_walls = None, []
            self.space.step(DT)
        return tuple(self.flags)

    def state(self):
        a, b = self.agent, self.block
        return np.array([a.position[0], a.position[1], a.velocity[0], a.velocity[1],
                         b.position[0], b.position[1], b.angle])


# ------------------------------------------------------------------ batched env
def _worker(conn, n, agent_r, physics):
    sims = [Sim(agent_r, physics) for _ in range(n)]
    while True:
        cmd, arg = conn.recv()
        if cmd == "reset":
            for s, st in zip(sims, arg):
                s.set(st)
            conn.send(np.stack([s.state() for s in sims]))
        elif cmd == "step":
            out = np.empty((n, 9))
            for i, (s, tg) in enumerate(zip(sims, arg)):
                out[i, 7:] = s.step(tg)
                out[i, :7] = s.state()
            conn.send(out)
        else:
            conn.close()
            return


T_COG = np.array([0.0, 45.0])                 # body-frame centre of gravity (gym-pusht averages the two parts)


def sample_starts(rng, n):
    """gym-pusht 0.1.6's reset distribution (pusht.py:276-284 + _set_state): agent ~ U[50, 450]^2,
    T placed at p ~ U[100, 400]^2 with angle 0 then rotated by theta ~ U(-pi, pi) about its centre of
    gravity -- so the CoG, not the origin, is uniform on [100, 400] + (0, 45). Continuous instead of
    integer draws. Deviations: starts with the agent overlapping the T (8% in gym-pusht) or the T
    through a wall (2.5%) are redrawn."""
    out = np.empty((n, 7))
    i = 0
    while i < n:
        p, th = rng.uniform(100, 400, 2), rng.uniform(-math.pi, math.pi)
        c, sn = math.cos(th), math.sin(th)
        origin = p + T_COG - np.array([c * T_COG[0] - sn * T_COG[1], sn * T_COG[0] + c * T_COG[1]])
        s = np.array([*rng.uniform(50, 450, 2), 0.0, 0.0, *origin, th])
        if dist_to_t(s[:2], s[4:7]) > AGENT_R and t_in_arena(s[4:7]):
            out[i] = s
            i += 1
    return out


class PushT:
    act_dim = 2
    obs_dim = 8
    # ground-truth read-outs in unit-square coordinates (agent / T-origin grids, as in point_push)
    agent_lo, agent_hi = ARENA[0] / 512, ARENA[1] / 512
    block_lo, block_hi = 0.0, 1.0

    def __init__(self, n_envs: int, device="cpu", ep_len: int = 200, start: str = "fixed",
                 action_scale: float = 100.0, workers: int = 12, seed: int = 0, tv: bool = False,
                 physics: str = "safe"):
        if tv:
            raise NotImplementedError("noisy TV is point_push-only for now")
        assert start in ("fixed", "random")
        self.E, self.device, self.ep_len, self.start, self.scale = n_envs, torch.device(device), ep_len, start, action_scale
        self.rng = np.random.default_rng(seed)
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        ctx = mp.get_context("fork")                             # workers never touch CUDA
        self.chunks = [c for c in np.array_split(np.arange(n_envs), min(workers, n_envs)) if len(c)]
        self.pipes, self.procs = [], []
        for c in self.chunks:
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, len(c), AGENT_R, physics), daemon=True)
            p.start()
            self.pipes.append(a)
            self.procs.append(p)
        self.s = np.tile(np.array(FIXED_START), (n_envs, 1))
        self.t = 0

    def close(self):
        for p in self.pipes:
            p.send(("close", None))
        for p in self.procs:
            p.join(timeout=5)

    def _scatter(self, cmd, arr):
        for p, c in zip(self.pipes, self.chunks):
            p.send((cmd, arr[c]))
        return np.concatenate([p.recv() for p in self.pipes])

    # ---------------------------------------------------------------- observation
    @staticmethod
    def obs_from_state(s):
        return torch.cat([(s[..., 0:2] - 256) / 256, s[..., 2:4] / V_SCALE, (s[..., 4:6] - 256) / 256,
                          torch.cos(s[..., 6:7]), torch.sin(s[..., 6:7])], -1)

    @staticmethod
    def state_from_obs(o):
        return torch.cat([o[..., 0:2] * 256 + 256, o[..., 2:4] * V_SCALE, o[..., 4:6] * 256 + 256,
                          torch.atan2(o[..., 7:8], o[..., 6:7])], -1)

    def state(self) -> torch.Tensor:
        return torch.as_tensor(self.s, dtype=torch.float32, device=self.device)

    def obs(self) -> torch.Tensor:
        return self.obs_from_state(self.state())

    # ground-truth read-out hooks used by run_curiosity.py / signals.py
    @staticmethod
    def agent_xy(st):
        return st[..., 0:2] / 512

    @staticmethod
    def block_xy(st):
        return st[..., 4:6] / 512

    n_count_cells = 10 ** 4 * 8

    def count_cell(self, st, bins=10, abins=8):
        ia = ((st[..., 0:2] - ARENA[0]) / (ARENA[1] - ARENA[0]) * bins).long().clamp(0, bins - 1)
        ib = (st[..., 4:6] / 512 * bins).long().clamp(0, bins - 1)
        th = torch.remainder(st[..., 6], 2 * math.pi) / (2 * math.pi)
        it = (th * abins).long().clamp(0, abins - 1)
        return (((ia[..., 0] * bins + ia[..., 1]) * bins + ib[..., 0]) * bins + ib[..., 1]) * abins + it

    # ---------------------------------------------------------------- dynamics
    def reset(self, random_start: bool | None = None) -> torch.Tensor:
        self.t = 0
        rnd = (self.start == "random") if random_start is None else random_start
        starts = sample_starts(self.rng, self.E) if rnd else np.tile(np.array(FIXED_START), (self.E, 1))
        self.s = self._scatter("reset", starts)
        return self.obs()

    def step(self, action: torch.Tensor):
        a = action.clamp(-1, 1).detach().to("cpu", torch.float64).numpy()
        s_old = self.s
        target = np.clip(s_old[:, :2] + self.scale * a, *ARENA)
        out = self._scatter("step", target)
        self.s, contact, t_wall = out[:, :7], out[:, 7] > 0, out[:, 8] > 0
        self.t += 1
        kp0, kp1 = keypoints(s_old[:, 4:7]), keypoints(self.s[:, 4:7])
        dev = self.device
        info = {"contact": torch.as_tensor(contact, device=dev),
                "block_move": torch.as_tensor(np.linalg.norm(kp1 - kp0, axis=-1).mean(-1) / 512,
                                              dtype=torch.float32, device=dev),
                "in_tv": torch.zeros(self.E, dtype=torch.bool, device=dev),
                "wall": torch.as_tensor(t_wall, device=dev),
                "act_applied": torch.as_tensor((target - s_old[:, :2]) / self.scale, dtype=torch.float32, device=dev)}
        return self.obs(), info
