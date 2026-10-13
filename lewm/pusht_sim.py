"""Exact-state Push-T simulation for CEM-in-the-simulator (from-scratch learning-signal arms).

stable-worldmodel's swm/PushT-v1 is used unchanged (physics, rendering, 224 px frames, relative actions x100 px,
10 Hz control / 10 x 0.01 s substeps). Two additions, both collection-only (evaluation runs the stock env):

  * clip_action: the commanded PD target agent_pos + 100 a is clipped to the arena [LO, HI]^2 (the inner wall faces
    at 7 / 504 px minus the 15 px agent radius). The stock agent is a KINEMATIC body that passes through the walls,
    and uniform-random actions leave it off-screen on 49% of steps. The PD loop is critically damped (k_v = 2 sqrt k_p),
    so a target inside the arena keeps the agent inside. The executed (clipped) action is what gets stored.
  * get_state / set_state: the full rigid-body state (agent pose + velocity, block pose + linear/angular velocity),
    set without stepping physics (the env's own _set_state steps one 0.01 s substep and drops the block velocity).
    Every planning segment, real or candidate, starts with set_state from the same vector, so the trajectory a
    candidate was scored on is bit-identical to the one executed (check_clone_exact).

A WorkerPool of forked processes (each with its own env) runs candidate rollouts and writes rendered frames into
shared-memory buffers; fork it before CUDA is initialised in the parent.
"""
from __future__ import annotations

import multiprocessing as mp
import os

import numpy as np

LO, HI = 22.0, 489.0
STATE_DIM = 10  # agent x, y, vx, vy | block x, y, angle, vx, vy, omega
FRAME = (224, 224, 3)


def single_thread():
    """One compute thread per process: forked env workers (and any helper process sharing the box with a training
    run) otherwise each spin up cv2 / torch / OpenMP pools sized to all 128 cores."""
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"
    import cv2
    cv2.setNumThreads(1)
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass


def make_env():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401  (registers swm/ envs)

    env = gym.make("swm/PushT-v1", disable_env_checker=True).unwrapped
    env.reset(seed=0)  # builds the pymunk space / bodies; every rollout then starts from set_state
    return env


def get_state(env) -> np.ndarray:
    a, b = env.agent, env.block
    return np.array([a.position.x, a.position.y, a.velocity.x, a.velocity.y,
                     b.position.x, b.position.y, b.angle, b.velocity.x, b.velocity.y, b.angular_velocity],
                    dtype=np.float64)


def set_state(env, s: np.ndarray) -> None:
    # Rebuild the pymunk space first: Chipmunk keeps contact arbiters (with warm-start impulses) alive for
    # collision_persistence = 3 steps, so re-posing bodies inside a used space is NOT a clean restore - two
    # rollouts from the same vector diverged by hundreds of px in testing. _setup() is the env's own builder
    # (walls, agent, block, collision handler; damping 0, as at reset).
    # The bodies are then removed and re-inserted at the target pose: _setup() inserts them at the instance's last
    # sampled reset pose, and the dynamic bounding-box tree built from that history decides the contact-solving
    # order, so two env instances given the same vector diverged (same-instance repeats were already exact).
    env._setup()
    a, b = env.agent, env.block
    env.space.remove(a, *a.shapes)
    env.space.remove(b, *b.shapes)
    a.position = (float(s[0]), float(s[1]))
    a.velocity = (float(s[2]), float(s[3]))
    # angle BEFORE position: Chipmunk rotates a body about its centre of gravity, and the T's CoG is not its
    # origin, so setting the angle second moves the origin by an amount that depends on the previous angle
    # (the env's own _set_state uses this order too).
    b.angle = float(s[6])
    b.position = (float(s[4]), float(s[5]))
    b.velocity = (float(s[7]), float(s[8]))
    b.angular_velocity = float(s[9])
    env.space.add(a, *a.shapes)
    env.space.add(b, *b.shapes)


def clip_action(env, a: np.ndarray) -> np.ndarray:
    a = np.clip(np.asarray(a, dtype=np.float64), -1.0, 1.0)
    pos = np.array([env.agent.position.x, env.agent.position.y])
    tgt = np.clip(pos + a * env.action_scale, LO, HI)
    return ((tgt - pos) / env.action_scale).astype(np.float32)


def rollout(env, s0: np.ndarray, actions: np.ndarray, render_every: int, frames_out: np.ndarray | None):
    """Run `actions` (T, 2) from state s0; render after every `render_every` steps into frames_out (T//k, H, W, 3).

    Returns (executed clipped actions (T, 2), per-step states (T, STATE_DIM), per-step contact counts (T,))."""
    set_state(env, s0)
    T = len(actions)
    exe = np.empty((T, 2), np.float32)
    states = np.empty((T, STATE_DIM), np.float64)
    contacts = np.empty(T, np.int32)
    k = 0
    for t in range(T):
        a2 = clip_action(env, actions[t])
        _, _, _, _, info = env.step(a2)
        exe[t] = a2
        states[t] = get_state(env)
        contacts[t] = info["n_contacts"]
        if frames_out is not None and (t + 1) % render_every == 0:
            frames_out[k] = env.render()
            k += 1
    return exe, states, contacts


def reset_state(env, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Stock reset (random agent / block position and block angle), returned as (state, first frame)."""
    env.reset(seed=int(seed))
    return get_state(env), env.render()


# ---------------------------------------------------------------- worker pool (fork; shared-memory frame buffers)
_G: dict = {}


def _shared_u8(shape) -> np.ndarray:
    raw = mp.RawArray("B", int(np.prod(shape)))
    return np.frombuffer(raw, dtype=np.uint8).reshape(shape)


def _worker_init():
    single_thread()
    _G["env"] = make_env()


def _worker_rollouts(job):
    """job = (buf_name, idx (n,), states (n, STATE_DIM), actions (n, T, 2), render_every)."""
    buf_name, idx, states, actions, render_every = job
    env, buf = _G["env"], _G[buf_name]
    T = actions.shape[1]
    exe = np.empty((len(idx), T, 2), np.float32)
    st = np.empty((len(idx), T, STATE_DIM), np.float64)
    ct = np.empty((len(idx), T), np.int32)
    for j, i in enumerate(idx):
        exe[j], st[j], ct[j] = rollout(env, states[j], actions[j], render_every, buf[i])
    return idx, exe, st, ct


def _worker_reset(seeds):
    env = _G["env"]
    out = []
    for s in seeds:
        st, fr = reset_state(env, s)
        out.append((st, fr))
    return out


class WorkerPool:
    """Forked env workers. Buffers: 'cand' (max_cands, n_render, H, W, 3), 'exec' (n_envs, T, H, W, 3)."""

    def __init__(self, n_workers: int, max_cands: int, n_render: int, n_envs: int, T: int):
        _G["cand"] = _shared_u8((max_cands, n_render) + FRAME)
        _G["exec"] = _shared_u8((n_envs, T) + FRAME)
        self.cand, self.exec = _G["cand"], _G["exec"]
        self.n_workers = n_workers
        self.pool = mp.get_context("fork").Pool(n_workers, initializer=_worker_init)

    def run(self, buf_name: str, states: np.ndarray, actions: np.ndarray, render_every: int, chunk: int = 0):
        n = len(states)
        chunk = chunk or max(1, -(-n // (self.n_workers * 2)))
        jobs = [(buf_name, np.arange(i, min(n, i + chunk)), states[i:i + chunk], actions[i:i + chunk], render_every)
                for i in range(0, n, chunk)]
        T = actions.shape[1]
        exe = np.empty((n, T, 2), np.float32)
        st = np.empty((n, T, STATE_DIM), np.float64)
        ct = np.empty((n, T), np.int32)
        for idx, e, s, c in self.pool.imap_unordered(_worker_rollouts, jobs):
            exe[idx], st[idx], ct[idx] = e, s, c
        return exe, st, ct

    def reset(self, seeds) -> tuple[np.ndarray, np.ndarray]:
        out = self.pool.map(_worker_reset, [[s] for s in seeds])
        states = np.stack([o[0][0] for o in out])
        frames = np.stack([o[0][1] for o in out])
        return states, frames

    def close(self):
        self.pool.terminate()


def check_clone_exact(n_trials: int = 20, T: int = 100, seg: int = 25, seed: int = 0) -> dict:
    """(1) Determinism - two rollouts from the same set_state vector must match bit for bit (states AND rendered
    frames); this is what makes the scored candidate identical to the executed one. (2) Segmenting cost
    (informational) - a T-step continuous simulation vs the same actions run in `seg`-step segments, each
    restarted with set_state (space rebuilt, contact warm-start dropped): max block position / angle deviation."""
    rng = np.random.default_rng(seed)
    env, env2 = make_env(), make_env()
    det_state, det_frames, contact_steps = 0.0, 0, 0
    seg_pos, seg_ang = [], []
    for k in range(n_trials):
        s0, _ = reset_state(env, 1000 + k)
        acts = np.clip(rng.normal(0, 0.6, (T, 2)), -1, 1)
        acts[: T // 4] = np.clip((s0[4:6] - s0[0:2]) / 100.0, -1, 1)   # walk into the block: contacts
        fa = np.zeros((T // 5,) + FRAME, np.uint8)
        fb = np.zeros_like(fa)
        _, st_a, ct = rollout(env, s0, acts, 5, fa)
        _, st_b, _ = rollout(env2, s0, acts, 5, fb)
        det_state = max(det_state, float(np.abs(st_a - st_b).max()))
        det_frames += int((fa != fb).any())
        contact_steps += int((ct > 0).sum())
        # continuous (no re-set) vs segmented
        set_state(env, s0)
        st_c = np.empty((T, STATE_DIM))
        for t in range(T):
            env.step(clip_action(env, acts[t]))
            st_c[t] = get_state(env)
        s, st_s = s0, []
        for i in range(0, T, seg):
            _, st_i, _ = rollout(env2, s, acts[i:i + seg], 10**9, None)
            st_s.append(st_i)
            s = st_i[-1]
        st_s = np.concatenate(st_s)
        seg_pos.append(float(np.linalg.norm(st_c[:, 4:6] - st_s[:, 4:6], axis=1).max()))
        seg_ang.append(float(np.abs(st_c[:, 6] - st_s[:, 6]).max()))
    return {"determinism_max_abs_state_diff": det_state, "determinism_frame_mismatches": det_frames,
            "contact_steps": contact_steps, "segmenting_block_pos_dev_px_median": float(np.median(seg_pos)),
            "segmenting_block_pos_dev_px_max": float(np.max(seg_pos)),
            "segmenting_block_ang_dev_rad_max": float(np.max(seg_ang))}
