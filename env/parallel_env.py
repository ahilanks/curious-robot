"""Vectorised SO-ARM101 envs with two interchangeable backends.

The training-loop API is identical across backends:
    reset()           -> stacked obs {"image", "proprio"}
    step(actions)     -> (stacked obs, stacked info)
    render_overhead() -> stacked overhead frames

  * VectorMujocoEnv        -- envs live in this process (sequential, or threaded
                              via `threads`). Simple, but its EGL render contexts
                              share the process with the trainer's CUDA work.
  * SubprocVectorMujocoEnv -- each env runs in its own `spawn` worker process.
  * SubprocSingleEnv       -- one env in a worker, for the eval/video rollout.

Why the subprocess backends exist: MuJoCo's GPU EGL renderer and PyTorch CUDA cannot
share a GPU -- accumulating CUDA work aborts a later render() (SIGABRT), even from a
separate process on this driver. So render workers run on the CPU (OSMesa) in their
own processes, parallelising across cores while the GPU stays dedicated to training.
"""
from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

from .mujoco_env import MujocoSO101Env, DEFAULT_SCENE

_INFO_KEYS = ("applied_torque", "qvel", "qvel_prev", "qpos", "safety_reward",
              "object_contacts", "table_contacts", "object_motion")


class VectorMujocoEnv:
    def __init__(
        self,
        n_envs: int = 8,
        scene_path: str | Path = DEFAULT_SCENE,
        wrist_resolution: int = 224,
        overhead_resolution: int = 256,
        frame_skip: int = 6,
        action_max: float = 0.3,
        dq_max: float = 100.0,
        safety_delta: float = 0.05,
        seed: int = 0,
        threads: int = 0,                # 0 = sequential
    ):
        self.n_envs = n_envs
        self.envs = [
            MujocoSO101Env(
                scene_path=scene_path,
                wrist_resolution=wrist_resolution,
                overhead_resolution=overhead_resolution,
                frame_skip=frame_skip,
                action_max=action_max,
                dq_max=dq_max,
                safety_delta=safety_delta,
                seed=seed + i,
            )
            for i in range(n_envs)
        ]
        self.n_dof = self.envs[0].n_dof
        self.tau_max = self.envs[0].tau_max
        self.wrist_resolution = wrist_resolution
        self.overhead_resolution = overhead_resolution
        self._executor = ThreadPoolExecutor(max_workers=threads) if threads > 0 else None

    def _map(self, fn, *iters):
        if self._executor is None:
            return [fn(*x) for x in zip(*iters)] if iters else [fn(e) for e in self.envs]
        return (list(self._executor.map(lambda args: fn(*args), zip(*iters))) if iters
                else list(self._executor.map(fn, self.envs)))

    def reset(self) -> dict[str, np.ndarray]:
        return self._stack_obs(self._map(lambda e: e.reset()))

    def reset_one(self, idx: int) -> dict[str, np.ndarray]:
        return self.envs[idx].reset()

    def step(self, actions: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
        actions = np.asarray(actions, dtype=np.float32)
        assert actions.shape == (self.n_envs, self.n_dof)
        outs = self._map(lambda env, a: env.step(a), self.envs, list(actions))
        obs_list, info_list = zip(*outs)
        info = {k: np.stack([d[k] for d in info_list]) for k in _INFO_KEYS}
        return self._stack_obs(obs_list), info

    @staticmethod
    def _stack_obs(obs_list):
        return {
            "image": np.stack([o["image"] for o in obs_list]),
            "proprio": np.stack([o["proprio"] for o in obs_list]),
        }

    def render_overhead(self) -> np.ndarray:
        return np.stack(self._map(lambda e: e.render_overhead()))

    def render_overhead_one(self, idx: int) -> np.ndarray:
        """Overhead frame for a single env (for cheap training-run snapshots)."""
        return self.envs[idx].render_overhead()

    def close(self) -> None:
        for env in self.envs:
            env.close()
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None


# --------------------------------------------------------------- subprocess backend
def _env_worker(remote, parent_remote, env_kwargs):
    """Run one MujocoSO101Env in a CUDA-free `spawn` worker, isolating its EGL render
    context from the trainer's CUDA context. Serves commands over `remote`."""
    parent_remote.close()
    os.environ.setdefault("MUJOCO_GL", "osmesa")       # CPU render; GPU EGL fights CUDA
    env = MujocoSO101Env(**env_kwargs)
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                remote.send(env.step(data))
            elif cmd == "reset":
                remote.send(env.reset())
            elif cmd == "render_overhead":
                remote.send(env.render_overhead())
            elif cmd == "spaces":
                remote.send((env.n_dof, env.tau_max))
            elif cmd == "close":
                break
            else:
                raise RuntimeError(f"unknown env-worker command {cmd!r}")
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        env.close()
        remote.close()


@contextlib.contextmanager
def _render_worker_spawn_env():
    """Env that spawned render workers inherit: OSMesa CPU rendering (GPU EGL aborts
    when it shares a GPU with the trainer's CUDA, even across processes on this
    driver) and no visible CUDA device (so torch in a worker can't grab the GPU)."""
    saved = {k: os.environ.get(k) for k in ("MUJOCO_GL", "CUDA_VISIBLE_DEVICES")}
    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class _EnvWorker:
    """Parent-side handle for one `_env_worker` process and its pipe."""

    def __init__(self, ctx, env_kwargs):
        self.remote, work_remote = ctx.Pipe()
        self.proc = ctx.Process(target=_env_worker,
                                args=(work_remote, self.remote, env_kwargs),
                                daemon=True)
        self.proc.start()
        work_remote.close()                            # the worker keeps the only copy

    def send(self, cmd, data=None):
        self.remote.send((cmd, data))

    def recv(self):
        return self.remote.recv()

    def close(self):
        try:
            self.remote.send(("close", None))
        except (BrokenPipeError, OSError):
            pass
        self.proc.join(timeout=5)
        if self.proc.is_alive():
            self.proc.terminate()
        with contextlib.suppress(OSError):
            self.remote.close()


class SubprocVectorMujocoEnv:
    """Drop-in for VectorMujocoEnv with each env in its own worker process."""

    def __init__(self, n_envs: int = 8, scene_path: str | Path = DEFAULT_SCENE,
                 wrist_resolution: int = 224, overhead_resolution: int = 256,
                 frame_skip: int = 6, action_max: float = 0.3, dq_max: float = 100.0,
                 safety_delta: float = 0.05, seed: int = 0, threads: int = 0):
        self.n_envs = n_envs                           # `threads` taken for API parity
        self.wrist_resolution = wrist_resolution
        self.overhead_resolution = overhead_resolution
        base = dict(scene_path=str(scene_path), wrist_resolution=wrist_resolution,
                    overhead_resolution=overhead_resolution, frame_skip=frame_skip,
                    action_max=action_max, dq_max=dq_max, safety_delta=safety_delta)
        ctx = mp.get_context("spawn")                  # fresh procs => no inherited CUDA
        with _render_worker_spawn_env():
            self._workers = [_EnvWorker(ctx, dict(base, seed=seed + i))
                             for i in range(n_envs)]
        self._workers[0].send("spaces")
        self.n_dof, self.tau_max = self._workers[0].recv()

    def reset(self) -> dict[str, np.ndarray]:
        for w in self._workers:
            w.send("reset")
        return VectorMujocoEnv._stack_obs([w.recv() for w in self._workers])

    def reset_one(self, idx: int) -> dict[str, np.ndarray]:
        self._workers[idx].send("reset")
        return self._workers[idx].recv()

    def step(self, actions: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
        actions = np.asarray(actions, dtype=np.float32)
        assert actions.shape == (self.n_envs, self.n_dof)
        for w, a in zip(self._workers, actions):
            w.send("step", a)
        obs_list, info_list = zip(*(w.recv() for w in self._workers))
        info = {k: np.stack([d[k] for d in info_list]) for k in _INFO_KEYS}
        return VectorMujocoEnv._stack_obs(obs_list), info

    def render_overhead(self) -> np.ndarray:
        for w in self._workers:
            w.send("render_overhead")
        return np.stack([w.recv() for w in self._workers])

    def render_overhead_one(self, idx: int) -> np.ndarray:
        """Overhead frame for a single env (for cheap training-run snapshots)."""
        self._workers[idx].send("render_overhead")
        return self._workers[idx].recv()

    def close(self) -> None:
        for w in self._workers:
            w.close()


class SubprocSingleEnv:
    """One MujocoSO101Env in a worker process, mirroring the single-env API that
    record_rollout() uses, so the eval/video render stays out of the CUDA process."""

    def __init__(self, scene_path: str | Path = DEFAULT_SCENE,
                 wrist_resolution: int = 224, overhead_resolution: int = 256,
                 frame_skip: int = 6, action_max: float = 0.3, dq_max: float = 100.0,
                 safety_delta: float = 0.05, seed: int = 0):
        self.wrist_resolution = wrist_resolution
        self.overhead_resolution = overhead_resolution
        kwargs = dict(scene_path=str(scene_path), wrist_resolution=wrist_resolution,
                      overhead_resolution=overhead_resolution, frame_skip=frame_skip,
                      action_max=action_max, dq_max=dq_max, safety_delta=safety_delta,
                      seed=seed)
        ctx = mp.get_context("spawn")
        with _render_worker_spawn_env():
            self._worker = _EnvWorker(ctx, kwargs)
        self._worker.send("spaces")
        self.n_dof, self.tau_max = self._worker.recv()

    def reset(self) -> dict[str, np.ndarray]:
        self._worker.send("reset")
        return self._worker.recv()

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
        self._worker.send("step", np.asarray(action, dtype=np.float32))
        return self._worker.recv()

    def render_overhead(self) -> np.ndarray:
        self._worker.send("render_overhead")
        return self._worker.recv()

    def close(self) -> None:
        self._worker.close()
