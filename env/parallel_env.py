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

Why the subprocess backends exist: the render workers offscreen-render on the GPU via EGL
(~0.3ms vs OSMesa's ~35ms per 224^2 frame). EGL and the trainer's CUDA context coexist fine
on one GPU -- they use different driver subsystems -- but only when they live in SEPARATE
processes: a render() in the CUDA process itself can SIGABRT once CUDA work accumulates. So
each env runs in its own CUDA-free `spawn` worker (CUDA_VISIBLE_DEVICES=''), rendering on the
GPU while the main process owns the CUDA context. `render_backend="osmesa"` falls back to CPU
offscreen rendering (no GPU / EGL unavailable). See `_render_worker_spawn_env` and
`_egl_first_init_lock` for the two halves of making the worker EGL init deterministic.
"""
from __future__ import annotations

import contextlib
import fcntl
import glob
import multiprocessing as mp
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

from .mujoco_env import MujocoSO101Env, DEFAULT_SCENE

_INFO_KEYS = ("applied_torque", "qvel", "qvel_prev", "qpos", "safety_reward",
              "object_contacts", "table_contacts", "object_motion", "ee_pos")


class VectorMujocoEnv:
    def __init__(
        self,
        n_envs: int = 8,
        scene_path: str | Path = DEFAULT_SCENE,
        wrist_resolution: int = 224,
        overhead_resolution: int = 256,
        encode_cam: str = "wrist",       # which camera fills obs["image"] (WM input); "overhead" = fixed view
        frame_skip: int = 6,
        action_max: float = 0.3,
        safety_delta: float = 9.0,
        seed: int = 0,
        threads: int = 0,                # 0 = sequential
        fixed_objects: bool = False,
    ):
        self.n_envs = n_envs
        self.envs = [
            MujocoSO101Env(
                scene_path=scene_path,
                wrist_resolution=wrist_resolution,
                overhead_resolution=overhead_resolution,
                encode_cam=encode_cam,
                frame_skip=frame_skip,
                action_max=action_max,
                safety_delta=safety_delta,
                seed=seed + i,
                fixed_objects=fixed_objects,
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

    def step_block_async(self, action_blocks: np.ndarray) -> None:
        action_blocks = np.asarray(action_blocks, dtype=np.float32)

        def run(env, ab):
            obs, infos = None, []
            last = len(ab) - 1
            for k, a_k in enumerate(ab):
                obs, info = env.step(a_k, render=(k == last))      # render only the kept (final) obs
                infos.append(info)
            return obs, infos
        self._pending = self._map(run, self.envs, list(action_blocks))

    def step_block_wait(self):
        results, self._pending = self._pending, None
        return (self._stack_obs([r[0] for r in results]),
                self._stack_sub_infos([r[1] for r in results]))

    @staticmethod
    def _stack_obs(obs_list):
        return {
            "image": np.stack([o["image"] for o in obs_list]),
            "proprio": np.stack([o["proprio"] for o in obs_list]),
        }

    @staticmethod
    def _stack_sub_infos(infos_per_env):
        """infos_per_env: list over envs of list over substeps of info-dicts ->
        list over substeps of {key: (n_envs, ...)} (so the loop accumulates per substep)."""
        n_sub = len(infos_per_env[0])
        return [{k: np.stack([infos_per_env[e][s][k] for e in range(len(infos_per_env))])
                 for k in _INFO_KEYS} for s in range(n_sub)]

    def render_overhead(self) -> np.ndarray:
        return np.stack(self._map(lambda e: e.render_overhead()))

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
    import sys as _sys
    gl = os.environ.get("MUJOCO_GL", "egl")            # inherited from _render_worker_spawn_env
    os.environ["MUJOCO_GL"] = gl
    os.environ["PYOPENGL_PLATFORM"] = gl               # match MUJOCO_GL (else inherited osmesa breaks egl)
    # The spawn BOOTSTRAP (child imports the trainer's __main__) may import mujoco under the parent's
    # osmesa platform -> mujoco.Renderer not attached. If so, purge + re-import the stack under `gl`.
    if "mujoco" in _sys.modules and not hasattr(_sys.modules["mujoco"], "Renderer"):
        for _m in [k for k in list(_sys.modules)
                   if k == "mujoco" or k.startswith("mujoco.") or k == "env.mujoco_env"]:
            _sys.modules.pop(_m, None)
    from env.mujoco_env import MujocoSO101Env as _Env
    with _egl_first_init_lock(gl):                      # serialise the workers' first EGL context creation
        env = _Env(**env_kwargs)                        # (builds both Renderers -> the racy eglInitialize)
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                remote.send(env.step(data))
            elif cmd == "step_block":          # run a whole action_block in-worker: 1 IPC round-trip / decision
                obs, infos = None, []
                last = len(data) - 1
                for k, a_k in enumerate(data): # data: (action_block, n_dof)
                    obs, info = env.step(a_k, render=(k == last))  # render only the kept (final) obs
                    infos.append(info)
                remote.send((obs, infos))      # final obs + per-substep info list
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


def _nvidia_egl_icd() -> str | None:
    """Path to the NVIDIA EGL vendor ICD (glvnd config), or None if not present. Forcing this as
    the sole __EGL_VENDOR_LIBRARY_FILENAMES makes eglQueryDevicesEXT enumerate ONLY the GPU(s):
    no Mesa software (llvmpipe) device for a worker to land on by accident, and no multi-ICD
    probing across nvidia+mesa on every worker (a per-process, driver-shared step that races)."""
    for d in ("/usr/share/glvnd/egl_vendor.d", "/etc/glvnd/egl_vendor.d"):
        hits = sorted(glob.glob(os.path.join(d, "*nvidia*.json")))
        if hits:
            return hits[0]
    return None


@contextlib.contextmanager
def _egl_first_init_lock(gl: str):
    """Serialise the FIRST EGL context creation across the spawn workers. N workers calling
    eglInitialize on the same GPU within a few ms can race the NVIDIA driver's global device-init
    path -> intermittent 'Cannot initialize a EGL device display'. An advisory file lock warms the
    contexts up one worker at a time; once a context exists, rendering is lock-free, so steady-state
    SPS is unaffected. No-op for osmesa (CPU, no driver-side device init to serialise)."""
    if gl != "egl":
        yield
        return
    f = open(os.path.join(tempfile.gettempdir(), "mujoco_egl_init.lock"), "w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(f, fcntl.LOCK_UN)
        f.close()


@contextlib.contextmanager
def _render_worker_spawn_env(gl: str = "egl"):
    """Env that spawned render workers inherit: `gl` render backend and NO visible CUDA device
    (so torch in a worker can't grab the GPU). `gl="egl"` = GPU offscreen rendering -- ~100x faster
    than osmesa (0.3ms vs 35ms per 224^2 frame) and VALIDATED to coexist with the trainer's CUDA
    context across processes on this driver (the old "EGL aborts next to CUDA" caveat does not hold
    here, because the worker is CUDA-free). `gl="osmesa"` = CPU fallback (no GPU / EGL unavailable)."""
    keys = ("MUJOCO_GL", "PYOPENGL_PLATFORM", "CUDA_VISIBLE_DEVICES",
            "MUJOCO_EGL_DEVICE_ID", "__EGL_VENDOR_LIBRARY_FILENAMES")
    saved = {k: os.environ.get(k) for k in keys}
    os.environ["MUJOCO_GL"] = gl
    # CRITICAL: the trainer's main proc imports mujoco under osmesa, which sets PYOPENGL_PLATFORM=osmesa
    # in the (inherited) environment. Children must override it to match `gl`, or PyOpenGL stays locked
    # to osmesa and `import mujoco` under egl raises "Cannot use EGL rendering platform".
    os.environ["PYOPENGL_PLATFORM"] = gl
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if gl == "egl":
        # Make GPU selection deterministic (the other half of "deterministic worker EGL init"; the
        # first-init race is handled by _egl_first_init_lock). Pin the physical GPU index so MuJoCo
        # uses eglGetPlatformDisplayEXT(devices[id]) instead of looping the (non-deterministic across
        # processes) eglQueryDevicesEXT order -- which on a box with 2 NVIDIA + 1 Mesa-software EGL
        # device can silently render a worker on the llvmpipe CPU device. The device id is a GLOBAL
        # physical index, unaffected by CUDA_VISIBLE_DEVICES; 0 is right on a single-GPU pod (override
        # via MUJOCO_EGL_DEVICE_ID for multi-GPU). Force the NVIDIA ICD so only the GPU is enumerable.
        os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
        icd = _nvidia_egl_icd()
        if icd and "__EGL_VENDOR_LIBRARY_FILENAMES" not in os.environ:
            os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = icd
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
                 encode_cam: str = "wrist", render_backend: str = "egl",
                 frame_skip: int = 6, action_max: float = 0.3,
                 safety_delta: float = 9.0, seed: int = 0, threads: int = 0,
                 fixed_objects: bool = False):
        self.n_envs = n_envs                           # `threads` taken for API parity
        self.wrist_resolution = wrist_resolution
        self.overhead_resolution = overhead_resolution
        base = dict(scene_path=str(scene_path), wrist_resolution=wrist_resolution,
                    overhead_resolution=overhead_resolution, encode_cam=encode_cam,
                    frame_skip=frame_skip,
                    action_max=action_max, safety_delta=safety_delta,
                    fixed_objects=fixed_objects)
        ctx = mp.get_context("spawn")                  # fresh procs => no inherited CUDA
        with _render_worker_spawn_env(render_backend):
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

    def step_block_async(self, action_blocks: np.ndarray) -> None:
        """Dispatch a full action_block per env (non-blocking). Workers run all
        action_block substeps while the trainer does GPU work; collect with
        step_block_wait(). This overlaps CPU rollout with GPU learning."""
        action_blocks = np.asarray(action_blocks, dtype=np.float32)
        assert action_blocks.shape[0] == self.n_envs and action_blocks.ndim == 3
        for w, ab in zip(self._workers, action_blocks):
            w.send("step_block", ab)

    def step_block_wait(self):
        """Collect step_block_async(): final stacked obs + a list (over the action_block)
        of per-substep infos, each {key: (n_envs, ...)}."""
        results = [w.recv() for w in self._workers]            # each: (final_obs, [info_0..info_{B-1}])
        obs = VectorMujocoEnv._stack_obs([r[0] for r in results])
        return obs, VectorMujocoEnv._stack_sub_infos([r[1] for r in results])

    def render_overhead(self) -> np.ndarray:
        for w in self._workers:
            w.send("render_overhead")
        return np.stack([w.recv() for w in self._workers])

    def close(self) -> None:
        for w in self._workers:
            w.close()


class SubprocSingleEnv:
    """One MujocoSO101Env in a worker process, mirroring the single-env API that
    record_rollout() uses, so the eval/video render stays out of the CUDA process."""

    def __init__(self, scene_path: str | Path = DEFAULT_SCENE,
                 wrist_resolution: int = 224, overhead_resolution: int = 256,
                 frame_skip: int = 6, action_max: float = 0.3,
                 safety_delta: float = 9.0, seed: int = 0):
        self.wrist_resolution = wrist_resolution
        self.overhead_resolution = overhead_resolution
        kwargs = dict(scene_path=str(scene_path), wrist_resolution=wrist_resolution,
                      overhead_resolution=overhead_resolution, frame_skip=frame_skip,
                      action_max=action_max, safety_delta=safety_delta,
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
