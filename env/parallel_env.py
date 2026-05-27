"""In-process vectorised SO-ARM101 envs. Sequential (or threaded) per-env step.

The training-loop API is identical to a single env: reset() -> stacked obs,
step(actions) -> (stacked obs, stacked info). Swap in a multiprocessing backend
later without touching the trainer.
"""
from __future__ import annotations

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

    def close(self) -> None:
        for env in self.envs:
            env.close()
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
