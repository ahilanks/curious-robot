"""Paper evaluation protocol (App. C.1): dataset-defined start/goal pairs from
held-out episodes (>= 8000), goal = state h primitive steps ahead, budget 2h,
5-block open-loop plans replanned every 5 blocks, 50 episodes per eval seed."""
from __future__ import annotations

import numpy as np
import torch

import stable_worldmodel as swm
from stable_worldmodel.policy import BasePolicy, WorldModelPolicy, PlanConfig

from .wm import image_transform, fit_action_scaler


def sample_eval_tasks(ds, h, seed, n=50, first_heldout_ep=8000):
    ep_idx = ds.get_col_data('ep_idx').astype(np.int64)
    step_idx = ds.get_col_data('step_idx').astype(np.int64)
    lens = np.asarray(ds.lengths).astype(np.int64)
    valid = np.nonzero((ep_idx >= first_heldout_ep) & (step_idx <= lens[ep_idx] - h - 1))[0]
    rng = np.random.default_rng(seed)
    rows = np.sort(valid[rng.choice(len(valid), n, replace=False)])
    return ep_idx[rows].tolist(), step_idx[rows].tolist(), rows


class NoopPolicy(BasePolicy):
    """Executes zero (raw) actions — the paper's no-op floor."""

    def __init__(self, action_dim, **kw):
        super().__init__(**kw)
        self.action_dim = action_dim

    def get_action(self, info_dict, **kw):
        n = self.env.num_envs
        return np.zeros((n, self.action_dim), dtype=np.float32)


class ReplayPolicy(BasePolicy):
    """Oracle: replays the dataset's own actions from each start row (harness check)."""

    def __init__(self, ds, rows, action_dim, **kw):
        super().__init__(**kw)
        self.acts = ds.get_col_data('action')
        self.rows = np.asarray(rows)
        self.t = 0
        self.action_dim = action_dim

    def get_action(self, info_dict, **kw):
        a = self.acts[self.rows + self.t].astype(np.float32)
        a = np.nan_to_num(a, nan=0.0)
        self.t += 1
        return a


def make_world(cfg, n_envs, budget):
    world = swm.World(cfg['env'], num_envs=n_envs, image_shape=(224, 224), max_episode_steps=2 * budget, **cfg['env_kwargs'])
    if fix_render(world, cfg):
        world.reset(seed=0)  # apply the (dirty-marked) recompile now, so later per-env settings (e.g. Reacher tau) persist
    return world


def fix_render(world, cfg):
    """Match the evaluation renders to the released dataset frames.

    Reacher: the dataset was rendered with older MuJoCo texture semantics where the
    floor's checker texture shows as 2x2 cells; under MuJoCo 3.13 `texuniform="true"`
    maps a single cell over the floor (uniform colour) -- an out-of-distribution
    shift for the frozen encoder that costs every planner ~30-40 points.
    `texuniform="false"` reproduces the dataset frames to <0.4 mean abs pixel error.
    """
    changed = False
    if 'Reacher' in cfg['env']:
        for env in world.envs.envs:
            e = env.unwrapped
            mat = e._mjcf_model.find('material', 'grid')
            if mat is not None:
                mat.texuniform = 'false'
                if hasattr(e, 'mark_dirty'):
                    e.mark_dirty()
                changed = True
    return changed


def set_reacher_tau(world, tau):
    for e in world.envs.envs:
        e.unwrapped.env.task.qpos_threshold = tau


def reacher_taus(world):
    return [e.unwrapped.env.task.qpos_threshold for e in world.envs.envs]


def make_wm_policy(solver, scaler, horizon=5, action_block=5):
    tf = image_transform()
    return WorldModelPolicy(solver=solver,
                            config=PlanConfig(horizon=horizon, receding_horizon=horizon, history_len=1,
                                              action_block=action_block, warm_start=False),
                            process={'action': scaler}, transform={'pixels': tf, 'goal': tf})


def run_eval(world, policy, ds, eps, starts, h, callables, video=None):
    world.set_policy(policy)
    res = world.evaluate(dataset=ds, episodes_idx=eps, start_steps=starts, goal_offset=h, eval_budget=2 * h,
                         callables=callables, video=video)
    return res
