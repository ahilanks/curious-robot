"""Hand-designed baseline solvers from stable-worldmodel, with local fixes."""
from __future__ import annotations

import torch
from stable_worldmodel.solver import CEMSolver, MPPISolver, GradientSolver  # noqa: F401


class GradientSolverFixed(GradientSolver):
    """swm 0.1.1 bug: init_action() only moves a *padded* init plan to the solver
    device; a full-horizon CPU init (what WorldModelPolicy passes with
    warm_start=False) then collides with the CUDA noise tensor."""

    def init_action(self, n_envs, actions=None):
        if actions is not None:
            actions = actions.to(self.device, self.dtype)
        return super().init_action(n_envs, actions)
