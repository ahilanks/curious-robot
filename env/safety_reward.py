"""Safety reward from servo data on real env transitions (README §Rewards).

    r_safe_t = -sum_i (|tau_i| / tau_max_i) * max(0, -tau_i * qddot_i - delta)
    qddot_i  = (qdot_i_t - qdot_i_{t-dt_safe}) / dt_safe

Penalises the arm for driving a joint *against* its own acceleration (a motor
fighting the link's momentum): the term -tau*qddot is positive only when torque
and acceleration oppose, and the hinge keeps it active past a deadband `delta`.
This pushes the policy toward smooth, compliant motion. The acceleration is only
used here; nothing else in the codebase consumes it.

NOTE (vs. le-wm/dreamer): the hinge is LINEAR here, per the README, not squared.
"""
from __future__ import annotations

import numpy as np
import torch


def safety_reward_np(
    applied_torque: np.ndarray,   # u^app_t, shape (..., n_dof)
    qvel: np.ndarray,             # qdot_t
    qvel_prev: np.ndarray,        # qdot_{t - dt_safe}
    tau_max: np.ndarray,          # per-joint torque limit, (n_dof,)
    dt_safe: float = 0.030,       # accel finite-diff window (README: Delta t_safe)
    delta: float = 0.05,          # safety deadband (README: delta)
) -> np.ndarray:
    """Vectorised numpy version. Inputs broadcast over the trailing dim n_dof."""
    tau_max = np.asarray(tau_max, dtype=np.float32)
    accel = (qvel - qvel_prev) / max(float(dt_safe), 1e-8)
    weight = np.abs(applied_torque) / np.maximum(tau_max, 1e-6)
    fight = np.maximum(0.0, -applied_torque * accel - delta)
    return -(weight * fight).sum(axis=-1).astype(np.float32)


def safety_reward_torch(
    applied_torque: torch.Tensor,
    qvel: torch.Tensor,
    qvel_prev: torch.Tensor,
    tau_max: torch.Tensor,
    dt_safe: float = 0.030,
    delta: float = 0.05,
) -> torch.Tensor:
    accel = (qvel - qvel_prev) / max(float(dt_safe), 1e-8)
    weight = applied_torque.abs() / tau_max.clamp_min(1e-6)
    fight = torch.clamp(-applied_torque * accel - delta, min=0.0)
    return -(weight * fight).sum(dim=-1)
