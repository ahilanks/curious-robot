"""Map a policy action to MuJoCo SO-ARM101 actuator commands (README §Actor + actuation).

README control law:
    a_t      = tanh(a_raw) in (-1, 1)^n
    dq_t     = a_t (.) dq_max                 # per-joint *delta* target (dq_max = code action_max)
    tau_t    = clip(Kp * dq_t - Kd * qdot_t, -tau_max, tau_max)

The SO101 XML drives each joint with a MuJoCo <position> actuator
(kp=499.11, kv=2.731, forcerange = +/-2.94 / +/-3.35 N.m). A position actuator
applies exactly

    tau = clip(kp * (ctrl - q) - kv * qdot, -forcerange, forcerange).

So if we command a *delta target* ctrl = q + dq, MuJoCo realises the README law
verbatim, with Kp=kp=499.11, Kd=kv=2.731, tau_max=forcerange. No torque actuator
or hand-rolled PD loop is needed; this adapter only converts (action, q) -> ctrl.

Verified servo constants: STS3215 stall torque 30 kg.cm = 2.94 N.m; kp/kv follow the
RBE501 DC-motor model (TheRobotStudio SO-ARM101), recalculated 2026-06-12 for firmware
P_gain=8: kp is linear in P (998.22 at P=16 -> 499.11), kv is back-EMF damping
(Km*Kb/R), independent of the firmware P/D registers.
"""
from __future__ import annotations

import numpy as np


class SOArmAdapter:
    def __init__(
        self,
        joint_low: np.ndarray,        # actuator ctrlrange low  (n_dof,)
        joint_high: np.ndarray,       # actuator ctrlrange high (n_dof,)
        action_max: float = 0.3,      # README dq^max: rad of joint delta per unit action
    ):
        self.joint_low = np.asarray(joint_low, dtype=np.float64)
        self.joint_high = np.asarray(joint_high, dtype=np.float64)
        self.action_max = float(action_max)

    def ctrl_target(self, action: np.ndarray, qpos: np.ndarray) -> np.ndarray:
        """action in (-1,1)^n, current joint angles qpos -> position-actuator target.

        target = clip(q + a * action_max, joint_range)
        """
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        return np.clip(qpos + a * self.action_max, self.joint_low, self.joint_high)
