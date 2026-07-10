"""Child-arm scripted sweep calibration smoke (no WM).

Adapts the parent's calibration-verified rise-drop-sweep keyframe cycle
(src/parent_vla.py ScriptedParentDriver) to the CHILD arm, driven through the
NORMAL action channel (delta-position, |a|<=1) so the resulting windows are
WM-explainable. Verifies the child-frame aim signs + contact yield before the
stage-2 contact probe spends GPU time on it.
"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "osmesa")
sys.path.insert(0, "/workspace/curious-robot")

import numpy as np
from env.mujoco_env import MujocoSO101Env

FRAME_SKIP, ACTION_MAX, ACTION_BLOCK = 6, 0.3, 5
SWEEP = np.deg2rad(22.0)
T_CYC = 90  # substeps per rise-drop-sweep cycle (parent-verified cadence)


TOUCHDOWN = (0.15, 0.55)   # fingertip ~r0.31 at table level: just BEYOND the target block
RAKE_IN = (0.65, 1.40)     # fingertip ~r0.21 at table level (static contact champion r~0.20)
BLOCK_R = 0.25             # teleport radius: middle of the raked band r[0.21, 0.31]


def child_aim(env):
    """(pan, r) bearing of block0 in the CHILD base frame (base at origin, +x).
    Depth poses are fixed constants (TOUCHDOWN/RAKE_IN); only the bearing is aimed."""
    b = env.data.xpos[env._object_body_ids[0]]
    rel = b[:2]
    return float(np.arctan2(rel[1], rel[0])), float(np.linalg.norm(rel))


class ChildPlow:
    """Convergence-gated RISE -> DROP-A -> DROP-B -> DRAG state machine (child arm).

    Why not the parent's TIMED cadence: the child's lift/elbow PD tracking converges
    far slower than the pan ramp, so a timed sweep crosses the block's bearing while
    the arm is still ~13 cm high/short (measured 2026-07-10).
    Why a RADIAL RAKE, not a pan-sweep at fixed depth: descending inside the block's
    radius plants the fingertips on the TABLE (friction pins every joint; lift stalls
    -0.20 vs target +0.30 forever), and block r=0.32 is the arm's reach EDGE at table
    level (fingertip max ~0.33 -- also why v2's front-spot walk got 0.4% contacts).
    Instead: touch down at max extension just BEYOND the block, then drag INWARD at
    depth -- the fingertip sweeps the whole r[0.21, 0.31] annulus and must cross the
    block. Contact seen during DROP-B (via observe()) short-circuits to RAKE.
    CHILD home is NOT the parent's all-zeros (that is a table-level pose for the
    child): home = (lift -1.0, elbow 0.6), ee_z ~0.25, pan pre-aimed while airborne.
    """
    HOME, HOVER = (-1.0, 0.6), (-0.2, 0.8)
    TOL, T_RISE_MAX, T_DROP_MAX, T_RAKE, N_PASS = 0.12, 40, 45, 35, 4

    def __init__(self, env):
        self.env = env
        self.contact_hit = False
        self.state, self.t, self.aim = "rise", 0, child_aim(env)

    def observe(self, n_contacts: int):
        if n_contacts > 0:
            self.contact_hit = True

    def _goto(self, lift, elbow, next_state, timeout_state=None):
        q = self.env.data.qpos[:self.env.n_dof]
        pan = self.aim[0]
        if max(abs(q[1] - lift), abs(q[2] - elbow)) < self.TOL:
            self.state, self.t = next_state, 0
            return None
        if self.t >= (self.T_RISE_MAX if self.state == "rise" else self.T_DROP_MAX):
            self.state, self.t = (timeout_state or next_state), 0
            return None
        self.t += 1
        return np.array([pan, lift, elbow, 0.5, 0.0, 0.3])

    def keyframe(self):
        if self.state == "rise":
            self.contact_hit = False                          # stale contacts don't carry over
            kf = self._goto(*self.HOME, "dropA")
            if kf is not None:
                return kf
            self.aim = child_aim(self.env)                    # re-aim while high
        if self.state == "dropA":
            kf = self._goto(*self.HOVER, "dropB", timeout_state="rise")
            if kf is not None:
                return kf
        if self.state == "dropB":
            if self.contact_hit:                              # already on a block = start raking
                self.state, self.t = "rake", 0
            else:
                kf = self._goto(*TOUCHDOWN, "rake", timeout_state="rise")
                if kf is not None:
                    return kf
                if self.state == "rise":                      # wedged: un-wedge via home
                    return self.keyframe()
        # rake: alternating inward/outward radial drags at depth, pan frozen at bearing
        pan = self.aim[0]
        k = self.t // self.T_RAKE
        if k >= self.N_PASS:
            self.state, self.t = "rise", 0
            return self.keyframe()
        lift, elbow = RAKE_IN if k % 2 == 0 else TOUCHDOWN
        self.t += 1
        return np.array([pan, lift, elbow, 0.5, 0.0, 0.3])


def teleport_front(env):
    addr = env._object_qpos_addrs[0]
    vadr = env._object_qvel_addrs[0]
    env.data.qpos[addr:addr + 2] = [BLOCK_R, 0.0]
    env.data.qpos[addr + 2] = env._object_resting_z(0)
    env.data.qvel[vadr:vadr + 6] = 0.0
    import mujoco
    mujoco.mj_forward(env.model, env.data)


if __name__ == "__main__":
    env = MujocoSO101Env(frame_skip=FRAME_SKIP, action_max=ACTION_MAX, encode_cam="wrist",
                         safety_delta=9.0, seed=11, fixed_objects=False)
    n_dof = env.n_dof
    for ep in range(8):
        env.reset(); teleport_front(env)
        plow = ChildPlow(env)
        contacts, cdec = 0, 0
        for dec in range(60):
            ncon = 0
            for _ in range(ACTION_BLOCK):
                q_kf = plow.keyframe()
                qpos = env.data.qpos[:n_dof].copy()
                a = np.clip((q_kf - qpos) / ACTION_MAX, -1, 1)
                _, info = env.step(a, render=False)
                plow.observe(int(info["object_contacts"]))
                ncon += int(info["object_contacts"])
            contacts += ncon; cdec += int(ncon > 0)
        b = env.data.xpos[env._object_body_ids[0]][:2]
        print(f"[ep {ep}] contact decisions {cdec}/60 (substep events {contacts})  "
              f"block0 -> ({b[0]:+.3f},{b[1]:+.3f}) r={np.hypot(*b):.3f}", flush=True)
    env.close()
