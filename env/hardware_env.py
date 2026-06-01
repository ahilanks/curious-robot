"""Hardware SO-ARM101 env — a drop-in for SubprocVectorMujocoEnv with n_envs=1.

Runs the SAME online-RL loop train.py runs in sim, against one physical SO-ARM101
(6x Feetech STS3215 over a TTL bus) + a wrist USB camera. The trainer can't tell the
difference: identical reset/reset_one/step_block_async/step_block_wait/render_overhead
API, identical obs {"image","proprio"}, identical per-substep info dict, identical
.n_dof/.tau_max/.wrist_resolution.

Three things make it match the sim distribution the safe15 weights were trained on:

1. TORQUE IS RECOMPUTED from the position-actuator PD law, never read off the servo.
   In MuJoCo, proprio[2*n_dof:] and r_safe both consume `data.actuator_force` — the
   realized <position>-actuator output  tau = clip(kp*(goal-q) - kv*qdot, +/-tau_max).
   We evaluate that SAME formula (kp=998.22, kv=2.731, tau_max=3.35 — the SO-101 XML /
   STS3215-at-P-gain-16 values) from the MEASURED q, qdot and the COMMANDED goal. The
   servo's Present_Load is a duty-cycle proxy (~3x off via gearbox friction) and is NOT
   used — feeding it would silently corrupt both proprio and r_safe.

2. REAL-TIME PACING: each control step is held to dt_safe = 0.030 s (read-to-read), so the
   safety reward's qddot = (qdot - qdot_prev)/dt_safe finite-diff spans the same window as
   sim (frame_skip 6 * timestep 0.005). The async block runs on a worker thread so the
   trainer's GPU/MPS updates overlap the physical rollout (same structure as the sim
   step_block_async/step_block_wait). If the learner is slower than the 0.030 s/step pace
   (likely on a laptop), only the GAP BETWEEN decisions stretches — each control step still
   reads exactly dt_safe after it commanded, so qddot stays valid.

3. NO PHYSICAL RESET. reset()/reset_one() READ the current servo state and return the obs
   WITHOUT moving the arm. "End of episode every n steps" (--max-episode-steps) is purely a
   bookkeeping boundary in the trainer (it refreshes the short WM history + logs an episode
   return); the arm just keeps going. So set --max-episode-steps large on hardware — a small
   value only adds frequent history breaks, it never moves the arm here.

Hardware I/O sits behind the ServoBus / Camera protocols at the bottom, so the env logic
(pacing, torque recompute, proprio assembly, safety reward, the async block, read-only reset)
is unit-testable with MockServoBus + MockCamera — `python -m env.hardware_env` runs that
self-test. Plug LeRobotFeetechBus + UsbCamera (with your per-joint tick<->radian calibration)
in for the real arm; this module imports neither lerobot nor mujoco at top level.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

import numpy as np

from .safety_reward import safety_reward_np

# Must mirror env/parallel_env.py::_INFO_KEYS (kept local so this module needs no mujoco).
_INFO_KEYS = ("applied_torque", "qvel", "qvel_prev", "qpos", "safety_reward",
              "object_contacts", "table_contacts", "object_motion")

# --- SO-101 actuation constants (env/SO101/so101_new_calib.xml) ------------------
# <position kp="998.22" kv="2.731" ...> on every joint; per-joint forcerange +/-3.35.
KP = 998.22
KV = 2.731
N_DOF = 6
TAU_MAX = np.array([3.35] * N_DOF, dtype=np.float32)                 # actuator_forcerange[:,1]
# Per-joint ctrlrange = joint limits used to clip the position target (XML <actuator>).
JOINT_LOW = np.array([-1.91986, -1.74533, -1.69, -1.65806, -2.74385, -0.17453], np.float32)
JOINT_HIGH = np.array([1.91986, 1.74533, 1.69, 1.65806, 2.84121, 1.74533], np.float32)
# dt_safe = frame_skip(6) * timestep(0.005) in sim; delta=15 was calibrated to this window.
DT_SAFE = 0.030


# =============================================================================== env
class HardwareSO101Env:
    """One physical SO-ARM101. Drop-in for SubprocVectorMujocoEnv (n_envs must be 1)."""

    def __init__(
        self,
        n_envs: int = 1,
        scene_path=None,                  # accepted for API parity; unused on hardware
        wrist_resolution: int = 224,
        overhead_resolution: int = 256,
        frame_skip: int = 6,              # accepted for parity; control_dt is fixed to DT_SAFE
        action_max: float = 0.3,
        dq_max: float = 100.0,
        safety_delta: float = 15.0,
        seed: int = 0,
        threads: int = 0,                 # accepted for parity; unused
        bus: "ServoBus | None" = None,
        camera: "Camera | None" = None,
        control_dt: float = DT_SAFE,
        vel_lowpass: float = 0.5,         # EMA on qdot (raw servo velocity is noisy -> qddot blows up r_safe)
    ):
        if n_envs != 1:
            raise ValueError(f"HardwareSO101Env is a single physical arm; n_envs must be 1 (got {n_envs})")
        self.n_envs = 1
        self.n_dof = N_DOF
        self.tau_max = TAU_MAX.copy()
        self.wrist_resolution = wrist_resolution
        self.overhead_resolution = overhead_resolution
        self.action_max = float(action_max)
        self.dq_max = float(dq_max)
        self.safety_delta = float(safety_delta)
        self.dt_safe = float(control_dt)
        self.vel_lowpass = float(vel_lowpass)

        self.bus = bus if bus is not None else _default_bus()
        self.camera = camera if camera is not None else _default_camera(wrist_resolution)

        self._exec = ThreadPoolExecutor(max_workers=1)   # runs the action block off the trainer thread
        self._pending = None
        self._q = np.zeros(N_DOF, np.float32)            # last commanded-from joint angles (pre-step q)
        self._qd_prev = np.zeros(N_DOF, np.float32)      # qdot at the previous control-step read
        self._qd_filt = np.zeros(N_DOF, np.float32)      # EMA velocity state
        self._last_frame = None

    # --- observation assembly -------------------------------------------------
    def _read(self) -> tuple[np.ndarray, np.ndarray]:
        """Read (q, qdot) in SI (rad, rad/s) and low-pass qdot."""
        q, qd_raw = self.bus.read()
        q = np.asarray(q, np.float32)
        qd_raw = np.asarray(qd_raw, np.float32)
        self._qd_filt = self.vel_lowpass * qd_raw + (1.0 - self.vel_lowpass) * self._qd_filt
        return q, self._qd_filt.copy()

    @staticmethod
    def _obs(image: np.ndarray, q: np.ndarray, qd: np.ndarray, tau: np.ndarray) -> dict:
        # proprio = [q, qdot, applied_torque] — exact order of MujocoSO101Env._get_obs().
        return {"image": image,
                "proprio": np.concatenate([q, qd, tau]).astype(np.float32)}

    # --- lifecycle: READ-ONLY reset (never commands the arm) -------------------
    def reset(self) -> dict[str, np.ndarray]:
        """Read current state; do NOT move the arm. Returns stacked (n_envs=1) obs."""
        q, qd_raw = self.bus.read()                      # seed the velocity EMA from the live reading
        self._qd_filt = np.asarray(qd_raw, np.float32).copy()
        self._q = np.asarray(q, np.float32)
        self._qd_prev = self._qd_filt.copy()
        tau = np.clip(-KV * self._qd_prev, -TAU_MAX, TAU_MAX).astype(np.float32)  # goal==q => tau=-kv*qdot
        self._last_frame = self.camera.read()
        o = self._obs(self._last_frame, self._q, self._qd_prev, tau)
        return {"image": o["image"][None], "proprio": o["proprio"][None]}

    def reset_one(self, idx: int = 0) -> dict[str, np.ndarray]:
        """Single-env reset (the trainer calls this on episode truncation). READ-ONLY:
        the arm does not move — only the trainer-side history/logging boundary advances."""
        o = self.reset()
        return {"image": o["image"][0], "proprio": o["proprio"][0]}

    # --- one paced control step ----------------------------------------------
    def _control_step(self, action: np.ndarray) -> tuple[dict, dict]:
        a = np.clip(np.asarray(action, np.float32), -1.0, 1.0)
        dq = np.clip(a * self.action_max, -self.dq_max, self.dq_max)
        goal = np.clip(self._q + dq, JOINT_LOW, JOINT_HIGH).astype(np.float32)   # delta-target, == SOArmAdapter

        t_start = time.perf_counter()
        self.bus.write_goal(goal)                         # servo's internal PD realizes the target
        _sleep_until(t_start + self.dt_safe)              # hold dt_safe so qddot finite-diff matches sim
        q_new, qd_new = self._read()                      # read AFTER the wait -> read-to-read == dt_safe

        # RECOMPUTE torque from the same PD law MuJoCo applies (NOT Present_Load).
        tau = np.clip(KP * (goal - q_new) - KV * qd_new, -TAU_MAX, TAU_MAX).astype(np.float32)
        r_safe = float(safety_reward_np(tau, qd_new, self._qd_prev, TAU_MAX,
                                        dt_safe=self.dt_safe, delta=self.safety_delta))
        self._last_frame = self.camera.read()
        info = {
            "applied_torque": tau,
            "qvel": qd_new,
            "qvel_prev": self._qd_prev,
            "qpos": q_new,
            "safety_reward": r_safe,
            "object_contacts": np.int64(0),               # logging-only on hardware -> stubbed
            "table_contacts": np.int64(0),
            "object_motion": np.float32(0.0),
        }
        obs = self._obs(self._last_frame, q_new, qd_new, tau)
        self._q = q_new
        self._qd_prev = qd_new
        return obs, info

    # --- async action block (overlaps the trainer's GPU/MPS work) -------------
    def step_block_async(self, action_blocks: np.ndarray) -> None:
        ab = np.asarray(action_blocks, np.float32)
        assert ab.ndim == 3 and ab.shape[0] == 1 and ab.shape[2] == N_DOF, \
            f"expected (1, action_block, {N_DOF}), got {ab.shape}"
        block = ab[0]

        def run():
            obs, infos = None, []
            for a_k in block:
                obs, info = self._control_step(a_k)
                infos.append(info)
            return obs, infos

        self._pending = self._exec.submit(run)

    def step_block_wait(self):
        obs, infos = self._pending.result()
        self._pending = None
        stacked_obs = {"image": obs["image"][None], "proprio": obs["proprio"][None]}
        sub_infos = [{k: np.stack([info[k]]) for k in _INFO_KEYS} for info in infos]
        return stacked_obs, sub_infos

    def render_overhead(self) -> np.ndarray:
        """No second camera on the minimal rig — return the last wrist frame (logging-only)."""
        frame = self._last_frame if self._last_frame is not None else \
            np.zeros((self.wrist_resolution, self.wrist_resolution, 3), np.uint8)
        return frame[None]

    def close(self) -> None:
        self._exec.shutdown(wait=True)
        try:
            self.bus.close()
        finally:
            self.camera.close()


def _sleep_until(deadline: float) -> None:
    """Pace to `deadline` (perf_counter seconds): sleep the bulk, spin the last ~1 ms so the
    30 ms control period holds tightly without burning a core the whole time."""
    while True:
        rem = deadline - time.perf_counter()
        if rem <= 0:
            return
        if rem > 0.0015:
            time.sleep(rem - 0.001)
        # else: busy-spin the final < ~1.5 ms


# ============================================================ hardware I/O contracts
class ServoBus(Protocol):
    def read(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (q, qdot): joint angles [rad] and velocities [rad/s], each shape (6,),
        in the SAME zero pose + sign convention as the MuJoCo model (this is what the
        per-joint tick<->radian CALIBRATION establishes)."""
        ...

    def write_goal(self, goal_rad: np.ndarray) -> None:
        """Command an absolute position target [rad], shape (6,)."""
        ...

    def close(self) -> None: ...


class Camera(Protocol):
    def read(self) -> np.ndarray:
        """Return one wrist frame as (H, W, 3) uint8 RGB (H=W=wrist_resolution)."""
        ...

    def close(self) -> None: ...


def _default_bus() -> "ServoBus":
    """Build the real bus from env vars when none is injected.
        SOARM_MOCK=1 dry-run the whole loop with the in-process MockServoBus (no hardware)
        SOARM_PORT   serial port of the TTL bus adapter (e.g. /dev/tty.usbmodemXXXX)
        SOARM_CALIB  path to a JSON of {offsets_ticks:[...6], signs:[...6], vel_scale: float}
    """
    import json
    import os
    if os.environ.get("SOARM_MOCK") or os.environ.get("SOARM_PORT") == "mock":
        print("[hardware] SOARM_MOCK set -> MockServoBus (no real arm; dry-run only)", flush=True)
        return MockServoBus()
    port = os.environ.get("SOARM_PORT")
    if not port:
        raise RuntimeError(
            "HardwareSO101Env: no bus injected and $SOARM_PORT is unset. Either pass "
            "bus=LeRobotFeetechBus(...) / a mock, or set SOARM_PORT (+ SOARM_CALIB).")
    calib = {}
    if os.environ.get("SOARM_CALIB"):
        with open(os.environ["SOARM_CALIB"]) as f:
            calib = json.load(f)
    return LeRobotFeetechBus(port=port, **calib)


def _default_camera(hw: int) -> "Camera":
    import os
    if os.environ.get("SOARM_MOCK") or os.environ.get("SOARM_PORT") == "mock":
        return MockCamera(hw=hw)
    return UsbCamera(index=int(os.environ.get("SOARM_CAM", "0")), hw=hw)


class LeRobotFeetechBus:
    """Real STS3215 bus via HuggingFace LeRobot's FeetechMotorsBus.

    YOU MUST run LeRobot's calibration first and supply offsets_ticks + signs so that the
    returned q matches the MuJoCo joint convention (zero pose AND direction). If it doesn't,
    proprio and the recomputed torque are in the wrong frame and r_safe is meaningless.

    The exact LeRobot import path / register names vary by version — adapt the two TODO
    lines to the lerobot you installed (Present_Position / Present_Speed / Goal_Position are
    the standard Feetech control-table fields). This is the ONE place the real arm is touched.
    """
    TICKS_PER_REV = 4096          # STS3215 12-bit absolute encoder
    _RAD_PER_TICK = 2.0 * np.pi / TICKS_PER_REV

    def __init__(self, port: str, motor_ids=(1, 2, 3, 4, 5, 6),
                 offsets_ticks=None, signs=None, vel_scale: float | None = None,
                 p_gain: int = 16):
        if offsets_ticks is None or signs is None or vel_scale is None:
            raise RuntimeError(
                "LeRobotFeetechBus needs calibration: offsets_ticks[6], signs[6], vel_scale "
                "(rad/s per raw velocity unit). Run LeRobot calibration and pass them "
                "(or via SOARM_CALIB json). Refusing to run uncalibrated on a real arm.")
        self.motor_ids = list(motor_ids)
        self.offsets = np.asarray(offsets_ticks, np.float64)
        self.signs = np.asarray(signs, np.float64)
        self.vel_scale = float(vel_scale)
        # TODO[lerobot]: construct the bus for your version, e.g.
        #   from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus
        #   self.bus = FeetechMotorsBus(port=port, motors={...STS3215...}); self.bus.connect()
        #   put every servo in position mode, set P-gain=p_gain, enable torque.
        from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus  # noqa: F401  (lazy)
        raise NotImplementedError(
            "Wire FeetechMotorsBus construction + position-mode/P-gain setup for your "
            "lerobot version here (see the TODO). The conversion math below is ready.")

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        pos_ticks = np.asarray(self.bus.read("Present_Position"), np.float64)   # TODO[lerobot] field name
        vel_raw = np.asarray(self.bus.read("Present_Speed"), np.float64)        # TODO[lerobot] field name
        q = (pos_ticks - self.offsets) * self._RAD_PER_TICK * self.signs
        qd = vel_raw * self.vel_scale * self.signs
        return q.astype(np.float32), qd.astype(np.float32)

    def write_goal(self, goal_rad: np.ndarray) -> None:
        ticks = np.asarray(goal_rad, np.float64) / self.signs / self._RAD_PER_TICK + self.offsets
        self.bus.write("Goal_Position", ticks.round().astype(int))             # TODO[lerobot] field name

    def close(self) -> None:
        if hasattr(self, "bus"):
            self.bus.disconnect()


class UsbCamera:
    """Wrist USB camera via OpenCV; returns (hw, hw, 3) uint8 RGB."""

    def __init__(self, index: int = 0, hw: int = 224):
        import cv2
        self._cv2 = cv2
        self.hw = hw
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"UsbCamera: could not open video index {index}")

    def read(self) -> np.ndarray:
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("UsbCamera: frame grab failed")
        frame = self._cv2.resize(frame, (self.hw, self.hw))
        return self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        if hasattr(self, "cap"):
            self.cap.release()


# ====================================================================== mock drivers
class MockServoBus:
    """In-process fake STS3215 bus: a 1st-order approach toward the last commanded goal
    (+ small noise), so the env's pacing / torque recompute / safety / threading are
    testable without hardware. Records every commanded goal so a test can assert that
    READ-ONLY reset issues no motion command."""

    def __init__(self, q0=None, seed: int = 0):
        self.q = np.zeros(N_DOF) if q0 is None else np.asarray(q0, float).copy()
        self.qd = np.zeros(N_DOF)
        self.goal = self.q.copy()
        self.goals_written: list[np.ndarray] = []
        self.rng = np.random.default_rng(seed)

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        err = self.goal - self.q                        # advance a fraction toward the goal
        self.qd = 0.3 * err / DT_SAFE + self.rng.normal(0, 0.02, N_DOF)
        self.q = self.q + self.qd * DT_SAFE
        return self.q.copy(), self.qd.copy()

    def write_goal(self, goal_rad: np.ndarray) -> None:
        self.goal = np.clip(goal_rad, JOINT_LOW, JOINT_HIGH).astype(float)
        self.goals_written.append(self.goal.copy())

    def close(self) -> None:
        pass


class MockCamera:
    def __init__(self, hw: int = 224, seed: int = 0):
        self.hw = hw
        self.rng = np.random.default_rng(seed)

    def read(self) -> np.ndarray:
        return self.rng.integers(0, 256, (self.hw, self.hw, 3), dtype=np.uint8)

    def close(self) -> None:
        pass


# ============================================================================ selftest
def _self_test() -> None:
    """`python -m env.hardware_env` — exercise the full interface against the mocks."""
    bus = MockServoBus(seed=0)
    env = HardwareSO101Env(n_envs=1, action_max=0.05, safety_delta=15.0,
                           bus=bus, camera=MockCamera())
    assert env.n_dof == 6 and env.wrist_resolution == 224
    assert env.tau_max.shape == (6,) and np.allclose(env.tau_max, 3.35)

    obs = env.reset()
    assert obs["image"].shape == (1, 224, 224, 3) and obs["image"].dtype == np.uint8
    assert obs["proprio"].shape == (1, 18) and obs["proprio"].dtype == np.float32

    # reset_one must NOT command the arm (read-only) ---------------------------
    n_goals = len(bus.goals_written)
    o_one = env.reset_one(0)
    assert o_one["image"].shape == (224, 224, 3) and o_one["proprio"].shape == (18,)
    assert len(bus.goals_written) == n_goals, "reset_one issued a motion command (must be read-only)"

    # one decision = action_block of 5 paced control steps ---------------------
    action_block = 5
    a = np.random.uniform(-1, 1, (1, action_block, 6)).astype(np.float32)
    t0 = time.perf_counter()
    env.step_block_async(a)
    obs, sub_infos = env.step_block_wait()
    dt = time.perf_counter() - t0

    assert obs["proprio"].shape == (1, 18)
    assert len(sub_infos) == action_block
    for info in sub_infos:
        assert set(info.keys()) == set(_INFO_KEYS)
        assert info["applied_torque"].shape == (1, 6)
        assert np.all(np.abs(info["applied_torque"]) <= 3.35 + 1e-4)   # torque clipped to forcerange
        assert np.isfinite(info["safety_reward"]).all()
        assert info["safety_reward"].item() <= 0.0                     # r_safe is a penalty (<= 0)
        assert info["object_contacts"].item() == 0                     # stubbed
    assert action_block * env.dt_safe - 0.01 <= dt, f"block ran too fast ({dt:.3f}s); pacing broken"
    assert len(bus.goals_written) == n_goals + action_block, "expected one Goal_Position per control step"

    env.close()
    print(f"[selftest] OK — block of {action_block} steps in {dt*1000:.0f} ms "
          f"(expected >= {action_block*env.dt_safe*1000:.0f} ms), proprio=(1,18), "
          f"{len(_INFO_KEYS)} info keys, torque clipped to +/-3.35, reset is read-only.")


if __name__ == "__main__":
    _self_test()
