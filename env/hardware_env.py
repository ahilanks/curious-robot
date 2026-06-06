"""Hardware SO-ARM101 env — a drop-in for SubprocVectorMujocoEnv with n_envs=1.

Runs the SAME online-RL loop train.py runs in sim, against one physical SO-ARM101
(6x Feetech STS3215 over a TTL bus) + a wrist USB camera. The trainer can't tell the
difference: identical reset/reset_one/step_block_async/step_block_wait/render_overhead
API, identical obs {"image","proprio"}, identical per-substep info dict, identical
.n_dof/.tau_max/.wrist_resolution.

Four things make it match the sim distribution the safe15 weights were trained on:

1. OBS TORQUE IS RECOMPUTED, REWARD TORQUE IS MEASURED — two torques, deliberately split.
   - proprio[2*n_dof:] (and info["applied_torque"]) use the sim position-actuator PD law
     tau = clip(kp*(goal-q) - kv*qdot, +/-tau_max) with kp=998.22 / kv=2.731 / tau_max=3.35
     (the SO-101 XML values) from MEASURED q, qdot and the COMMANDED goal. This keeps the
     OBSERVATION distribution the sim-trained encoder saw (sim's `data.actuator_force`
     saturates ~87% of moving samples; the recompute ~90% — measured 2026-06-03).
   - r_safe uses tau_meas = kt * Present_Current (measured motor effort, joint-frame sign,
     EMA-filtered like qdot). The recompute is a COUNTERFACTUAL on hardware: kp=998 saturates
     at 0.19 deg of tracking error, so a P=16 servo chasing 30 ms goals pegs |tau|=tau_max
     whenever moving, and every arrival/stall (qdot drops while error > 0.2 deg) scored as a
     max-weight "fight" — that artifact dominated the sim->real r_safe gap (-35 vs -190 at
     matched config). Measured current's sign tracks the REAL drive direction, so normal
     servo arrivals score compliant, as they do in sim where the scored torque is the one
     actually braking the joint. Both r_safe variants are logged per control step
     (info["safety_reward"] = measured-live, info["r_safe_recompute"] = old diagnostic) so a
     bench A/B can attribute improvements. (Present_Load remains unused: duty-cycle proxy.)

2. REAL-TIME PACING: each control step is held to dt_safe = 0.030 s (read-to-read), so the
   safety reward's qddot = (qdot - qdot_prev)/dt_safe finite-diff spans the same window as
   sim (frame_skip 6 * timestep 0.005). The async block runs on a worker thread so the
   trainer's GPU/MPS updates overlap the physical rollout (same structure as the sim
   step_block_async/step_block_wait). If the learner is slower than the 0.030 s/step pace
   (likely on a laptop), only the GAP BETWEEN decisions stretches — each control step still
   reads exactly dt_safe after it commanded, so qddot stays valid.

2b. TARGET PACING (Goal_Speed): sim alpha-interpolates each delta-target across the 30 ms
   window (6 x 5 ms substeps, mujoco_env step); with the servo's factory Acceleration=0 +
   static GoalSpeed=2000 the real arm instead RACED to each goal and dead-stopped — a
   move-stop sawtooth 5x per decision (the physically-confirmed start-stop jerk). FeetechBus
   now writes a per-step Goal_Speed = |delta_ticks|/pace_dt so the firmware executes each
   move as a constant-velocity ramp spanning the window — sim's target ramp by construction.
   Disable for A/B with SOARM_NO_PACE=1 (or pace_dt=0 in the calib json).

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

import os
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
        cur_lowpass: float | None = None,  # EMA on measured torque; None = vel_lowpass so tau_meas and the
    ):                                     # qddot it multiplies share a time constant (mismatch can flip -tau*qddot)
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
        self.cur_lowpass = float(cur_lowpass) if cur_lowpass is not None else float(vel_lowpass)

        self.bus = bus if bus is not None else _default_bus()
        self.camera = camera if camera is not None else _default_camera(wrist_resolution)

        self._exec = ThreadPoolExecutor(max_workers=1)   # runs the action block off the trainer thread
        self._pending = None
        self._q = np.zeros(N_DOF, np.float32)            # last commanded-from joint angles (pre-step q)
        self._qd_prev = np.zeros(N_DOF, np.float32)      # qdot at the previous control-step read
        self._qd_filt = np.zeros(N_DOF, np.float32)      # EMA velocity state
        self._tau_filt = np.zeros(N_DOF, np.float32)     # EMA measured-torque state (kt * Present_Current)
        self._last_frame = None
        self._ctrl_steps = 0                             # control-step counter (for SOARM_DEBUG prints)
        self._debug_every = int(os.environ.get("SOARM_DEBUG", "0"))   # print both r_safe variants every N steps

    # --- observation assembly -------------------------------------------------
    def _read(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read (q, qdot, tau_meas) in SI (rad, rad/s, N*m); low-pass qdot and tau_meas
        with matched time constants. tau_meas is clipped to +/-tau_max so the r_safe
        weight |tau|/tau_max stays <= 1 (strict sim form parity)."""
        q, qd_raw, tau_raw = self.bus.read()
        q = np.asarray(q, np.float32)
        qd_raw = np.asarray(qd_raw, np.float32)
        tau_raw = np.asarray(tau_raw, np.float32)
        self._qd_filt = self.vel_lowpass * qd_raw + (1.0 - self.vel_lowpass) * self._qd_filt
        self._tau_filt = self.cur_lowpass * tau_raw + (1.0 - self.cur_lowpass) * self._tau_filt
        return q, self._qd_filt.copy(), np.clip(self._tau_filt, -TAU_MAX, TAU_MAX).astype(np.float32)

    @staticmethod
    def _obs(image: np.ndarray, q: np.ndarray, qd: np.ndarray, tau: np.ndarray) -> dict:
        # proprio = [q, qdot, applied_torque] — exact order of MujocoSO101Env._get_obs().
        return {"image": image,
                "proprio": np.concatenate([q, qd, tau]).astype(np.float32)}

    # --- lifecycle: READ-ONLY reset (never commands the arm) -------------------
    def reset(self) -> dict[str, np.ndarray]:
        """Read current state; do NOT move the arm. Returns stacked (n_envs=1) obs."""
        q, qd_raw, tau_raw = self.bus.read()             # seed the EMA filters from the live reading
        self._qd_filt = np.asarray(qd_raw, np.float32).copy()
        self._tau_filt = np.asarray(tau_raw, np.float32).copy()
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
        q_new, qd_new, tau_meas = self._read()            # read AFTER the wait -> read-to-read == dt_safe

        # OBS torque: the sim PD-law recompute (distribution-match for the sim-trained
        # encoder; see module docstring #1). NOT used for the reward.
        tau = np.clip(KP * (goal - q_new) - KV * qd_new, -TAU_MAX, TAU_MAX).astype(np.float32)
        # REWARD torque: measured motor effort (kt * Present_Current). Its sign tracks the
        # real drive direction, so servo arrivals/stalls are not billed as max-torque fights.
        r_safe = float(safety_reward_np(tau_meas, qd_new, self._qd_prev, TAU_MAX,
                                        dt_safe=self.dt_safe, delta=self.safety_delta))
        r_safe_rec = float(safety_reward_np(tau, qd_new, self._qd_prev, TAU_MAX,
                                            dt_safe=self.dt_safe, delta=self.safety_delta))
        self._last_frame = self.camera.read()
        self._ctrl_steps += 1
        if self._debug_every and self._ctrl_steps % self._debug_every == 0:
            spd = getattr(self.bus, "_last_speeds", None)   # per-step paced Goal_Speed (FeetechBus)
            print(f"[hw dbg] ctrl_step={self._ctrl_steps} r_safe meas={r_safe:.1f} "
                  f"recomp={r_safe_rec:.1f} |tau_meas|max={np.abs(tau_meas).max():.2f} "
                  f"|qd|max={np.abs(qd_new).max():.2f}"
                  + (f" pace={list(spd)}" if spd is not None else ""), flush=True)
        info = {
            "applied_torque": tau,
            "qvel": qd_new,
            "qvel_prev": self._qd_prev,
            "qpos": q_new,
            "safety_reward": r_safe,
            "object_contacts": np.int64(0),               # logging-only on hardware -> stubbed
            "table_contacts": np.int64(0),
            "object_motion": np.float32(0.0),
            # diagnostics (not in _INFO_KEYS -> dropped by step_block_wait stacking; for
            # bench scripts calling _control_step directly and the SOARM_DEBUG print):
            "tau_meas": tau_meas,                         # the torque the live r_safe scored
            "r_safe_recompute": r_safe_rec,               # old recompute-metric r_safe (A/B attribution)
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
    def read(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (q, qdot, tau_meas): joint angles [rad], velocities [rad/s] and MEASURED
        joint torques [N*m] (kt * Present_Current on real hardware), each shape (6,), in
        the SAME zero pose + sign convention as the MuJoCo model (this is what the
        per-joint tick<->radian CALIBRATION establishes). tau_meas feeds ONLY r_safe;
        the observation torque is recomputed env-side from the sim PD law."""
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
        SOARM_MOCK=1    dry-run the whole loop with the in-process MockServoBus (no hardware)
        SOARM_PORT      serial port of the TTL bus adapter (e.g. /dev/tty.usbmodemXXXX)
        SOARM_CALIB     path to a JSON of {offsets_ticks:[...6], signs:[...6], vel_scale: float,
                        p_gain, d_gain, goal_speed, pace_dt, acceleration, kt} (later keys optional)
        SOARM_NO_PACE=1 disable per-step Goal_Speed pacing (A/B against the legacy race-and-stop)
        SOARM_DEBUG=N   print both r_safe variants + pacing every N control steps (env-side)
    """
    import json
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
    if os.environ.get("SOARM_NO_PACE"):
        calib["pace_dt"] = 0.0
        print("[hardware] SOARM_NO_PACE set -> legacy static Goal_Speed (no per-step pacing)", flush=True)
    return FeetechBus(port=port, **calib)


def _default_camera(hw: int) -> "Camera":
    import os
    if os.environ.get("SOARM_MOCK") or os.environ.get("SOARM_PORT") == "mock":
        return MockCamera(hw=hw)
    return UsbCamera(index=int(os.environ.get("SOARM_CAM", "0")), hw=hw)


class FeetechBus:
    """Real STS3215 bus via Feetech's scservo_sdk (the same wire protocol LeRobot uses under
    the hood). Speaks raw 0-4095 ticks, mapped into the MuJoCo joint frame by the calibration
    (offsets_ticks, signs, vel_scale) measured against the so101_new_calib.xml model and stored
    in SOARM_CALIB json. This is the ONE place the real arm is energized.

    STS3215 control table (protocol_end=0): Torque_Enable 40, Acceleration 41, Goal_Position 42,
    Goal_Speed 46, Present_Position 56, Present_Speed 58 (sign-magnitude), Present_Current 69
    (sign-magnitude, ~6.5 mA/LSB), Mode 33, P_gain 21. The SO-101 ships Mode=0 (position mode).
    P-gain/D-gain default to the shipped-stable 16/32 (tunable via the calib json; raising them
    did NOT smooth safe15's real-arm jerk — that was the unpaced goal staircase, see logistics.md).

    PACING: with pace_dt > 0 (default: DT_SAFE) every write_goal also writes a per-servo
    Goal_Speed = |delta_ticks|/pace_dt (clamped [1, goal_speed]) so the firmware ramps each move
    across the control window instead of racing at the static cap and dead-stopping — the
    hardware analog of sim's alpha-interpolated target. `goal_speed` doubles as the pacing CAP
    and as the static value when pacing is off. Speed is written BEFORE position (the position
    write triggers the move). Goal_Speed=0 means MAX on Feetech, hence the >=1 clamp.
    `acceleration` (reg 41, ~100 ticks/s^2 per LSB, 0 = no firmware ramp = factory default) is
    written at construction for optional bench experiments; note a finite value caps how fast
    the servo reaches its paced speed — too low adds lag, so it stays 0 unless the bench says
    otherwise. kt [N*m/A] scales Present_Current to joint torque for r_safe (placeholder 1.0
    until bench-calibrated; it sits in the calib json next to vel_scale — both sensor scales).

    enable_torque() sets each goal to the CURRENT position BEFORE enabling torque, so the arm
    HOLDS where it is instead of snapping to the stale Goal_Position register (0 at boot).
    """
    ADDR_TORQUE_ENABLE = 40
    ADDR_ACCELERATION = 41
    ADDR_GOAL_POSITION = 42
    ADDR_PRESENT_POSITION = 56
    ADDR_PRESENT_SPEED = 58
    ADDR_PRESENT_CURRENT = 69
    ADDR_MODE = 33
    ADDR_P_GAIN = 21
    ADDR_D_GAIN = 22
    ADDR_GOAL_SPEED = 46
    TICKS_PER_REV = 4096          # STS3215 12-bit absolute encoder
    _RAD_PER_TICK = 2.0 * np.pi / TICKS_PER_REV
    _AMPS_PER_LSB = 0.0065        # Present_Current unit (Feetech STS3215 datasheet)

    def __init__(self, port: str, motor_ids=(1, 2, 3, 4, 5, 6),
                 offsets_ticks=None, signs=None, vel_scale: float | None = None,
                 p_gain: int = 16, d_gain: int = 32, goal_speed: int = 2000,
                 enable_torque: bool = True, max_step_ticks: int = 300,
                 pace_dt: float = DT_SAFE, acceleration: int = 0, kt: float = 1.0):
        if offsets_ticks is None or signs is None or vel_scale is None:
            raise RuntimeError(
                "FeetechBus needs calibration: offsets_ticks[6], signs[6], vel_scale "
                "(rad/s per raw velocity unit). Run the so101 calibration and pass them via "
                "SOARM_CALIB json. Refusing to run uncalibrated on a real arm.")
        from scservo_sdk import PortHandler, PacketHandler, COMM_SUCCESS
        self._OK = COMM_SUCCESS
        self.motor_ids = list(motor_ids)
        self.offsets = np.asarray(offsets_ticks, np.float64)
        self.signs = np.asarray(signs, np.float64)
        self.vel_scale = float(vel_scale)
        self.max_step_ticks = int(max_step_ticks)    # backstop: cap any single commanded jump
        self.goal_speed = int(goal_speed)            # pacing cap / static speed when pacing off
        self.pace_dt = float(pace_dt)                # 0 disables per-step Goal_Speed pacing
        self.kt = float(kt)                          # N*m per A: Present_Current -> joint torque
        self._last_pos = None                         # last good Present_Position (set by enable_torque)
        self._last_speeds = None                      # last paced Goal_Speed values (SOARM_DEBUG)
        self._torque = False
        self.port = PortHandler(port)
        if not self.port.openPort():
            raise RuntimeError(f"FeetechBus: could not open port {port}")
        self.port.setBaudRate(1_000_000)
        self.pk = PacketHandler(0)
        for sid in self.motor_ids:                       # presence check + config (P-gain/speed; no motion)
            _, comm, _ = self.pk.ping(self.port, sid)
            if comm != COMM_SUCCESS:
                raise RuntimeError(f"FeetechBus: servo id {sid} not responding on {port}")
            mode = self.pk.read1ByteTxRx(self.port, sid, self.ADDR_MODE)[0]
            if mode != 0:
                raise RuntimeError(f"FeetechBus: servo {sid} Mode={mode}, need 0 (position).")
            if self.pk.read1ByteTxRx(self.port, sid, self.ADDR_P_GAIN)[0] != int(p_gain):
                self.pk.write1ByteTxRx(self.port, sid, self.ADDR_P_GAIN, int(p_gain))      # tracking stiffness (EEPROM)
            if self.pk.read1ByteTxRx(self.port, sid, self.ADDR_D_GAIN)[0] != int(d_gain):
                self.pk.write1ByteTxRx(self.port, sid, self.ADDR_D_GAIN, int(d_gain))      # damping
            self.pk.write2ByteTxRx(self.port, sid, self.ADDR_GOAL_SPEED, int(goal_speed))  # SRAM, resets on power-cycle
            self.pk.write1ByteTxRx(self.port, sid, self.ADDR_ACCELERATION, int(acceleration))  # 0 = factory (no ramp)
        if enable_torque:
            self.enable_torque()

    def enable_torque(self) -> None:
        """SAFE energize: set each Goal_Position to its CURRENT Present_Position, THEN enable
        torque, so the arm holds where it is rather than snapping to a stale goal (= 0 at boot)."""
        cur_pos = []
        for sid in self.motor_ids:
            cur, c, _ = self.pk.read2ByteTxRx(self.port, sid, self.ADDR_PRESENT_POSITION)
            if c != self._OK:
                raise RuntimeError(f"FeetechBus: read failed on servo {sid} before torque-on")
            self.pk.write2ByteTxRx(self.port, sid, self.ADDR_GOAL_POSITION, int(cur))
            self.pk.write1ByteTxRx(self.port, sid, self.ADDR_TORQUE_ENABLE, 1)
            cur_pos.append(cur)
        self._last_pos = np.asarray(cur_pos, np.float64)   # seed last-good for read()/write_goal guards
        self._torque = True

    def disable_torque(self) -> None:
        for sid in self.motor_ids:
            self.pk.write1ByteTxRx(self.port, sid, self.ADDR_TORQUE_ENABLE, 0)
        self._torque = False

    def read(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pos, spd, cur = [], [], []
        for i, sid in enumerate(self.motor_ids):
            p, cp, _ = self.pk.read2ByteTxRx(self.port, sid, self.ADDR_PRESENT_POSITION)
            if cp != self._OK:
                p, cp, _ = self.pk.read2ByteTxRx(self.port, sid, self.ADDR_PRESENT_POSITION)  # 1 retry
            if cp != self._OK:                               # NEVER fall back to 0 -> phantom slam;
                p = self._last_pos[i] if self._last_pos is not None else self.offsets[i]  # keep last good
            pos.append(p)
            s, cs, _ = self.pk.read2ByteTxRx(self.port, sid, self.ADDR_PRESENT_SPEED)
            mag = (s & 0x7FFF) if cs == self._OK else 0      # Present_Speed sign-magnitude; 0 vel on drop
            spd.append(-mag if (cs == self._OK and (s & 0x8000)) else mag)
            c, cc, _ = self.pk.read2ByteTxRx(self.port, sid, self.ADDR_PRESENT_CURRENT)
            cmag = (c & 0x7FFF) if cc == self._OK else 0     # Present_Current sign-magnitude; 0 on drop
            cur.append(-cmag if (cc == self._OK and (c & 0x8000)) else cmag)   # (EMA env-side smooths drops)
        pos = np.asarray(pos, np.float64); spd = np.asarray(spd, np.float64)
        cur = np.asarray(cur, np.float64)
        self._last_pos = pos.copy()                          # refresh last-good
        q = (pos - self.offsets) * self._RAD_PER_TICK * self.signs
        qd = spd * self.vel_scale * self.signs
        tau = cur * self._AMPS_PER_LSB * self.kt * self.signs   # measured joint torque [N*m]
        return q.astype(np.float32), qd.astype(np.float32), tau.astype(np.float32)

    @staticmethod
    def _pace_speed(delta_ticks: float, pace_dt: float, cap: int) -> int:
        """Goal_Speed [ticks/s] that traverses |delta_ticks| in exactly pace_dt: the firmware
        executes the move as a constant-velocity ramp spanning the control window (sim's
        alpha-interpolated target, done in firmware). ceil -> arrive marginally early rather
        than late; clamp >= 1 because Goal_Speed=0 means MAX speed on Feetech."""
        return int(min(cap, max(1, np.ceil(abs(delta_ticks) / max(pace_dt, 1e-6)))))

    def write_goal(self, goal_rad: np.ndarray) -> None:
        ticks = np.asarray(goal_rad, np.float64) / self.signs / self._RAD_PER_TICK + self.offsets
        if self._last_pos is not None:                       # backstop: no single command may jump more
            ticks = np.clip(ticks, self._last_pos - self.max_step_ticks,   # than max_step_ticks from the
                            self._last_pos + self.max_step_ticks)          # current position
        ticks = np.clip(np.round(ticks), 0, 4095).astype(int)   # final guard against out-of-range
        pace = self.pace_dt > 0 and self._last_pos is not None
        if pace:
            self._last_speeds = [self._pace_speed(t - lp, self.pace_dt, self.goal_speed)
                                 for t, lp in zip(ticks, self._last_pos)]
        for i, (sid, t) in enumerate(zip(self.motor_ids, ticks)):
            if pace:   # speed BEFORE position: the position write triggers the move
                self.pk.write2ByteTxRx(self.port, sid, self.ADDR_GOAL_SPEED, self._last_speeds[i])
            self.pk.write2ByteTxRx(self.port, sid, self.ADDR_GOAL_POSITION, int(t))

    def close(self) -> None:
        # leave torque as-is (holding the last pose) so the arm does not drop on shutdown;
        # call disable_torque() explicitly to free it.
        if hasattr(self, "port"):
            self.port.closePort()


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
    (+ small noise), so the env's pacing / torque split / safety / threading are
    testable without hardware. Records every commanded goal so a test can assert that
    READ-ONLY reset issues no motion command. read() returns a fake measured torque whose
    SIGN tracks the drive direction (goal - q), ~0 at rest — the property the current-based
    r_safe relies on (real source: kt * Present_Current)."""

    def __init__(self, q0=None, seed: int = 0):
        self.q = np.zeros(N_DOF) if q0 is None else np.asarray(q0, float).copy()
        self.qd = np.zeros(N_DOF)
        self.goal = self.q.copy()
        self.goals_written: list[np.ndarray] = []
        self.rng = np.random.default_rng(seed)

    def read(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        err = self.goal - self.q                        # advance a fraction toward the goal
        self.qd = 0.3 * err / DT_SAFE + self.rng.normal(0, 0.02, N_DOF)
        self.q = self.q + self.qd * DT_SAFE
        tau = np.clip(3.35 * np.tanh(8.0 * err) - 0.5 * self.qd, -3.35, 3.35) \
            + self.rng.normal(0, 0.05, N_DOF)           # drive-direction effort, ~0 at rest
        return self.q.copy(), self.qd.copy(), tau.astype(np.float32)

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

    # obs/reward torque DECOUPLING (load-bearing: a swap silently reintroduces the OOD-proprio
    # or fake-fight bug). applied_torque must be EXACTLY the sim PD-law recompute, and must
    # NOT be the measured torque the live r_safe scored.
    q_before = env._q.copy()
    a_one = np.full(N_DOF, 0.5, np.float32)
    o_one, info = env._control_step(a_one)
    assert "tau_meas" in info and "r_safe_recompute" in info
    goal_exp = np.clip(q_before + np.clip(np.clip(a_one, -1, 1) * env.action_max,
                                          -env.dq_max, env.dq_max), JOINT_LOW, JOINT_HIGH)
    tau_exp = np.clip(KP * (goal_exp - info["qpos"]) - KV * info["qvel"], -TAU_MAX, TAU_MAX)
    assert np.allclose(info["applied_torque"], tau_exp, atol=1e-5), \
        "proprio torque must stay the sim PD-law recompute"
    assert not np.allclose(info["applied_torque"], info["tau_meas"]), \
        "applied_torque == tau_meas: obs/reward torque split was lost"
    assert np.all(np.abs(info["tau_meas"]) <= 3.35 + 1e-4) and np.isfinite(info["r_safe_recompute"])
    assert np.allclose(o_one["proprio"][2 * N_DOF:], info["applied_torque"])   # proprio carries the recompute

    # Goal_Speed pacing math (single source used by FeetechBus.write_goal; the mock never
    # exercises the register path, so pin the function here)
    ps = FeetechBus._pace_speed
    assert ps(0.0, DT_SAFE, 2000) == 1                      # no move -> min (0 would mean MAX speed)
    assert ps(-30.0, DT_SAFE, 2000) == 1000                 # 30 ticks over 30 ms -> 1000 ticks/s (sign-free)
    assert ps(45.0, DT_SAFE, 2000) == 1500
    assert ps(90.0, DT_SAFE, 2000) == 2000                  # capped at goal_speed
    assert ps(65.0, 0.030, 2000) == 2000                    # action_max 0.1-ish full step -> caps

    env.close()
    print(f"[selftest] OK — block of {action_block} steps in {dt*1000:.0f} ms "
          f"(expected >= {action_block*env.dt_safe*1000:.0f} ms), proprio=(1,18), "
          f"{len(_INFO_KEYS)} info keys, torque clipped to +/-3.35, reset is read-only, "
          f"obs-torque=PD-recompute / reward-torque=measured split verified, pacing math pinned.")


if __name__ == "__main__":
    _self_test()
