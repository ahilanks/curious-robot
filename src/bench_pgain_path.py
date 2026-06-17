"""Bench: P-gain (stiffness, reg 21) sweep under a CONTINUOUS multi-waypoint path.

Unlike bench_dgain_wide (stop-and-settle per waypoint), this drives ONE smooth joint-space
path THROUGH all waypoints: Catmull-Rom spline (C1 — velocity continuous through each
point, no stop between segments) with a global min-jerk time warp (ease-in/out only at the
global start/end). Per gain: tracking error, accel/jerk RMS of the MEASURED motion (the
smoothness numbers), peak |tau|.

Guards (lessons from the 2026-06-10 D=254 event + the table):
  - the table is LEVEL WITH THE BASE: the reference is hard-capped on the downward side
    (lift <= +0.15, elbow <= +0.30, wrist <= +0.30 rad) so even saggy low-P tracking
    keeps the gripper well above the base plane; pan stays in the cable-safe band
  - live current abort: 5 consecutive ticks with |tau| > 20 on any joint stops the path
  - Torque_Enable (reg 40) verified after every run — firmware overload protection cuts
    torque SILENTLY; if it tripped, re-energize safely and stop the sweep
  - P restored to 16 at the end (live-only on this rig — a power cycle also restores 16)

    python src/bench_pgain_path.py --port /dev/cu.usbmodem5AA90245791
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from env.hardware_env import FeetechBus, JOINT_HIGH, JOINT_LOW  # noqa: E402
from src.bench_dgain_wide import goto, min_jerk, set_dgain       # noqa: E402

DEG = 180.0 / np.pi
PAN_SAFE = (0.05, 0.70)
# downward caps (positive lift/elbow/wrist pitch the arm toward the table)
DOWN_CAP = np.array([np.inf, 0.15, 0.30, 0.30, np.inf, np.inf])
HOME = np.array([0.35, 0.00, 0.00, 0.00, 0.00, 0.50])

#  pan   lift  elbow  wrist  roll  grip   — varied directions, ends back at home
WAYPOINTS = np.array([
    HOME,
    [0.65, -0.40, -0.60, -0.40, -0.50, 0.20],   # right, fold-side raise, roll left
    [0.50,  0.10,  0.25,  0.25,  0.50, 0.80],   # shallow forward reach, roll right
    [0.10, -0.70, -0.30, -0.70,  0.00, 0.10],   # left, lean back, wrist up
    [0.30, -0.20,  0.30, -0.20, -0.60, 0.70],   # mid, gentle forward elbow
    [0.60, -0.90, -0.90, -0.50,  0.40, 0.30],   # right, toward the fold
    HOME,
], dtype=np.float64)


def clip_ref(q: np.ndarray) -> np.ndarray:
    q = np.clip(q, JOINT_LOW, np.minimum(JOINT_HIGH, DOWN_CAP))
    q[0] = np.clip(q[0], *PAN_SAFE)
    return q


def catmull_rom(pts: np.ndarray, u: float) -> np.ndarray:
    """C1 spline through pts (endpoints duplicated -> zero end tangents); u in [0, n_seg]."""
    p = np.vstack([pts[0], pts, pts[-1]])
    k = int(np.clip(np.floor(u), 0, len(pts) - 2))
    t = u - k
    p0, p1, p2, p3 = p[k], p[k + 1], p[k + 2], p[k + 3]
    return 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                  + (-p0 + 3 * p1 - 3 * p2 + p3) * t * t * t)


def set_pgain(bus: FeetechBus, val: int) -> None:
    """Live P write with torque ON (verified 2026-06-09); EEPROM-wait + comm-checked
    readback (an immediate readback after a write burst drops and reads as 0)."""
    for sid in bus.motor_ids:
        bus.pk.write1ByteTxRx(bus.port, sid, bus.ADDR_P_GAIN, int(val))
    time.sleep(0.1)
    for sid in bus.motor_ids:
        for _ in range(4):
            v, c, _ = bus.pk.read1ByteTxRx(bus.port, sid, bus.ADDR_P_GAIN)
            if c == bus._OK:
                break
            time.sleep(0.05)
        if c != bus._OK or v != int(val):
            raise RuntimeError(f"P-gain readback failed on servo {sid}: got {v}")


def torque_flags(bus: FeetechBus) -> list:
    out = []
    for sid in bus.motor_ids:
        v, c, _ = bus.pk.read1ByteTxRx(bus.port, sid, bus.ADDR_TORQUE_ENABLE)
        out.append(int(v) if c == bus._OK else None)
    return out


def run_path(bus: FeetechBus, total_t: float):
    """Drive the spline once; returns (t, q_ref, q, tau) logs + aborted flag."""
    n_seg = len(WAYPOINTS) - 1
    log_t, log_ref, log_q, log_tau = [], [], [], []
    hot = 0
    t0 = time.perf_counter()
    while True:
        now = time.perf_counter() - t0
        u = n_seg * min_jerk(now / total_t)
        ref = clip_ref(catmull_rom(WAYPOINTS, u))
        bus.write_goal(ref)
        q, _, tau = bus.read()
        log_t.append(now); log_ref.append(ref); log_q.append(q); log_tau.append(tau)
        hot = hot + 1 if np.abs(tau).max() > 20.0 else 0
        if hot >= 5:                                   # sustained fight: stop and hold here
            bus.write_goal(q.astype(np.float64))
            print("  !! current abort: |tau|>20 for 5 ticks — path stopped", flush=True)
            return (np.asarray(log_t), np.asarray(log_ref),
                    np.asarray(log_q), np.asarray(log_tau), True)
        if now >= total_t:
            time.sleep(0.5)                            # settle before the end-pose read
            return (np.asarray(log_t), np.asarray(log_ref),
                    np.asarray(log_q), np.asarray(log_tau), False)
        time.sleep(max(0.0, 0.030 - (time.perf_counter() - t0 - now)))


def metrics(t, ref, q, tau):
    err = np.abs(q - ref) * DEG
    dt = np.gradient(t)
    qd = np.gradient(q, axis=0) / dt[:, None]
    qdd = np.gradient(qd, axis=0) / dt[:, None]
    jerk = np.gradient(qdd, axis=0) / dt[:, None]
    return dict(track_mean=float(err.mean()), track_max=float(err.max()),
                acc_rms=float(np.sqrt((qdd ** 2).mean())) * DEG,
                jerk_rms=float(np.sqrt((jerk ** 2).mean())) * DEG,
                taumax=float(np.abs(tau).max()))


def main(a):
    import json
    calib = json.loads(Path(a.calib).read_text())
    gains = [int(g) for g in (a.gains or {"p": "8,16,32,48", "d": "8,32,128"}[a.reg]).split(",")]
    if a.reg == "d":
        if any(g > 128 for g in gains):  # D=254 chattered into overload shutdown (2026-06-10)
            raise SystemExit("refusing D-gain > 128: known overload-shutdown regime")
        set_gain, restore, label = set_dgain, 32, "D"
    else:
        set_gain, restore, label = set_pgain, a.restore_pgain, "P"
    bus = FeetechBus(port=a.port, **{k: calib[k] for k in
                                     ("offsets_ticks", "signs", "vel_scale", "p_gain",
                                      "d_gain", "goal_speed", "pace_dt", "acceleration", "kt")
                                     if k in calib})
    results = {}
    try:
        for g in gains:
            temps = bus.read_temps()
            print(f"\n=== {label}-gain {g} ===  temps {temps.tolist()}", flush=True)
            if temps.max() > 48.0:
                print("  !! temps too high — stopping the sweep", flush=True)
                break
            set_gain(bus, g)
            goto(bus, HOME, 2.0)                       # same start pose for every gain
            time.sleep(0.3)
            t, ref, q, tau, aborted = run_path(bus, a.path_time)
            tq = torque_flags(bus)
            m = metrics(t, ref, q, tau)
            m["aborted"] = aborted
            results[g] = m
            print(f"  track err {m['track_mean']:.2f}d mean / {m['track_max']:.2f}d max   "
                  f"acc RMS {m['acc_rms']:.0f} d/s^2   jerk RMS {m['jerk_rms']/1000:.1f} kd/s^3   "
                  f"|tau|max {m['taumax']:.2f}   torque {tq}", flush=True)
            if any(v != 1 for v in tq):
                print("  !! OVERLOAD PROTECTION TRIPPED — re-energizing, stopping sweep", flush=True)
                bus.enable_torque()                    # goal := current, then torque on
                break
            if aborted:
                break
    finally:
        set_gain(bus, restore)
        goto(bus, HOME, 2.0)
        print(f"\n{label}-gain restored to {restore}; arm holding at home.", flush=True)
        bus.close()

    if len(results) > 1:
        print("\n=== summary ===", flush=True)
        for g, m in results.items():
            print(f"{label}={g:>3}: track {m['track_mean']:>5.2f}d/{m['track_max']:>5.2f}d   "
                  f"acc {m['acc_rms']:>5.0f} d/s^2   jerk {m['jerk_rms']/1000:>6.1f} kd/s^3   "
                  f"|tau|max {m['taumax']:>5.2f}{'   ABORTED' if m['aborted'] else ''}",
                  flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--port", default="/dev/cu.usbmodem5AA90245791")
    p.add_argument("--calib", default=str(PROJECT_ROOT / "so101_calib.json"))
    p.add_argument("--reg", choices=("p", "d"), default="p", help="which gain to sweep")
    p.add_argument("--gains", default=None,
                   help="comma list, run in order (default: p->8,16,32,48 / d->8,32,128)")
    p.add_argument("--path-time", type=float, default=6.0, help="s for the whole 6-segment path")
    p.add_argument("--restore-pgain", type=int, default=16)
    main(p.parse_args())
