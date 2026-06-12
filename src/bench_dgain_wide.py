"""Bench: D-gain UP-sweep (default 32 -> 128 -> 254) under a wide multi-waypoint motion.

The 2026-06-09 bench swept D DOWN (32->16->8->4): no closed-loop effect — gear friction
owns the damping and the step probe showed ZERO overshoot at any D (nothing to damp).
This probes the other direction: is MORE velocity damping feelable at all? Expected
signature, if any: added LAG at fast arrivals (later settle, larger arrival error),
NOT reduced overshoot (there is none to reduce).

Drives the SAME 6-waypoint min-jerk cycle at each D and prints per-gain arrival /
overshoot / settle / lag metrics, so the gains are directly comparable. Guards:
  - base pan (J1) kept in [+0.05, +0.70] rad — NEGATIVE pan winds the wrist-cam cable
    (load 546 / ~100 LSB current AT REST at -0.48 rad, bench 2026-06-09)
  - pan strain check at every hold: sustained rest-current on servo 1 aborts the cycle
  - temp check between phases (abort > 48 degC; daemon's gate is 50)
  - D restored to --restore-dgain (default 32) on exit — P/D live in EEPROM and the
    campaign config is frozen at P16/D32 for sim parity. (collect_daemon's FeetechBus
    constructor would self-heal it on next boot anyway, but don't rely on that.)

    python src/bench_dgain_wide.py --port /dev/cu.usbmodem5AA90245791
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

JOINTS = ("pan", "lift", "elbow", "wrist", "roll", "grip")
PAN_SAFE = (0.05, 0.70)          # cable wind-up: stay positive, near the low-tension home
DEG = 180.0 / np.pi

# Wide cycle around the calibrated zero (upper arm vertical, forearm forward).
# Negative lift/elbow/wrist head toward the gravity-stable fold (known-safe interior);
# positive excursions are kept moderate so the gripper stays well above the table.
WAYPOINTS = np.array([
    #  pan   lift  elbow  wrist  roll  grip
    [0.35,  0.00,  0.00,  0.00,  0.00, 0.50],   # home: zero-ish, mid-pan, half-open
    [0.70, -0.30, -0.50, -0.50,  0.00, 0.20],   # pan to right edge, forearm raised
    [0.05,  0.20,  0.40,  0.30,  0.50, 0.80],   # pan to left edge, gentle forward reach
    [0.45, -0.50,  0.60, -0.80, -0.60, 0.00],   # lean back, forearm down, wrist up
    [0.20, -0.90, -0.90, -0.60,  0.30, 0.60],   # partway toward the fold
    [0.35,  0.00,  0.00,  0.00,  0.00, 0.50],   # back home
], dtype=np.float64)


def min_jerk(tau: float) -> float:
    tau = min(max(tau, 0.0), 1.0)
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


def set_dgain(bus: FeetechBus, val: int) -> None:
    """Live D-gain write (reg 22) with torque ON — verified path (32->8->32 toggle,
    2026-06-09). Read back to confirm; a silent miss would invalidate the whole A/B."""
    for sid in bus.motor_ids:
        bus.pk.write1ByteTxRx(bus.port, sid, bus.ADDR_D_GAIN, int(val))
    time.sleep(0.1)            # EEPROM commit: an immediate readback drops (reads as 0)
    rb = []
    for sid in bus.motor_ids:  # comm-checked read with retry — a dropped read is not a 0
        for _ in range(4):
            v, c, _ = bus.pk.read1ByteTxRx(bus.port, sid, bus.ADDR_D_GAIN)
            if c == bus._OK:
                break
            time.sleep(0.05)
        rb.append(v if c == bus._OK else None)
    if any(r != int(val) for r in rb):
        raise RuntimeError(f"D-gain readback {rb} != {val}")


def goto(bus: FeetechBus, target: np.ndarray, seg_time: float):
    """Min-jerk move from the current pose to `target`; returns (t, q, tau) motion log."""
    q0 = bus.read()[0].astype(np.float64)
    log_t, log_q, log_tau = [], [], []
    t0 = time.perf_counter()
    while True:
        now = time.perf_counter() - t0
        s = min_jerk(now / seg_time)
        bus.write_goal(q0 + (target - q0) * s)
        q, _, tau = bus.read()
        log_t.append(now); log_q.append(q); log_tau.append(tau)
        if now >= seg_time:
            return np.asarray(log_t), np.asarray(log_q), np.asarray(log_tau)
        time.sleep(max(0.0, 0.030 - (time.perf_counter() - t0 - now)))


def hold(bus: FeetechBus, hold_time: float):
    """Read-only settle watch at the last commanded goal."""
    log_t, log_q, log_tau = [], [], []
    t0 = time.perf_counter()
    while True:
        now = time.perf_counter() - t0
        q, _, tau = bus.read()
        log_t.append(now); log_q.append(q); log_tau.append(tau)
        if now >= hold_time:
            return np.asarray(log_t), np.asarray(log_q), np.asarray(log_tau)
        time.sleep(0.030)


def run_cycle(bus: FeetechBus, seg_time: float, hold_time: float, cur_lsb_per_nm: float):
    """One pass over WAYPOINTS; per-segment metrics (deg / ms / N*m)."""
    rows = []
    for k in range(1, len(WAYPOINTS)):
        target, prev = WAYPOINTS[k], WAYPOINTS[k - 1]
        mt, mq, mtau = goto(bus, target, seg_time)
        ht, hq, htau = hold(bus, hold_time)

        move = target - prev
        moving = np.abs(move) * DEG > 5.0                    # joints that actually travel
        err_h = (hq - target) * DEG                          # hold error trace (T, 6)
        arrive = float(np.abs(err_h[0][moving]).max())
        over = float(max(0.0, (err_h[:, moving] * np.sign(move)[moving]).max()))
        settle_dev = float(np.abs(err_h[ht >= hold_time - 0.3][:, moving]).mean())
        within = np.abs(err_h[:, moving]).max(1) < 3.0
        t_settle = float(ht[within][0] * 1000) if within.any() else float(hold_time * 1000)

        # cable-strain guard: pan rest-current over the back half of the hold
        pan_rest_lsb = float(np.abs(htau[ht >= hold_time / 2, 0]).mean() * cur_lsb_per_nm)
        rows.append(dict(seg=k, mover=JOINTS[int(np.abs(move).argmax())],
                         move_deg=float(np.abs(move).max() * DEG), arrive_deg=arrive,
                         over_deg=over, settle_deg=settle_dev, t_settle_ms=t_settle,
                         taumax=float(np.abs(np.concatenate([mtau, htau])).max()),
                         pan_rest_lsb=pan_rest_lsb))
        if pan_rest_lsb > 80.0:                              # strain signature: high current AT REST
            print(f"  !! pan rest-current {pan_rest_lsb:.0f} LSB at seg {k} — "
                  f"cable strain, aborting cycle", flush=True)
            break
    return rows


def main(a):
    import json
    calib = json.loads(Path(a.calib).read_text())
    gains = [int(g) for g in a.gains.split(",")]
    cur_lsb_per_nm = 1.0 / (FeetechBus._AMPS_PER_LSB * float(calib.get("kt", 1.0)))

    # Clip every waypoint to the joint limits + the pan cable-safe box, whatever was typed.
    np.clip(WAYPOINTS, JOINT_LOW, JOINT_HIGH, out=WAYPOINTS)
    np.clip(WAYPOINTS[:, 0], PAN_SAFE[0], PAN_SAFE[1], out=WAYPOINTS[:, 0])

    bus = FeetechBus(port=a.port, **{k: calib[k] for k in
                                     ("offsets_ticks", "signs", "vel_scale", "p_gain",
                                      "d_gain", "goal_speed", "pace_dt", "acceleration", "kt")
                                     if k in calib})
    all_rows = {}
    try:
        for g in gains:
            temps = bus.read_temps()
            print(f"\n=== D-gain {g} ===  temps {temps.tolist()}", flush=True)
            if temps.max() > 48.0:
                print("  !! temps too high — stopping the sweep", flush=True)
                break
            set_dgain(bus, g)
            time.sleep(0.5)
            rows = run_cycle(bus, a.seg_time, a.hold_time, cur_lsb_per_nm)
            all_rows[g] = rows
            hdr = f"{'seg':>3} {'mover':>6} {'move':>6} {'arrive':>7} {'over':>6} " \
                  f"{'settle':>7} {'t_set':>7} {'|tau|':>6} {'panI':>5}"
            print(hdr + "\n" + "-" * len(hdr), flush=True)
            for r in rows:
                print(f"{r['seg']:>3} {r['mover']:>6} {r['move_deg']:>5.1f}d "
                      f"{r['arrive_deg']:>6.2f}d {r['over_deg']:>5.2f}d "
                      f"{r['settle_deg']:>6.2f}d {r['t_settle_ms']:>5.0f}ms "
                      f"{r['taumax']:>6.2f} {r['pan_rest_lsb']:>5.0f}", flush=True)
    finally:
        set_dgain(bus, a.restore_dgain)
        print(f"\nD-gain restored to {a.restore_dgain} (read-back OK); arm left holding.",
              flush=True)
        bus.close()

    if len(all_rows) > 1:
        print("\n=== per-gain means (moving joints only) ===", flush=True)
        for g, rows in all_rows.items():
            if rows:
                print(f"D={g:>3}: arrive {np.mean([r['arrive_deg'] for r in rows]):.2f}d  "
                      f"overshoot {np.mean([r['over_deg'] for r in rows]):.2f}d  "
                      f"settle {np.mean([r['settle_deg'] for r in rows]):.2f}d  "
                      f"t_settle {np.mean([r['t_settle_ms'] for r in rows]):.0f}ms", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--port", default="/dev/cu.usbmodem5AA90245791")
    p.add_argument("--calib", default=str(PROJECT_ROOT / "so101_calib.json"))
    p.add_argument("--gains", default="32,128,254", help="comma list, run in order")
    p.add_argument("--seg-time", type=float, default=1.2, help="s per waypoint move")
    p.add_argument("--hold-time", type=float, default=0.8, help="s settle watch per waypoint")
    p.add_argument("--restore-dgain", type=int, default=32)
    main(p.parse_args())
