"""Bench: ground the TorqueGuard cap (so101_calib.json `current_cap_a`) in THIS arm's
measured current draw, then optionally live-fire the guard end-to-end.

WHY. The 1.5 A cap shipped as a first guess. The guard only protects if the cap sits in
the gap between what the arm draws in NORMAL motion (must not false-trip) and what it
draws in a genuine FIGHT (must trip). Neither number has been measured on this arm —
the 2026-06 bench only established that a real press pegs the tau clip (>= ~0.33 A);
STS3215 spec stall is ~2.7 A. This script measures both ends, supervised.

PHASES (all at the low-tension HOME pose, inside the safe envelope):
  A FREE    guard in measure-only mode; gentle min-jerk segments between envelope-safe
            waypoints; report per-joint |current| p50/p95/max -> the false-trip floor.
  B BLOCK   the arm slowly oscillates lift+elbow while YOU grip the forearm and BLOCK
            the motion for --resist-secs; report the draw -> the fight ceiling. This is
            the deploy failure mode itself (servo drives toward a goal it cannot reach).
            NOT a static-hold resist: at P=16 a held pose yields compliantly under hand
            pressure at idle-level current — that measures nothing (first-bench lesson).
  C FIRE    (--fire) re-arm the guard at --cap (or the suggested cap), block again:
            expect per-joint PIN prints, then the leaky bucket CUTS torque and raises
            TorqueLimitExceeded; the script catches it, verifies Torque_Enable==0 on
            every servo, and reports PASS. The arm goes LIMP at HOME — keep hands clear
            of pinch points and let it settle.

    python src/bench_torqueguard.py --port $SOARM_PORT              # measure A+B, suggest cap
    python src/bench_torqueguard.py --port $SOARM_PORT --fire       # + live-fire at the suggestion

SUPERVISED ONLY. E-stop within reach. Nothing here is a training entry point.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from env.hardware_env import (FeetechBus, TorqueGuard, TorqueLimitExceeded,  # noqa: E402
                              HOME, clip_safe, DT_SAFE)
from src.bench_dgain_wide import min_jerk                                    # noqa: E402

JOINTS = ["pan", "lift", "elbow", "wrist", "roll", "grip"]


def drive(bus: FeetechBus, target: np.ndarray, seg_time: float, currents: list) -> None:
    """Min-jerk move to `target` (envelope-clipped), sampling |current| [A] every 30 ms."""
    q0 = bus.read()[0].astype(np.float64)
    target = clip_safe(np.asarray(target, np.float64).copy())
    t0 = time.perf_counter()
    while True:
        now = time.perf_counter() - t0
        bus.write_goal(q0 + (target - q0) * min_jerk(now / seg_time))
        bus.read()
        currents.append(bus._last_cur_a.copy())
        if now >= seg_time:
            return
        time.sleep(max(0.0, DT_SAFE - (time.perf_counter() - t0 - now)))


def blocked_moves(bus: FeetechBus, secs: float, currents: list, announce: str = "") -> None:
    """The deploy failure mode, measurable by hand: the arm slowly oscillates lift+elbow
    (envelope-safe, ~0.26 rad/s peak) while YOU block the forearm — the servo drives toward
    a goal it cannot reach and the current climbs. A static soft hold (P=16) is NOT a fight:
    it yields compliantly under hand pressure and draws idle-level current — the first
    version of this bench measured exactly that and grounded nothing. Prints a per-second
    max-|current| readout so the resister can SEE the draw respond and push accordingly;
    with `announce` it also prints live guard pin state (--fire phase)."""
    q0 = bus.read()[0].astype(np.float64)
    t0 = time.perf_counter()
    last_pin, last_sec = "", -1
    while (now := time.perf_counter() - t0) < secs:
        phase = np.sin(2.0 * np.pi * now / 6.0)                     # 6 s period, gentle
        bus.write_goal(clip_safe(q0 + np.array([0, 0.12, 0.25, 0, 0, 0]) * phase))
        bus.read()                                    # guard runs HERE (pin/cut like a real run)
        currents.append(bus._last_cur_a.copy())
        sec = int(now)
        if sec != last_sec:
            recent = np.asarray(currents[-33:])                     # ~ the last second
            print(f"  t={sec:2d}s  max|A| {recent.max():.2f}", flush=True)
            last_sec = sec
        pin = ",".join(JOINTS[i] for i in np.flatnonzero(bus._pin_mask)) or "-"
        if pin != last_pin and announce:
            print(f"  [{announce}] t={now:4.1f}s pinned: {pin}   ({bus.guard.report()})", flush=True)
            last_pin = pin
        time.sleep(max(0.0, DT_SAFE - ((time.perf_counter() - t0) % DT_SAFE)))


def stats(tag: str, currents: list) -> np.ndarray:
    a = np.asarray(currents)                          # (N, 6) amps
    print(f"\n[{tag}] {len(a)} samples, |current| per joint [A]:")
    print(f"  {'joint':6s} {'p50':>6s} {'p95':>6s} {'max':>6s}")
    for j, name in enumerate(JOINTS):
        print(f"  {name:6s} {np.percentile(a[:, j], 50):6.3f} "
              f"{np.percentile(a[:, j], 95):6.3f} {a[:, j].max():6.3f}")
    print(f"  ALL max = {a.max():.3f} A")
    return a


def main(args: argparse.Namespace) -> None:
    calib = json.loads(Path(args.calib).read_text())
    trip_steps = int(calib.get("cap_trip_steps", 5))
    calib = {k: v for k, v in calib.items() if k != "current_cap_a"}   # measure phases: guard off
    bus = FeetechBus(port=args.port, current_cap_a=0.0, **calib)
    print(f"[bench] guard in measure-only mode ({bus.guard.report()}); homing...")
    free: list = []
    drive(bus, HOME, 2.5, [])                                          # settle at HOME, unrecorded

    # A) free-motion floor: gentle sweeps inside the envelope
    for tgt in (HOME + [0.25, 0, 0, 0, 0, 0], HOME + [-0.25, 0.10, 0, 0, 0, 0],
                HOME + [0, 0, 0.25, 0.20, 0, 0], HOME + [0, 0, -0.30, -0.30, 0.5, -0.3], HOME):
        drive(bus, tgt, 2.0, free)
    a_free = stats("A free motion", free)

    # B) supervised fight ceiling: BLOCK the moving arm (not resist a static hold)
    input(f"\n[B] For {args.resist_secs:.0f}s the arm will slowly swing lift+elbow. GRIP the "
          f"forearm and BLOCK the motion — hold it still against the drive, hard, the whole "
          f"time. Watch the live max|A| readout climb while you block. ENTER to start...")
    resist: list = []
    blocked_moves(bus, args.resist_secs, resist)
    a_res = stats("B blocked moves", resist)

    # fight ceiling = p95 of the per-step arm max: robust to the idle samples around grips,
    # unlike a median (the first bench's median landed IN the idle floor and grounded nothing)
    free_max, fight = a_free.max(), float(np.percentile(a_res.max(axis=1), 95))
    suggest = args.cap if args.cap > 0 else round(max(1.3 * free_max,
                                                      0.5 * (free_max + fight)), 2)
    print(f"\n[verdict] free-motion max {free_max:.2f} A | blocked-move p95-of-max {fight:.2f} A "
          f"-> suggested current_cap_a = {suggest} (trip_steps {trip_steps})")
    if fight < 1.5 * free_max:
        print("[verdict] WARNING: no clean separation between free motion and the blocked "
              "fight — block harder (stop the link fully), or the cap cannot be grounded. "
              "NOT live-firing.")
        bus.disable_torque()
        return

    # C) optional live-fire: prove pin -> cut -> limp, end to end
    if args.fire:
        bus.guard = TorqueGuard(cap_a=suggest, trip_steps=trip_steps, n_dof=len(JOINTS))
        input(f"\n[C] LIVE-FIRE at cap {suggest} A: block the swinging forearm again — expect "
              f"pins, then a loud cut and a LIMP arm. Hands clear of pinch points. ENTER...")
        try:
            blocked_moves(bus, 60.0, [], announce="fire")
            print("[C] no trip in 60 s — resist harder or lower --cap; leaving torque ON.")
        except TorqueLimitExceeded as e:
            print(f"[C] CUT: {e}")
            regs = [bus.pk.read1ByteTxRx(bus.port, sid, bus.ADDR_TORQUE_ENABLE)[0]
                    for sid in bus.motor_ids]
            assert not any(regs), f"Torque_Enable regs not all 0 after cut: {regs}"
            print(f"[C] PASS — Torque_Enable=0 on all servos, arm limp. "
                  f"Set current_cap_a={suggest} in {args.calib}.")
            return
    bus.disable_torque()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--port", default=os.environ.get("SOARM_PORT"), required=os.environ.get("SOARM_PORT") is None)
    p.add_argument("--calib", default=os.environ.get("SOARM_CALIB", "so101_calib.json"))
    p.add_argument("--resist-secs", type=float, default=10.0)
    p.add_argument("--cap", type=float, default=0.0, help="override the suggested cap for --fire")
    p.add_argument("--fire", action="store_true", help="live-fire the guard at the suggested/--cap value")
    main(p.parse_args())
