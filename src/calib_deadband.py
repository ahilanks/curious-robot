"""Tune the safety deadband delta FROM THE REAL ARM, against what YOU judge as bad.

The safety reward penalizes the per-joint hinge argument  arg = max(0, -tau * qddot)
only above delta. The benign side is already measured (deterministic policy runs:
true-dt arg P99.9 ~ 1.2-3.1, max 4.3 over ~2.9k substeps). This script collects the
MISSING side: events you deliberately cause and label as bad, so delta can be pinned
in the gap between "never penalize" and "always penalize".

Modes (all record (t, q, qd_raw, tau_raw) at the 30 ms control cadence, label windows,
and print a delta/lambda_safe suggestion at the end; npz saved for re-analysis):

  hold      Arm holds its pose, torque ON. You interact by hand: tap it, push a link,
            block a joint, twist the wrist — each in a capture window you rate 0-3.
                0 = totally fine   1 = acceptable   2 = bad (shouldn't do this)
                3 = very bad (never)
            Baseline windows ('b') record untouched holding as severity 0.
  reversal  Scripted stressor: square-wave reversals on one joint with escalating
            amplitude (each goal is paced to land in 30 ms, so amplitude == speed).
            You confirm each level before it runs and rate it after. This reproduces
            the worst thing the policy itself can do (violent direction flips).
  analyze   Re-run the analysis on saved npz file(s); pass them as positional args.
            Mix sessions freely (hold + reversal + multiple days).

Usage (arm powered, e-stop in reach; gains are read from the servos and recorded):
    export SOARM_PORT=/dev/cu.usbmodem5AA90245791 SOARM_CALIB=so101_calib.json
    python src/calib_deadband.py hold
    python src/calib_deadband.py reversal --joint 3 --max-amp 0.25
    python src/calib_deadband.py analyze runs/deadband_calib/*.npz

Pin delta on the TRUE-dt numbers and land the true-dt qddot fix in _control_step with
the same lineage — the runtime's hardcoded /0.030 inflates args by ~1.5x at the real
~44 ms loop, so a delta tuned here only means what it says once that fix is in.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

from env.hardware_env import (_default_bus, DT_SAFE, TAU_MAX,   # noqa: E402
                              JOINT_LOW, JOINT_HIGH, N_DOF)

TAU_CLIP = float(np.max(TAU_MAX))          # 3.35, all joints
SEV_HELP = "0=totally fine  1=acceptable  2=bad  3=very bad"
TEMP_ABORT = 55                            # C; stop a stressor session past this


class Recorder:
    """Paced sampler: bus.read() every DT_SAFE inside capture windows only, so
    consecutive samples within a window are valid finite-difference pairs."""

    def __init__(self, bus):
        self.bus = bus
        self.t, self.q, self.qd, self.tau = [], [], [], []
        self.windows = []                  # (i0, i1, severity, mode, note)

    def capture(self, secs, on_tick=None):
        """Record for `secs`; on_tick(k) may write goals (reversal mode). Returns
        (i0, i1) sample range."""
        i0 = len(self.t)
        n = max(int(round(secs / DT_SAFE)), 2)
        t_next = time.perf_counter()
        for k in range(n):
            if on_tick is not None:
                on_tick(k)
            t_next += DT_SAFE
            while time.perf_counter() < t_next - 2e-4:
                time.sleep(2e-4)
            q, qd, tau = self.bus.read()
            self.t.append(time.perf_counter())
            self.q.append(np.asarray(q, np.float32))
            self.qd.append(np.asarray(qd, np.float32))
            self.tau.append(np.asarray(tau, np.float32))
        return i0, len(self.t)

    def label(self, rng, severity, mode, note=""):
        self.windows.append((rng[0], rng[1], int(severity), mode, note))

    def save(self, out_dir, tag, gains):
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"deadband_{tag}_{int(time.time())}.npz"
        np.savez_compressed(
            path,
            t=np.asarray(self.t, np.float64), q=np.asarray(self.q),
            qd=np.asarray(self.qd), tau=np.asarray(self.tau),
            win_i0=np.asarray([w[0] for w in self.windows], np.int64),
            win_i1=np.asarray([w[1] for w in self.windows], np.int64),
            win_sev=np.asarray([w[2] for w in self.windows], np.int64),
            win_mode=np.asarray([w[3] for w in self.windows]),
            win_note=np.asarray([w[4] for w in self.windows]),
            dt_safe=np.float64(DT_SAFE), p_gain=gains[0], d_gain=gains[1])
        print(f"[calib] wrote {path} ({len(self.t)} samples, {len(self.windows)} windows)")
        return path


def read_gains(bus):
    """Record the servo P/D gains this session ran at (they shape tau response —
    a delta tuned at P8/D16 is not automatically valid at P16/D32)."""
    try:
        p = [bus.pk.read1ByteTxRx(bus.port, sid, bus.ADDR_P_GAIN)[0] for sid in bus.motor_ids]
        d = [bus.pk.read1ByteTxRx(bus.port, sid, bus.ADDR_D_GAIN)[0] for sid in bus.motor_ids]
        return np.asarray(p, np.int64), np.asarray(d, np.int64)
    except AttributeError:                  # mock bus
        return np.full(N_DOF, -1), np.full(N_DOF, -1)


def ask_severity():
    while True:
        s = input(f"  severity? {SEV_HELP} (or note text after the digit): ").strip()
        if s[:1] in "0123":
            return int(s[0]), s[1:].strip()
        print("  need 0/1/2/3")


def hinge_args(t, qd, tau):
    """Per-pair per-joint hinge argument, true-dt and /0.030 conventions."""
    dt = np.diff(t)[:, None]
    ok = (dt[:, 0] > 1e-4) & (dt[:, 0] < 5 * DT_SAFE)      # drop cross-gap pairs
    tau1 = np.clip(tau[1:], -TAU_CLIP, TAU_CLIP)
    dqd = np.diff(qd, axis=0)
    a_true = np.maximum(0.0, -tau1 * dqd / dt)[ok]
    a_030 = np.maximum(0.0, -tau1 * dqd / DT_SAFE)[ok]
    return a_true, a_030


def analyze(files):
    pools = {}                              # severity -> list of per-window MAX args
    flat = {}                               # severity -> all pair args (true-dt)
    for f in files:
        d = np.load(f, allow_pickle=True)
        t, qd, tau = d["t"], d["qd"], d["tau"]
        for i0, i1, sev in zip(d["win_i0"], d["win_i1"], d["win_sev"]):
            if i1 - i0 < 3:
                continue
            at, a0 = hinge_args(t[i0:i1], qd[i0:i1], tau[i0:i1])
            if not len(at):
                continue
            pools.setdefault(int(sev), []).append((at.max(), a0.max()))
            flat.setdefault(int(sev), []).append(at.max(1))   # worst joint per tick
    if not pools:
        print("[analyze] no labeled windows found")
        return
    print(f"\n{'sev':>3} {'wins':>5} | per-window MAX hinge arg (true-dt) "
          f"      min   P50   max  |  (/0.030) min   max")
    for sev in sorted(pools):
        m = np.asarray(pools[sev])
        print(f"{sev:>3} {len(m):>5} | {m[:,0].min():9.2f} {np.median(m[:,0]):6.2f} "
              f"{m[:,0].max():6.2f}  | {m[:,1].min():9.2f} {m[:,1].max():6.2f}")
    benign = np.concatenate([np.asarray(pools[s])[:, 0] for s in pools if s <= 1]) \
        if any(s <= 1 for s in pools) else None
    bad = np.concatenate([np.asarray(pools[s])[:, 0] for s in pools if s >= 2]) \
        if any(s >= 2 for s in pools) else None
    POLICY_CEIL = 4.3                       # max true-dt arg, ~2.9k policy substeps at P8/D16
                                            # (campaign gains; a P16/D32 bench measured 8.0)
    print(f"\n[ref] deterministic-policy benign ceiling (jerk benches): ~{POLICY_CEIL}")
    if bad is None:
        print("[analyze] no severity>=2 windows yet — collect bad events to pin delta")
        return
    floor = max(benign.max() if benign is not None else 0.0, POLICY_CEIL)
    ceil = bad.min()
    if ceil <= floor:
        print(f"[analyze] OVERLAP: worst benign {floor:.1f} >= mildest bad {ceil:.1f} — "
              f"delta alone can't separate; pick delta near {np.median(bad):.1f} (median bad) "
              f"and accept some missed mild events, or re-label borderline windows")
        delta = float(np.sqrt(max(floor, 1e-3) * np.median(bad)))
    else:
        delta = float(np.sqrt(floor * ceil))            # geometric midpoint of the gap
        print(f"[analyze] gap [{floor:.1f}, {ceil:.1f}] -> suggested delta = {delta:.1f} (true-dt)")
    # lambda_safe: make one median bad substep cost about one decision's curiosity term
    med_bad = float(np.median(bad))
    per_sub = max(med_bad - delta, 1e-3)                # |tau|/tau_max ~ 1 at impact
    CUR_TERM = 10.0                                     # lambda_cur*symlog(r_cur) observed ~4-11
    lam = CUR_TERM / per_sub
    print(f"[analyze] median bad exceedance {per_sub:.1f} -> lambda_safe ~ {lam:.1f} makes one "
          f"median bad substep cancel ~one decision's curiosity (~{CUR_TERM}); scale to taste. "
          f"Also re-pin probation_rsafe to ~ -(lambda_safe * per_sub) per flagged decision.")
    print("[analyze] REMINDER: valid only with the true-dt qddot fix landed, at the gains "
          "recorded in these sessions.")


def interp_to(bus, target, secs=1.5):
    q0 = bus.read()[0]
    n = max(int(secs / 0.1), 1)
    for k in range(1, n + 1):
        bus.write_goal(q0 + (np.asarray(target) - q0) * k / n)
        time.sleep(0.1)
        bus.read()                          # max_step_ticks clamp is vs the last READ


def main():
    p = argparse.ArgumentParser(description="Tune the safety deadband from the real arm")
    p.add_argument("mode", choices=["hold", "reversal", "analyze"])
    p.add_argument("files", nargs="*", help="npz files for analyze mode")
    p.add_argument("--joint", type=int, default=3, help="reversal joint index (3=wrist flex)")
    p.add_argument("--max-amp", type=float, default=0.25, help="reversal amplitude cap [rad]")
    p.add_argument("--window", type=float, default=4.0, help="hold capture window [s]")
    p.add_argument("--out", default="runs/deadband_calib")
    args = p.parse_args()

    if args.mode == "analyze":
        if not args.files:
            args.files = sorted(str(f) for f in (PROJECT_ROOT / args.out).glob("*.npz"))
        if not args.files:
            raise SystemExit("no npz files to analyze")
        analyze(args.files)
        return

    bus = _default_bus()
    rec = Recorder(bus)
    gains = read_gains(bus)
    temps = bus.read_temps()
    print(f"[calib] gains P={gains[0].tolist()} D={gains[1].tolist()} temps={temps.tolist()}")
    print(f"[calib] {args.mode} mode — e-stop in reach. Severity scale: {SEV_HELP}")

    if args.mode == "hold":
        print(f"[hold] arm holds its pose. Enter=capture {args.window:.0f}s window (do the "
              f"event inside it), b=10s untouched baseline, q=quit+save")
        while True:
            cmd = input("[hold] Enter/b/q > ").strip().lower()
            if cmd == "q":
                break
            if cmd == "b":
                print("  recording 10s baseline — don't touch the arm")
                rng = rec.capture(10.0)
                rec.label(rng, 0, "hold-baseline")
                continue
            print(f"  recording {args.window:.0f}s — do the event NOW")
            rng = rec.capture(args.window)
            sev, note = ask_severity()
            rec.label(rng, sev, "hold", note)
    else:                                   # reversal stressor
        j = int(args.joint)
        q0 = np.asarray(bus.read()[0], np.float64)
        amps = [a for a in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30) if a <= args.max_amp + 1e-9]
        half = 8                            # ticks per half-cycle (0.24 s) — 4 full reversals
        for amp in amps:
            lo = max(q0[j] - amp, JOINT_LOW[j] + 0.02)
            hi = min(q0[j] + amp, JOINT_HIGH[j] - 0.02)
            if input(f"[rev] joint {j} square-wave ±{amp:.2f} rad "
                     f"({lo:+.2f}..{hi:+.2f}), 4 cycles — Enter to run, q to stop > ") \
                    .strip().lower() == "q":
                break

            def tick(k, lo=lo, hi=hi):
                if k < 8 or k >= 8 + 8 * half:          # 8-tick lead-in/out at rest
                    return
                goal = q0.copy()
                goal[j] = hi if ((k - 8) // half) % 2 == 0 else lo
                bus.write_goal(np.clip(goal, JOINT_LOW, JOINT_HIGH))

            rng = rec.capture((16 + 8 * half) * DT_SAFE, on_tick=tick)
            interp_to(bus, q0, secs=1.0)
            sev, note = ask_severity()
            rec.label(rng, sev, f"reversal-j{j}-a{amp:.2f}", note)
            t_now = bus.read_temps()
            print(f"  temps {t_now.tolist()}")
            if np.max(t_now) >= TEMP_ABORT:
                print(f"[rev] temp >= {TEMP_ABORT}C — stopping")
                break

    temps1 = bus.read_temps()
    print(f"[calib] temps after: {temps1.tolist()}")
    path = rec.save(PROJECT_ROOT / args.out, args.mode, gains)
    analyze([str(path)])


if __name__ == "__main__":
    main()
