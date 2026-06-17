"""Measure the sim-side scales that gate the 2026-06-12 experiment campaign, on the
kp=499.11 (P8-matched) plant — the logistics caveat: "sim-side arg distribution at
delta=9 UNVERIFIED — measure sim benign args before a long sim pretrain".

Per condition (smooth sinusoid / policy-like dither / uniform random / violent
square-wave, at action_max 0.1 and 0.3) this reports:
  * hinge-arg (-tau*qddot) per-joint percentiles + frac of joint-samples above
    delta in {1, 5, 9, 15}  -> does delta=9 stay silent on benign sim motion?
  * projected r_safe at each delta and the lambda_safe=2.2 reward contribution
    vs a cur_contrib ~ 10 yardstick                  -> freeze risk check
  * |tau| mean and saturation fraction (|tau| > 0.95 tau_max)   -> kp regime
  * energy mean_i |tau_i * qd_i|                     -> sets --w-energy scale
  * action-rate mean_dim (a_t - a_{t-1})^2 across env steps -> sets --w-action-rate

Writes a JSON summary to runs/sim_scales/<tag>.json and prints a table.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from env.mujoco_env import MujocoSO101Env  # noqa: E402

DELTAS = (1.0, 5.0, 9.0, 15.0)
LAMBDA_SAFE = 2.2
CUR_YARDSTICK = 10.0          # typical lambda_cur*log1p(r_cur) per decision in past runs


def run_condition(env: MujocoSO101Env, kind: str, n_steps: int, action_max: float,
                  rng: np.random.Generator) -> dict:
    n = env.n_dof
    env.reset()
    args_hist, tau_hist, qd_hist, energy_hist, rate_hist = [], [], [], [], []
    prev_a = np.zeros(n, dtype=np.float64)
    phase = rng.uniform(0, 2 * np.pi, n)
    freq = rng.uniform(0.05, 0.15, n)            # cycles per env-step: slow reach-like sweeps
    for t in range(n_steps):
        if kind == "smooth":                      # gentle sinusoidal joint sweeps
            a = 0.6 * np.sin(2 * np.pi * freq * t + phase)
        elif kind == "dither":                    # safe15-style reversal dither: ~70% sign flips
            flip = rng.random(n) < 0.7
            a = np.where(flip, -np.sign(prev_a + 1e-9), np.sign(prev_a + 1e-9)) \
                * rng.uniform(0.4, 1.0, n)
        elif kind == "random":                    # uniform policy-free exploration (old warmup)
            a = rng.uniform(-1, 1, n)
        elif kind == "violent":                   # worst case: full-amplitude square wave
            a = (1.0 if (t // 3) % 2 == 0 else -1.0) * np.ones(n)
        else:
            raise ValueError(kind)
        obs, info = env.step(a.astype(np.float32))
        tau = info["applied_torque"].astype(np.float64)
        qd = info["qvel"].astype(np.float64)
        qd_prev = info["qvel_prev"].astype(np.float64)
        qdd = (qd - qd_prev) / env.dt_safe
        args_hist.append(-tau * qdd)              # the per-joint hinge argument
        tau_hist.append(np.abs(tau))
        qd_hist.append(np.abs(qd))
        energy_hist.append(np.abs(tau * qd).mean())
        rate_hist.append(((a - prev_a) ** 2).mean())
        prev_a = a
    args = np.asarray(args_hist)                  # (T, n)
    taus = np.asarray(tau_hist)
    qds = np.asarray(qd_hist)
    tau_max = env.tau_max.astype(np.float64)
    out = {
        "kind": kind, "action_max": action_max, "steps": n_steps,
        "arg_p50": float(np.percentile(args, 50)),
        "arg_p90": float(np.percentile(args, 90)),
        "arg_p99": float(np.percentile(args, 99)),
        "arg_max": float(args.max()),
        "tau_mean": float(taus.mean()),
        "tau_sat_frac": float((taus > 0.95 * tau_max[None, :]).mean()),
        "qd_mean": float(qds.mean()),
        "energy_mean": float(np.mean(energy_hist)),
        "action_rate_mean": float(np.mean(rate_hist)),
    }
    for d in DELTAS:
        fight = np.maximum(0.0, args - d)
        weight = taus / tau_max[None, :]
        r_safe = -(weight * fight).sum(axis=1)            # per env-step, summed over joints
        out[f"frac_fire_d{d:g}"] = float((args > d).mean())
        out[f"r_safe_d{d:g}"] = float(r_safe.mean())
        out[f"contrib_d{d:g}"] = float(LAMBDA_SAFE * r_safe.mean())
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--tag", default="kp499")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = []
    for action_max in (0.1, 0.3):
        env = MujocoSO101Env(action_max=action_max, seed=args.seed)
        rng = np.random.default_rng(args.seed)
        for kind in ("smooth", "dither", "random", "violent"):
            r = run_condition(env, kind, args.steps, action_max, rng)
            rows.append(r)
            print(f"[{kind:7s} a_max={action_max}] arg p50/p90/p99/max = "
                  f"{r['arg_p50']:7.2f}/{r['arg_p90']:7.2f}/{r['arg_p99']:7.2f}/{r['arg_max']:8.2f}  "
                  f"sat={r['tau_sat_frac']:.2f} |tau|={r['tau_mean']:.2f} "
                  f"E={r['energy_mean']:.3f} rate={r['action_rate_mean']:.3f}", flush=True)
            for d in DELTAS:
                print(f"    delta={d:>4g}: fires on {100*r[f'frac_fire_d{d:g}']:5.1f}% joint-samples, "
                      f"r_safe={r[f'r_safe_d{d:g}']:8.3f}, "
                      f"lam22-contrib={r[f'contrib_d{d:g}']:8.2f} (cur yardstick ~{CUR_YARDSTICK})",
                      flush=True)
        env.close()
    out_dir = PROJECT_ROOT / "runs" / "sim_scales"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.tag}.json"
    path.write_text(json.dumps(rows, indent=2))
    print(f"[saved] {path}", flush=True)


if __name__ == "__main__":
    main()
