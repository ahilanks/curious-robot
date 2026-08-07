"""Mechanical action_max sweep — how much can dq^max grow before per-step locality breaks?

Context (2026-08-07): the oh_* overhead lineage trains at action_max=0.05 rad/joint/step
(action_block=5 -> <=0.25 rad commanded per decision). The user wants action_max as large
as possible, ideally unbounded. Since the adapter computes
    target = clip(q + a*action_max, joint_range)
"no action max at all" is well-defined: once action_max >= the largest joint span, the
joint-range clip binds instead and the action space is full-range absolute position
control. And the plant itself bounds per-step motion: the position servo torque-saturates
at |dq_err| ~ tau_max/kp ~ 6 mrad, so far targets are approached at a torque-limited slew
rate, not teleported to. This probe measures where REALIZED per-step motion plateaus.

Per rung (action_max value) x drive pattern (random block-actions like warmup/CEM
exploration; violent +/-1 square wave as the adversarial bound), over N env steps:
  * realized per-env-step |dq| (max joint) and EE displacement  -> the locality curve
  * realized per-decision (action_block=5) EE travel            -> what the WM must predict
  * realized/commanded ratio + joint-range clip-bind fraction   -> uncapped-equivalence
  * |qvel| p99, torque saturation frac                          -> plant regime
  * object knock-offs (mid-episode respawn teleports!) + contacts + object motion
  * non-finite qpos/qvel                                        -> physics stability

Env mirrors the oh_solo recipe: frame_skip=6, encode_cam=overhead_close, fixed_objects,
seed 0, no parent. render=False throughout (no images needed).

Writes runs/sim_scales/amax_sweep.json and prints a table.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from env.mujoco_env import MujocoSO101Env  # noqa: E402

ACTION_BLOCK = 5          # oh_solo decision = 5 sub-actions
STEPS = 1500              # env steps per (rung, pattern): 300 decisions
EPISODE_STEPS = 200       # oh_solo max_episode_steps: reset cadence matters for knock-off stats


def run_condition(env: MujocoSO101Env, kind: str, amax: float,
                  rng: np.random.Generator) -> dict:
    n = env.n_dof
    env.reset()
    knock = [0]
    orig_respawn = env._maybe_respawn_fallen

    def counted_respawn():
        knock[0] += sum(1 for i in range(env.n_objects)
                        if env.data.xpos[env._object_body_ids[i], 2] < env.table_drop_threshold)
        orig_respawn()

    env._maybe_respawn_fallen = counted_respawn

    dq_step, ee_step, ee_dec = [], [], []
    ratio, clip_bind = [], []
    qd_hist, sat_hist, contacts, obj_motion = [], [], 0, 0.0
    nonfinite = 0
    prev_ee = None
    prev_dec_ee = None
    for t in range(STEPS):
        if t > 0 and t % EPISODE_STEPS == 0:
            env.reset()
            prev_ee = None
            prev_dec_ee = None
        k = t % ACTION_BLOCK
        if kind == "random":                  # fresh U(-1,1) sub-action each env step (warmup/CEM-like)
            a = rng.uniform(-1, 1, n)
        elif kind == "violent":               # full-amplitude square wave per block
            a = (1.0 if (t // ACTION_BLOCK) % 2 == 0 else -1.0) * np.ones(n)
        else:
            raise ValueError(kind)
        q_now = env.data.qpos[:n].copy()
        raw = a * amax                        # pre-clip commanded delta
        target = env.adapter.ctrl_target(a.astype(np.float32), q_now)
        cmd = target - q_now                  # post-clip commanded delta
        clip_bind.append(float(np.mean(np.abs(cmd) < np.abs(raw) - 1e-9)))

        _, info = env.step(a.astype(np.float32), render=False)
        q_new = info["qpos"].astype(np.float64)
        dq = np.abs(q_new - q_now)
        dq_step.append(dq.max())
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.abs(q_new - q_now) / np.maximum(np.abs(cmd), 1e-9)
        ratio.append(float(np.median(np.clip(r, 0, 2))))
        ee = info["ee_pos"].astype(np.float64)
        if prev_ee is not None:
            ee_step.append(float(np.linalg.norm(ee - prev_ee)))
        if k == ACTION_BLOCK - 1:             # decision boundary: EE travel across the block
            if prev_dec_ee is not None:
                ee_dec.append(float(np.linalg.norm(ee - prev_dec_ee)))
            prev_dec_ee = ee
        prev_ee = ee
        qd_hist.append(np.abs(info["qvel"]).max())
        sat_hist.append(float((np.abs(info["applied_torque"]) > 0.95 * env.tau_max).mean()))
        contacts += int(info["object_contacts"])
        obj_motion += float(info["object_motion"])
        if not (np.isfinite(q_new).all() and np.isfinite(info["qvel"]).all()):
            nonfinite += 1

    env._maybe_respawn_fallen = orig_respawn
    dq_step, ee_step, qd_hist = map(np.asarray, (dq_step, ee_step, qd_hist))
    ee_dec = np.asarray(ee_dec) if ee_dec else np.asarray([0.0])
    return {
        "kind": kind, "action_max": amax, "steps": STEPS,
        "dq_step_p50": float(np.percentile(dq_step, 50)),
        "dq_step_p95": float(np.percentile(dq_step, 95)),
        "dq_step_max": float(dq_step.max()),
        "ee_step_p50": float(np.percentile(ee_step, 50)),
        "ee_step_p95": float(np.percentile(ee_step, 95)),
        "ee_step_max": float(ee_step.max()),
        "ee_dec_p50": float(np.percentile(ee_dec, 50)),
        "ee_dec_p95": float(np.percentile(ee_dec, 95)),
        "realized_over_cmd_p50": float(np.median(ratio)),
        "clip_bind_frac": float(np.mean(clip_bind)),
        "qvel_p99": float(np.percentile(qd_hist, 99)),
        "tau_sat_frac": float(np.mean(sat_hist)),
        "knockoffs": knock[0],
        "contacts": contacts,
        "object_motion": obj_motion,
        "nonfinite_steps": nonfinite,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="amax_sweep")
    args = ap.parse_args()

    probe_env = MujocoSO101Env(action_max=0.05, encode_cam="overhead_close",
                               frame_skip=6, fixed_objects=True, seed=args.seed)
    span = float((probe_env.ctrl_high - probe_env.ctrl_low).max())
    probe_env.close()
    uncap = float(np.ceil(span * 10) / 10)    # >= largest joint span: joint-range clip binds, not action_max
    rungs = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, uncap]
    print(f"[amax-sweep] largest joint span = {span:.3f} rad -> uncapped-equivalent rung = {uncap}", flush=True)

    rows = []
    for amax in rungs:
        env = MujocoSO101Env(action_max=amax, encode_cam="overhead_close",
                             frame_skip=6, fixed_objects=True, seed=args.seed)
        rng = np.random.default_rng(args.seed)
        for kind in ("random", "violent"):
            r = run_condition(env, kind, amax, rng)
            rows.append(r)
            print(f"[{kind:7s} amax={amax:<5g}] dq/step p50/p95/max = "
                  f"{r['dq_step_p50']:.4f}/{r['dq_step_p95']:.4f}/{r['dq_step_max']:.4f} rad  "
                  f"EE/step p95 = {r['ee_step_p95']*100:5.2f} cm  EE/dec p95 = {r['ee_dec_p95']*100:5.2f} cm  "
                  f"real/cmd = {r['realized_over_cmd_p50']:.2f}  clip-bind = {r['clip_bind_frac']:.2f}  "
                  f"|qd|p99 = {r['qvel_p99']:5.2f}  sat = {r['tau_sat_frac']:.2f}  "
                  f"knock = {r['knockoffs']:3d}  contact = {r['contacts']:4d}  nonfinite = {r['nonfinite_steps']}",
                  flush=True)
        env.close()

    out_dir = PROJECT_ROOT / "runs" / "sim_scales"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.tag}.json"
    path.write_text(json.dumps(rows, indent=2))
    print(f"[saved] {path}", flush=True)


if __name__ == "__main__":
    main()
