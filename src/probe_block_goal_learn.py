"""Block-shift goal pursuit: give the FULL frozen act-stack a photo of the current
scene with ONE block teleported 5 cm, from the arm's CURRENT pose (pose component
~zero by construction), and measure whether the child MOVES THAT BLOCK toward the
photographed position over a long episode.

This is the behavioral bottom line the one-decision planner-visibility probe cannot
see: closed-loop replanning + dwell + latent feedback over `--budget` decisions.
Condition A pursues the shifted-block photo; condition B (control) pursues the
UNSHIFTED photo of the same state while we still measure block motion toward the
same phantom spot -- deliberateness = closure(A) - closure(B).
"""
import os, sys, json, argparse
os.environ.setdefault("MUJOCO_GL", "osmesa")
sys.path.insert(0, "/workspace/curious-robot")

import numpy as np
import torch
import mujoco
from types import SimpleNamespace
from env.mujoco_env import MujocoSO101Env
from src.train import encode_obs
from src.eval_goal_photo import build_wm, act_stack
from src.probe_planner_visibility import save_state, restore_state

ap = argparse.ArgumentParser()
ap.add_argument("--ckpts", required=True, help="comma-separated label=path pairs")
ap.add_argument("--scenes", type=int, default=6)
ap.add_argument("--budget", type=int, default=60)
ap.add_argument("--shift", type=float, default=0.05, help="block goal displacement (m, tangential)")
ap.add_argument("--seed", type=int, default=41)
ap.add_argument("--out", default="runs/probe_block_goal")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
device = torch.device("cuda")

ckpts = [kv.split("=", 1) for kv in args.ckpts.split(",")]
ck0 = torch.load(ckpts[0][1], map_location="cpu", weights_only=False)
a0 = SimpleNamespace(**ck0["args"])

env = MujocoSO101Env(frame_skip=a0.frame_skip, action_max=a0.action_max, encode_cam=a0.wm_cam,
                     safety_delta=a0.safety_delta, seed=args.seed, fixed_objects=False)
n_dof = env.n_dof


def teleport(xy):
    adr = env._object_qpos_addrs[0]; vadr = env._object_qvel_addrs[0]
    env.data.qpos[adr:adr + 2] = xy
    env.data.qpos[adr + 2] = env._object_resting_z(0)
    env.data.qvel[vadr:vadr + 6] = 0.0
    mujoco.mj_forward(env.model, env.data)


def settle(n_dec=3):
    for _ in range(n_dec * a0.action_block):
        env.step(np.zeros(n_dof, np.float32), render=False)


# ---------- stage scenes once (checkpoint-independent) ----------
scenes = []
for s in range(args.scenes):
    env.reset()
    teleport([0.25, 0.0])
    settle()
    obs = env._get_obs()
    snap = save_state(env)
    b0 = env.data.xpos[env._object_body_ids[0]][:2].copy()
    b_goal = b0 + np.array([0.0, args.shift])
    # goal photo: SAME arm pose, block shifted
    teleport(b_goal)
    goal_px = env._get_obs()["image"].copy()
    restore_state(env, snap)
    # control photo: same state, no shift
    ctrl_px = obs["image"].copy()
    scenes.append(dict(snap=snap, b0=b0, b_goal=b_goal, goal_px=goal_px, ctrl_px=ctrl_px,
                       prop=obs["proprio"].copy()))
    print(f"[scene {s}] block0 ({b0[0]:+.3f},{b0[1]:+.3f}) -> goal ({b_goal[0]:+.3f},{b_goal[1]:+.3f})", flush=True)

results = {}
for label, path in ckpts:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    wm, a = build_wm(ck, n_dof, device)
    a_dim = n_dof * a.action_block
    H = a.history_size
    rows = []
    for si, sc in enumerate(scenes):
        row = {}
        for cond, gpx in (("shift", sc["goal_px"]), ("ctrl", sc["ctrl_px"])):
            restore_state(env, sc["snap"])
            obs = env._get_obs()
            zstar = encode_obs(wm, gpx[None], sc["prop"][None], device)
            z = encode_obs(wm, obs["image"][None], obs["proprio"][None], device)
            hist_z = z.unsqueeze(0).repeat(H, 1, 1)
            hist_a = torch.zeros(H, 1, a_dim, device=device)
            gap0 = float(np.linalg.norm(sc["b0"] - sc["b_goal"]))
            min_gap, ncon, d_at_min = gap0, 0, None
            d0 = float((z - zstar).norm(dim=-1)[0])
            for t in range(args.budget):
                act, gd = act_stack(wm, a, hist_z, hist_a, z, zstar, device)
                hist_a = torch.cat([hist_a[1:], act.unsqueeze(0)], 0)
                subs = act[0].detach().cpu().numpy().reshape(a.action_block, n_dof)
                nc = 0
                for k, sub in enumerate(np.clip(subs, -1, 1)):
                    obs, info = env.step(sub.astype(np.float32), render=(k == a.action_block - 1))
                    nc += int(info["object_contacts"])
                ncon += nc
                z = encode_obs(wm, obs["image"][None], obs["proprio"][None], device)
                hist_z = torch.cat([hist_z[1:], z.unsqueeze(0)], 0)
                b = env.data.xpos[env._object_body_ids[0]][:2]
                gap = float(np.linalg.norm(b - sc["b_goal"]))
                if gap < min_gap:
                    min_gap, d_at_min = gap, float((z - zstar).norm(dim=-1)[0])
            row[cond] = dict(gap0=gap0, min_gap=min_gap, closure_mm=(gap0 - min_gap) * 1000,
                             contacts=ncon, d0=d0)
        row["deliberate_mm"] = row["shift"]["closure_mm"] - row["ctrl"]["closure_mm"]
        rows.append(row)
        print(f"[{label} scene {si}] shift: closure {row['shift']['closure_mm']:+.1f}mm con {row['shift']['contacts']:3d} d0 {row['shift']['d0']:.1f} | "
              f"ctrl: closure {row['ctrl']['closure_mm']:+.1f}mm con {row['ctrl']['contacts']:3d} | "
              f"deliberate {row['deliberate_mm']:+.1f}mm", flush=True)
    agg = dict(
        shift_closure_mm=float(np.median([r["shift"]["closure_mm"] for r in rows])),
        ctrl_closure_mm=float(np.median([r["ctrl"]["closure_mm"] for r in rows])),
        deliberate_mm=float(np.median([r["deliberate_mm"] for r in rows])),
        shift_contacts=float(np.median([r["shift"]["contacts"] for r in rows])),
        ctrl_contacts=float(np.median([r["ctrl"]["contacts"] for r in rows])),
        shift_d0=float(np.median([r["shift"]["d0"] for r in rows])),
        scenes=rows)
    results[label] = agg
    print(f"[{label}] MEDIANS: shift closure {agg['shift_closure_mm']:+.1f}mm (contacts {agg['shift_contacts']:.0f}) "
          f"vs ctrl {agg['ctrl_closure_mm']:+.1f}mm (contacts {agg['ctrl_contacts']:.0f}) -> "
          f"DELIBERATE {agg['deliberate_mm']:+.1f}mm  [goal shift {args.shift*1000:.0f}mm, d0 {agg['shift_d0']:.1f}]", flush=True)
    del wm, ck
    torch.cuda.empty_cache()

env.close()
with open(os.path.join(args.out, "results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"[done] {args.out}/results.json", flush=True)
