"""Planner-visibility probe: can the DEPLOYED planner (cem_plan, horizon=1,
terminal ||z-z*||^2 cost) SEE and CHOOSE contact today?

Motivated by the 2026-07-11 emergence directive: deliberate manipulation can only
emerge through CEM + latent distance, and CEM can only prefer a contact action if
the WM's one-step prediction of it moves z toward the goal. This probe measures
exactly that, per checkpoint, using the real cem_plan and the checkpoint's own
deployment CEM params.

Protocol per scene: drive the child to a PRE-CONTACT state s* (ChildPlow is used
ONLY to stage the scene and to supply one known contact-producing action -- pure
measurement, no training data is produced); the goal photo is the REAL outcome of
executing that action from s* (one-decision-reachable by construction, pose
component near zero). Then:
  A) SCORING: rank the true contact action among {pan-mirrored twin (matched arm
     motion, no contact), zero action, 200 CEM-style Gaussian candidates} by
     one-step predicted cost vs REAL executed cost (env rewound between candidates).
     Visibility = contact action ranked in the elite set by prediction.
  B) BEHAVIOR: run the real cem_plan from s* toward z*; execute its plan; did the
     planner's chosen action produce contact / move the block toward the goal?
"""
import os, sys, json, argparse
os.environ.setdefault("MUJOCO_GL", "osmesa")
sys.path.insert(0, "/workspace/curious-robot")

import numpy as np
import torch
import mujoco
from types import SimpleNamespace
from env.mujoco_env import MujocoSO101Env
from model.state_encoder import WorldModel, pred_dims_from_args
from src.train import to_norm_pixel, cem_plan
from src.smoke_child_sweep import ChildPlow, teleport_front

ap = argparse.ArgumentParser()
ap.add_argument("--ckpts", required=True, help="comma-separated label=path pairs")
ap.add_argument("--scenes", type=int, default=6)
ap.add_argument("--seed", type=int, default=21)
ap.add_argument("--out", default="runs/probe_planner_vis")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
device = torch.device("cuda")

ckpts = [kv.split("=", 1) for kv in args.ckpts.split(",")]
ck0 = torch.load(ckpts[0][1], map_location="cpu", weights_only=False)
a = SimpleNamespace(**ck0["args"])
Hb, N_DOF = a.history_size, 6

# ---------------- env state snapshot/rewind ----------------
def save_state(env):
    return dict(qpos=env.data.qpos.copy(), qvel=env.data.qvel.copy(),
                ctrl=env.data.ctrl.copy(), prev_ctrl=env._prev_ctrl.copy(),
                prev_qvel=env._prev_qvel.copy(), prev_obj=env._prev_obj_xpos.copy())

def restore_state(env, st):
    env.data.qpos[:] = st["qpos"]; env.data.qvel[:] = st["qvel"]
    env.data.ctrl[:] = st["ctrl"]
    env._prev_ctrl = st["prev_ctrl"].copy()
    env._prev_qvel = st["prev_qvel"].copy()
    env._prev_obj_xpos = st["prev_obj"].copy()
    mujoco.mj_forward(env.model, env.data)

def exec_decision(env, act30, render_last=True):
    """Execute one 30-dim decision (5x6 substeps, caller-clamped like deployment)."""
    subs = np.clip(act30.reshape(a.action_block, N_DOF), -1, 1)
    ncon, obs = 0, None
    for k, sub in enumerate(subs):
        obs, info = env.step(sub, render=(render_last and k == len(subs) - 1))
        ncon += int(info["object_contacts"])
    return obs, ncon

# ---------------- stage scenes ----------------
env = MujocoSO101Env(frame_skip=a.frame_skip, action_max=a.action_max, encode_cam=a.wm_cam,
                     safety_delta=a.safety_delta, seed=args.seed, fixed_objects=False)
rng = np.random.default_rng(7)
scenes = []
attempt = 0
while len(scenes) < args.scenes and attempt < args.scenes * 3:
    attempt += 1
    obs = env.reset(); teleport_front(env)
    obs = env._get_obs()
    plow = ChildPlow(env)
    px_hist, act_hist = [obs["image"]], []
    found = None
    for dec in range(40):
        snap = save_state(env)
        b_before = env.data.xpos[env._object_body_ids[0]][:2].copy()
        subs = []
        ncon = 0
        for k in range(a.action_block):
            q_kf = plow.keyframe()
            qpos = env.data.qpos[:N_DOF].copy()
            sub = np.clip((q_kf - qpos) / a.action_max, -1, 1).astype(np.float32)
            subs.append(sub)
            obs, info = env.step(sub, render=(k == a.action_block - 1))
            plow.observe(int(info["object_contacts"]))
            ncon += int(info["object_contacts"])
        act30 = np.concatenate(subs)
        disp = float(np.linalg.norm(env.data.xpos[env._object_body_ids[0]][:2] - b_before))
        if dec >= Hb and ncon > 0 and disp > 0.002:
            found = dict(snap=snap, contact_act=act30, goal_px=obs["image"].copy(),
                         goal_disp=disp, goal_ncon=ncon,
                         hist_px=[p.copy() for p in px_hist[-Hb:]],
                         hist_act=[q.copy() for q in (act_hist or [np.zeros(30, np.float32)])[-Hb:]],
                         b_star=b_before.copy())
            break
        px_hist.append(obs["image"].copy()); act_hist.append(act30)
    if found is None:
        continue
    while len(found["hist_act"]) < Hb:                      # left-pad with zeros if early
        found["hist_act"].insert(0, np.zeros(30, np.float32))
    while len(found["hist_px"]) < Hb:
        found["hist_px"].insert(0, found["hist_px"][0])

    # candidates: [true contact, pan-mirrored twin, zeros] + 200 CEM-style samples
    mirror = found["contact_act"].reshape(a.action_block, N_DOF).copy()
    mirror[:, 0] *= -1.0
    cands = [found["contact_act"], mirror.reshape(-1), np.zeros(30, np.float32)]
    cands += list(rng.normal(0.0, a.cem_init_std, size=(200, 30)).astype(np.float32))
    cands = np.stack(cands)

    # execute every candidate from s*, record REAL outcome (checkpoint-independent)
    post_px, post_con, post_bxy = [], [], []
    for ci, c in enumerate(cands):
        restore_state(env, found["snap"])
        obs, ncon = exec_decision(env, c)
        b_after = env.data.xpos[env._object_body_ids[0]][:2].copy()
        post_px.append(obs["image"].copy()); post_con.append(ncon); post_bxy.append(b_after)
    # candidate 0 IS the goal generator: its post block xy is the goal block position
    b_goal = post_bxy[0].copy()
    det_err = abs(float(np.linalg.norm(post_bxy[0] - found["b_star"])) - found["goal_disp"])
    restore_state(env, found["snap"])
    scenes.append(dict(**found, cands=cands, post_px=np.stack(post_px),
                       post_con=np.array(post_con), post_bxy=np.stack(post_bxy),
                       b_goal=b_goal, det_err=det_err))
    print(f"[scene {len(scenes)}] contact dec found: disp {found['goal_disp']*1000:.1f}mm "
          f"ncon {found['goal_ncon']} det_err {det_err*1000:.2f}mm", flush=True)
env.close()
assert scenes, "no pre-contact scenes staged"

# ---------------- per-checkpoint scoring + behavior ----------------
def build_wm(ck):
    aa = SimpleNamespace(**ck["args"])
    wm = WorldModel(n_dof=N_DOF, action_block=aa.action_block, history_size=aa.history_size,
                    dropout=aa.wm_dropout, use_proprio=not aa.no_proprio,
                    **pred_dims_from_args(ck["args"])).to(device)
    wm.load_state_dict(ck["wm"]); wm.eval()
    [p.requires_grad_(False) for p in wm.parameters()]
    return wm

@torch.no_grad()
def encode(wm, imgs):
    prop = np.zeros((len(imgs), 3 * N_DOF), np.float32)
    return wm.encode(to_norm_pixel(np.stack(imgs), device),
                     torch.as_tensor(prop, device=device).float())

results = {}
env2 = MujocoSO101Env(frame_skip=a.frame_skip, action_max=a.action_max, encode_cam=a.wm_cam,
                      safety_delta=a.safety_delta, seed=args.seed, fixed_objects=False)
for label, path in ckpts:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    wm = build_wm(ck)
    rows = []
    for si, sc in enumerate(scenes):
        z_hist = encode(wm, sc["hist_px"])                                  # (Hb, D)
        zg = encode(wm, [sc["goal_px"]])[0]                                 # (D,)
        z_post = encode(wm, list(sc["post_px"]))                            # (C, D)
        acts = torch.as_tensor(sc["cands"], device=device).float()          # (C, 30)
        C = acts.shape[0]
        hist_a = torch.as_tensor(np.stack(sc["hist_act"]), device=device).float()  # (Hb, 30)
        # one-step prediction for every candidate, exactly cem_plan's inner step
        z0 = z_hist.unsqueeze(0).expand(C, Hb, -1)
        a_seq = torch.cat([hist_a.unsqueeze(0).expand(C, Hb, -1), acts.unsqueeze(1)], 1)
        with torch.no_grad():
            znext = wm.predict(z0[:, -Hb:], wm.action_encoder(a_seq[:, -Hb:]))[:, -1]
        pred_cost = (znext - zg).pow(2).sum(-1).cpu().numpy()
        real_cost = (z_post - zg).pow(2).sum(-1).cpu().numpy()
        rank_pred = int((pred_cost < pred_cost[0]).sum())                   # 0 = best
        rank_real = int((real_cost < real_cost[0]).sum())
        rho = float(np.corrcoef(np.argsort(np.argsort(pred_cost)),
                                np.argsort(np.argsort(real_cost)))[0, 1])   # spearman
        # behavioral: the real planner
        plan = cem_plan(wm, z_hist.unsqueeze(1), hist_a.unsqueeze(1), zg.unsqueeze(0),
                        K=a.cem_samples, iters=a.cem_iters, elite=a.cem_elites,
                        init_std=a.cem_init_std, horizon=1, device=device,
                        gamma=a.cem_gamma, mppi_temp=a.cem_mppi_temp,
                        early_stop_tol=a.cem_early_stop,
                        early_stop_min_iters=a.cem_early_stop_min_iters)
        # snapshot carries the FULL qpos/qvel (arm + every block), so restoring onto
        # env2's fresh reset reproduces the staged scene exactly
        env2.reset(); restore_state(env2, sc["snap"])
        _, plan_ncon = exec_decision(env2, plan[0, 0].float().cpu().numpy())
        b_after = env2.data.xpos[env2._object_body_ids[0]][:2].copy()
        # directional: did the plan move the block TOWARD its goal position?
        gap0 = float(np.linalg.norm(sc["b_star"] - sc["b_goal"]))
        plan_prog = gap0 - float(np.linalg.norm(b_after - sc["b_goal"]))
        cand_prog = gap0 - np.linalg.norm(sc["post_bxy"][3:] - sc["b_goal"][None], axis=1)
        rows.append(dict(rank_pred=rank_pred, rank_real=rank_real, rho=rho,
                         elite=rank_pred < a.cem_elites,
                         mirror_gap=float(pred_cost[0] - pred_cost[1]),
                         base_contact_frac=float((sc["post_con"][3:] > 0).mean()),
                         base_prog_mm=float(np.median(cand_prog)) * 1000,
                         plan_contact=int(plan_ncon > 0),
                         plan_prog_mm=plan_prog * 1000))
    agg = dict(
        rank_pred_med=float(np.median([r["rank_pred"] for r in rows])),
        rank_real_med=float(np.median([r["rank_real"] for r in rows])),
        elite_frac=float(np.mean([r["elite"] for r in rows])),
        rho_med=float(np.median([r["rho"] for r in rows])),
        mirror_gap_med=float(np.median([r["mirror_gap"] for r in rows])),
        base_contact_frac=float(np.mean([r["base_contact_frac"] for r in rows])),
        base_prog_mm_med=float(np.median([r["base_prog_mm"] for r in rows])),
        plan_contact_frac=float(np.mean([r["plan_contact"] for r in rows])),
        plan_prog_mm_med=float(np.median([r["plan_prog_mm"] for r in rows])),
        scenes=rows)
    results[label] = agg
    print(f"[{label}] contact-action pred-rank med {agg['rank_pred_med']:.0f}/203 "
          f"(real-rank med {agg['rank_real_med']:.0f}) elite {agg['elite_frac']:.0%} "
          f"rho {agg['rho_med']:.2f} mirror-gap {agg['mirror_gap_med']:+.1f} | "
          f"CEM plan: contact {agg['plan_contact_frac']:.0%} (base {agg['base_contact_frac']:.0%}) "
          f"goal-progress {agg['plan_prog_mm_med']:+.1f}mm (base {agg['base_prog_mm_med']:+.1f}mm)", flush=True)
    del wm, ck
    torch.cuda.empty_cache()
env2.close()

with open(os.path.join(args.out, "results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"[done] {args.out}/results.json", flush=True)
