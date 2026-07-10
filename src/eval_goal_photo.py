"""Goal-photo evaluation: can the frozen planner recreate a randomly photographed pose?

Protocol (per goal): reset -> smooth random-walk `--walk` decisions -> snapshot the wrist
view as the GOAL PHOTO o* (+ ground-truth qpos*) -> reset elsewhere -> run the frozen
CEM act stack (identical to train.py: replan-every-step, dwell shrink/hold, same eps)
toward z* = encode(o*) for `--budget` decisions. Score latent arrival (min ||z-z*|| < eps,
the training criterion) AND physical inf-norm joint error at the closest-approach step.
Goals bin by INITIAL latent distance -> the empirical reachability-vs-distance curve that
the curriculum's d budget is supposed to buy. All constants come from the ckpt's saved
args, so the act stack cannot drift from what the checkpoint was trained with.

Usage:
  python src/eval_goal_photo.py --ckpt <path.pt> [--goals 24] [--budget 120] [--walk 40]
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from env.parallel_env import SubprocVectorMujocoEnv                    # noqa: E402
from model.state_encoder import WorldModel, pred_dims_from_args        # noqa: E402
from src.train import cem_plan, encode_obs, scrub_torque_obs           # noqa: E402


def build_wm(ckpt, n_dof, device):
    a = SimpleNamespace(**ckpt["args"])
    wm = WorldModel(n_dof=n_dof, action_block=a.action_block,
                    history_size=a.history_size, dropout=a.wm_dropout,
                    use_proprio=not a.no_proprio,
                    **pred_dims_from_args(a)).to(device)
    wm.load_state_dict(ckpt["wm"])
    wm.eval()
    for p in wm.parameters():
        p.requires_grad_(False)
    return wm, a


def act_stack(wm, a, hist_z, hist_a, z, zstar, device):
    """One decision of the EXACT training act path (train.py CEM branch + dwell)."""
    plan = cem_plan(wm, hist_z, hist_a, zstar, a.cem_samples, a.cem_iters, a.cem_elites,
                    a.cem_init_std, a.cem_horizon, device, gamma=a.cem_gamma,
                    min_std=a.cem_min_std, mppi_temp=a.cem_mppi_temp,
                    early_stop_tol=a.cem_early_stop,
                    early_stop_min_iters=a.cem_early_stop_min_iters)
    act = plan[:, 0].clamp(-1.0, 1.0)
    gd = (z - zstar).norm(dim=-1)
    eps = a.goal_reach_eps
    if a.dwell_shrink_start > 0:
        scale = (gd / (a.dwell_shrink_start * eps)).clamp(a.dwell_shrink_min, 1.0)
        act = act * scale.unsqueeze(1)
    if a.dwell_hold_mult > 0 and float(gd[0]) < a.dwell_hold_mult * eps:
        act = torch.zeros_like(act)
    return act, float(gd[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--goals", type=int, default=24)
    p.add_argument("--budget", type=int, default=120, help="planner decisions per goal")
    p.add_argument("--walk", type=int, default=40, help="smooth random-walk decisions to a goal pose")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/goal_photo_eval")
    p.add_argument("--fixed-objects", action="store_true",
                   help="OVERRIDE: identical block layout for the photo and the attempt. Without "
                        "this (and with a ckpt trained fixed_objects=False), every reset re-rolls "
                        "the blocks, so the photo contains a scene the arm cannot recreate and the "
                        "latent distance carries an irreducible mismatch (~10-20 units).")
    cli = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(cli.ckpt, map_location="cpu", weights_only=False)
    a0 = SimpleNamespace(**ckpt["args"])
    out = Path(cli.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cli.seed)

    env = SubprocVectorMujocoEnv(n_envs=1, frame_skip=a0.frame_skip, action_max=a0.action_max,
                                 encode_cam=a0.wm_cam, safety_delta=a0.safety_delta,
                                 seed=cli.seed, threads=1, render_backend="egl",
                                 fixed_objects=cli.fixed_objects or getattr(a0, "fixed_objects", False))
    n_dof = env.n_dof
    wm, a = build_wm(ckpt, n_dof, device)
    a_dim = n_dof * a.action_block
    H = a.history_size
    eps = a.goal_reach_eps

    def obs_fix(obs):
        if getattr(a, "no_torque_obs", False):
            scrub_torque_obs(obs, n_dof)
        return obs

    def step_block(act):
        env.step_block_async(act.detach().cpu().numpy().reshape(1, a.action_block, n_dof))
        obs, infos = env.step_block_wait()
        return obs_fix(obs), infos

    results = []
    walk_lens = [max(cli.walk // 8, 3), cli.walk // 4, cli.walk // 2, cli.walk]
    for g in range(cli.goals):
        # --- goal photo: smooth OU walk from reset (in-distribution pose). Walk length
        #     cycles short->long so the d0 bins all populate (a long walk always lands
        #     far outside the trained radius). ---
        obs = obs_fix(env.reset())
        walk = torch.zeros(1, n_dof)
        beta = getattr(a, "smooth_beta", 0.8)
        for _ in range(walk_lens[g % len(walk_lens)]):
            subs = []
            for _ in range(a.action_block):
                walk = (beta * walk + (1 - beta) * torch.randn(1, n_dof)).clamp(-1, 1)
                subs.append(walk.clone())
            act = torch.stack(subs, 1).reshape(1, a_dim)
            obs, infos = step_block(act)
        goal_px = obs["image"].copy()
        goal_prop = obs["proprio"].copy()
        goal_qpos = goal_prop[0, :n_dof].copy()          # proprio = [qpos, qvel, u]
        zstar = encode_obs(wm, goal_px, goal_prop, device)

        # --- plan back from a fresh reset ---
        obs = obs_fix(env.reset())
        z = encode_obs(wm, obs["image"], obs["proprio"], device)
        hist_z = z.unsqueeze(0).repeat(H, 1, 1)
        hist_a = torch.zeros(H, 1, a_dim, device=device)
        d0 = float((z - zstar).norm(dim=-1)[0])
        best = {"d": d0, "t": 0,
                "qerr": float(np.abs(obs["proprio"][0, :n_dof] - goal_qpos).max()),
                "px": obs["image"].copy()}
        arrived_t = -1
        for t in range(1, cli.budget + 1):
            act, gd = act_stack(wm, a, hist_z, hist_a, z, zstar, device)
            hist_a = torch.cat([hist_a[1:], act.unsqueeze(0)], 0)
            obs, infos = step_block(act)
            z = encode_obs(wm, obs["image"], obs["proprio"], device)
            hist_z = torch.cat([hist_z[1:], z.unsqueeze(0)], 0)
            d = float((z - zstar).norm(dim=-1)[0])
            qerr = float(np.abs(obs["proprio"][0, :n_dof] - goal_qpos).max())
            if d < best["d"]:
                best = {"d": d, "t": t, "qerr": qerr, "px": obs["image"].copy()}
            if arrived_t < 0 and d < eps:
                arrived_t = t
        rec = {"goal": g, "d0": d0, "min_d": best["d"], "arrived": bool(best["d"] < eps),
               "t_arrive": arrived_t, "t_best": best["t"], "qerr_at_best": best["qerr"],
               "goal_qpos": goal_qpos.tolist()}
        results.append(rec)
        np.savez_compressed(out / f"goal_{g:02d}.npz", goal_px=goal_px[0],
                            best_px=best["px"][0])
        print(f"[goal {g:02d}] d0={d0:5.2f} min_d={best['d']:5.2f} "
              f"{'ARRIVED@'+str(arrived_t) if rec['arrived'] else 'missed':>11s} "
              f"qerr_best={best['qerr']:.3f} rad", flush=True)

    env.close()
    (out / "results.json").write_text(json.dumps(results, indent=1))
    bins = [(0, 5), (5, 10), (10, 14), (14, 99)]
    print("\n=== reachability vs initial latent distance ===")
    for lo, hi in bins:
        rs = [r for r in results if lo <= r["d0"] < hi]
        if rs:
            arr = np.mean([r["arrived"] for r in rs])
            qe = np.mean([r["qerr_at_best"] for r in rs])
            print(f"d0 [{lo:>2},{hi:>2}): n={len(rs):2d} arrival {arr:.2f}  "
                  f"mean qerr@best {qe:.3f} rad")
    tot = np.mean([r["arrived"] for r in results])
    print(f"overall: {tot:.2f} arrival over {len(results)} photo goals (eps={eps})")


if __name__ == "__main__":
    main()
