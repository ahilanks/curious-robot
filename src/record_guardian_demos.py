"""Record scripted-guardian demonstration episodes for SmolVLA finetuning.

Zero-shot SmolVLA never commits to tabletop contact on sim renders (2026-07-10
verdict), so the VLA guardian goes through imitation: the calibration-verified
scripted fleet cycle (rate-limited for smooth, imitable motion) generates
contact-rich episodes; SmolVLA finetunes on them and contributes what the script
cannot -- per-episode variability and instruction conditioning ("each time moving
the block is different").

Episodes are raw npz (parent_view + overhead frames, parent qpos, absolute joint
targets in rad = SmolVLA's native action space, instruction, contact counts);
conversion to LeRobotDataset happens in a separate step (needs lerobot installed).
Only contact-committed episodes are kept (>= --min-contacts substeps touching).
"""
import os, sys, argparse
os.environ["MUJOCO_GL"] = "egl"          # A100 render path: ~10x faster than osmesa
sys.path.insert(0, "/workspace/curious-robot")

import numpy as np
from env.mujoco_env import MujocoSO101Env
from src.parent_vla import ParentFleet

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", type=int, default=150, help="KEPT episodes")
ap.add_argument("--substeps", type=int, default=240, help="per episode (~8s at 30Hz)")
ap.add_argument("--min-contacts", type=int, default=12)
ap.add_argument("--rate", type=float, default=0.02)
ap.add_argument("--seed", type=int, default=51)
ap.add_argument("--out", default="runs/guardian_demos")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

env = MujocoSO101Env(frame_skip=6, action_max=0.05, encode_cam="wrist", safety_delta=15.0,
                     seed=args.seed, fixed_objects=True, parent_arm=True,
                     parent_pos=(0.55, 0.0, 0.02))
n_dof = env.n_dof
rng = np.random.default_rng(args.seed)


class Wrap:
    def parent_context(self):
        return [env.parent_context()]


kept, attempt = 0, 0
while kept < args.episodes and attempt < args.episodes * 4:
    attempt += 1
    env.reset()
    fleet = ParentFleet(1, mode="scripted", rate=args.rate, log=lambda *a, **k: None)
    child_walk = attempt % 2 == 0                    # half the demos: child gently moving
    walk = np.zeros(n_dof, np.float32)
    pv, ov, qp, act = [], [], [], []
    ncon_steps = 0
    instr = ""
    for t in range(args.substeps):
        if t % 5 == 0:                               # fleet refills on the decision cadence
            tgt_block = fleet.next_blocks(Wrap(), 5)
            ti = 0
            instr = fleet.instr[0] or instr
        target = tgt_block[0, ti]; ti = min(ti + 1, 4)
        # observation BEFORE the action (imitation pairs o_t -> a_t)
        pv.append(env.render_parent_view())
        ov.append(env.render_overhead())
        qp.append(env.parent_qpos().copy())
        act.append(np.asarray(target, np.float32))
        env.set_parent_target(target)
        if child_walk:
            walk = np.clip(0.85 * walk + 0.15 * rng.standard_normal(n_dof).astype(np.float32), -1, 1)
            _, info = env.step(0.4 * walk, render=False)
        else:
            _, info = env.step(np.zeros(n_dof, np.float32), render=False)
        ncon_steps += int(info.get("parent_object_contacts", 0) > 0)
    if ncon_steps < args.min_contacts:
        continue
    np.savez_compressed(os.path.join(args.out, f"ep_{kept:04d}.npz"),
                        parent_view=np.stack(pv).astype(np.uint8),
                        overhead=np.stack(ov).astype(np.uint8),
                        qpos=np.stack(qp).astype(np.float32),
                        action=np.stack(act).astype(np.float32),
                        instruction=np.array(instr),
                        contact_steps=ncon_steps, child_walk=child_walk)
    kept += 1
    if kept % 10 == 0:
        print(f"[demo {kept}/{args.episodes}] kept (attempt {attempt}, contacts {ncon_steps}/{args.substeps}, "
              f"child_walk={child_walk}) instr='{instr}'", flush=True)
env.close()
print(f"[done] kept {kept} episodes over {attempt} attempts -> {args.out}", flush=True)
