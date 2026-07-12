"""Behavioral validation gate for the finetuned SmolVLA guardian.

Measures what actually matters before any recipe swap: does the VLA-driven
ParentFleet (the EXACT integration path train.py uses) produce block contact
in sim? Bar: mean parent_object_contacts fraction > 0.05/substep — the same
threshold the demo recorder used as its keep-bar (12/240). Zero-shot
smolvla_base scored 0.000 (2026-07-10 verdict); pass --model-id
lerobot/smolvla_base to re-measure that baseline in this harness.

Mirrors record_guardian_demos.py conditions: same env build (big table, fixed
layout, base 0.55), alternating child-still / child-walking episodes, fleet
rate limit 0.02. Also logs per-episode instructions + block displacement — the
"each time moving the block is different" evidence the user asked for.
"""
import os, sys, argparse, json
os.environ["MUJOCO_GL"] = "egl"          # A100 render path: ~10x faster than osmesa
sys.path.insert(0, "/workspace/curious-robot")

import numpy as np
from env.mujoco_env import MujocoSO101Env
from src.parent_vla import ParentFleet

ap = argparse.ArgumentParser()
ap.add_argument("--model-id", default="runs/smolvla_guardian/checkpoints/last/pretrained_model")
ap.add_argument("--episodes", type=int, default=8)
ap.add_argument("--substeps", type=int, default=240, help="per episode (~8s at 30Hz)")
ap.add_argument("--rate", type=float, default=0.02)
ap.add_argument("--seed", type=int, default=123)
ap.add_argument("--out", default="runs/vla_guardian_validation.json")
args = ap.parse_args()

env = MujocoSO101Env(frame_skip=6, action_max=0.05, encode_cam="wrist", safety_delta=15.0,
                     seed=args.seed, fixed_objects=True, parent_arm=True,
                     parent_pos=(0.55, 0.0, 0.02))
n_dof = env.n_dof
rng = np.random.default_rng(args.seed)


class Wrap:
    def parent_context(self):
        return [env.parent_context()]


fleet = ParentFleet(1, mode="smolvla", rate=args.rate, model_id=args.model_id)
eps = []
for ep in range(args.episodes):
    env.reset()
    child_walk = ep % 2 == 1                         # half the episodes: child gently moving
    walk = np.zeros(n_dof, np.float32)
    block0 = {i: env.data.xpos[b][:2].copy() for i, b in enumerate(env._object_body_ids)}
    contact_steps, instrs, ti, tgt_block = 0, [], 0, None
    for t in range(args.substeps):
        if t % 5 == 0:                               # fleet refills on the decision cadence
            tgt_block = fleet.next_blocks(Wrap(), 5)
            ti = 0
            if fleet.instr[0] and (not instrs or instrs[-1] != fleet.instr[0]):
                instrs.append(fleet.instr[0])
        target = tgt_block[0, ti]; ti = min(ti + 1, 4)
        env.set_parent_target(target)
        if child_walk:
            walk = np.clip(0.85 * walk + 0.15 * rng.standard_normal(n_dof).astype(np.float32), -1, 1)
            _, info = env.step(0.4 * walk, render=False)
        else:
            _, info = env.step(np.zeros(n_dof, np.float32), render=False)
        contact_steps += int(info.get("parent_object_contacts", 0) > 0)
    disp_cm = 100 * float(sum(np.linalg.norm(env.data.xpos[b][:2] - block0[i])
                              for i, b in enumerate(env._object_body_ids)))
    eps.append({"contact_frac": contact_steps / args.substeps, "block_disp_cm": disp_cm,
                "child_walk": child_walk, "instructions": instrs})
    print(f"[ep {ep}] contact/substep {eps[-1]['contact_frac']:.3f} | block disp {disp_cm:.1f} cm | "
          f"child_walk={child_walk} | {instrs[:2]}", flush=True)

env.close()
mean_c = float(np.mean([e["contact_frac"] for e in eps]))
res = {"model_id": args.model_id, "episodes": args.episodes, "substeps": args.substeps,
       "rate": args.rate, "seed": args.seed,
       "mean_contact_frac": mean_c,
       "mean_block_disp_cm": float(np.mean([e["block_disp_cm"] for e in eps])),
       "pass_bar_0.05": mean_c > 0.05, "eps": eps}
json.dump(res, open(args.out, "w"), indent=2)
print(f"RESULT mean contact/substep {mean_c:.3f} (bar 0.05, zero-shot 0.000) -> "
      f"{'PASS' if mean_c > 0.05 else 'FAIL'} | mean disp {res['mean_block_disp_cm']:.1f} cm "
      f"-> {args.out}", flush=True)
