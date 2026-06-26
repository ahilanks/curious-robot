"""Open-loop multi-step prediction error of the world model -- the "is it learning
dynamics?" metric. Collects a policy trajectory, then for each horizon h rolls the
predictor h steps open-loop (feeding its own output back with the real actions) and
compares to the real latent. Reports pred MSE and the persistence baseline per h.

    python src/eval_predictor.py --ckpt runs/curious/ckpt_0050000.pt --steps 500
"""
import argparse
import json
import os
import platform
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw" if platform.system() == "Darwin" else "egl")

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.state_encoder import WorldModel, pred_dims_from_args       # noqa: E402
from env.mujoco_env import MujocoSO101Env         # noqa: E402
from src.train import Actor, encode_obs, load_actor_state, resolve_ckpt   # noqa: E402


@torch.no_grad()
def collect(env, wm, actor, n_dof, action_block, n_steps, device):
    obs = env.reset()
    z = encode_obs(wm, obs["image"][None], obs["proprio"][None], device)[0]
    zs, acts = [z], []
    for _ in range(n_steps):
        a = actor(z[None])
        acts.append(a[0])
        a_env = a[0].cpu().numpy().reshape(action_block, n_dof)
        for k in range(action_block):
            obs, _ = env.step(a_env[k])
        z = encode_obs(wm, obs["image"][None], obs["proprio"][None], device)[0]
        zs.append(z)
    return torch.stack(zs), torch.stack(acts)      # (N+1, D), (N, A)


@torch.no_grad()
def rollout_error(wm, Z, A, H, maxh):
    N = A.shape[0]
    starts = torch.arange(H - 1, N - maxh + 1)
    if len(starts) < 2:
        return {}
    win = lambda X, off: X[(starts[:, None] + off + torch.arange(-H + 1, 1)[None, :])]
    ctx_z, ctx_a = win(Z, 0), win(A, 0)            # Z[t-H+1..t], A[t-H+1..t]
    cur = wm.predict(ctx_z, wm.action_encoder(ctx_a))
    roll, roll_a = ctx_z, wm.action_encoder(ctx_a)
    mean_z = Z.mean(0)                              # constant-mean baseline (mean-collapse check)
    out = {}
    for h in range(1, maxh + 1):
        zhat = cur[:, -1]
        z_true = Z[starts + h]
        out[h] = {"pred_mse": float(((zhat - z_true) ** 2).sum(-1).mean()),
                  "persist_mse": float(((Z[starts] - z_true) ** 2).sum(-1).mean()),
                  "mean_mse": float(((mean_z - z_true) ** 2).sum(-1).mean())}
        if h < maxh:
            na = wm.action_encoder(A[starts + h][:, None, :])
            roll = torch.cat([roll[:, 1:], zhat[:, None]], 1)
            roll_a = torch.cat([roll_a[:, 1:], na], 1)
            cur = wm.predict(roll, roll_a)
    return out


def main():
    p = argparse.ArgumentParser(description="Open-loop WM prediction error (HF or local ckpt).")
    p.add_argument("--name", default="baseline", help="run name (HF folder) to pull from")
    p.add_argument("--step", type=int, default=None, help="checkpoint step (default: latest)")
    p.add_argument("--hf-repo", default=None, help="HF repo id (else $HF_UPLOAD_REPO_ID)")
    p.add_argument("--ckpt", default=None, help="local .pt path (overrides HF fetch)")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--maxh", type=int, default=20)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default=None, help="json output path (default derived from run/ckpt)")
    args = p.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = torch.load(resolve_ckpt(args.ckpt, args.name, args.step, args.hf_repo),
                      map_location=device, weights_only=False)
    ca = ckpt["args"]; n_dof = 6
    wm = WorldModel(n_dof=n_dof, action_block=ca["action_block"],
                    history_size=ca["history_size"], **pred_dims_from_args(ca)).to(device)
    wm.load_state_dict(ckpt["wm"]); wm.eval()
    actor = Actor(wm.z_dim, n_dof * ca["action_block"]).to(device)
    load_actor_state(actor, ckpt["actor"]); actor.eval()
    env = MujocoSO101Env(action_max=ca["action_max"],
                         safety_delta=ca["safety_delta"], seed=args.seed)

    Z, A = collect(env, wm, actor, n_dof, ca["action_block"], args.steps, device)
    res = rollout_error(wm, Z, A, ca["history_size"], args.maxh)
    env.close()

    print(f"{'h':>3} {'pred_mse':>12} {'persist_mse':>12} {'mean_mse':>12} {'pred/persist':>12} {'pred/mean':>12}")
    for h, v in res.items():
        pp = v["pred_mse"] / max(v["persist_mse"], 1e-9)   # ~1.0 = persistence-collapse
        pm = v["pred_mse"] / max(v["mean_mse"], 1e-9)       # ~1.0 = mean-collapse (predicts the batch mean)
        print(f"{h:>3} {v['pred_mse']:>12.4f} {v['persist_mse']:>12.4f} {v['mean_mse']:>12.4f} "
              f"{pp:>12.3f} {pm:>12.3f}")
    if args.out:
        out = Path(args.out)
    elif args.ckpt:
        out = Path(args.ckpt).with_suffix(".pred_eval.json")
    else:
        tag = args.name if args.step is None else f"{args.name}_{args.step:07d}"
        out = Path(f"{tag}.pred_eval.json")
    out.write_text(json.dumps(res, indent=2))
    print(f"[eval] wrote {out}")


if __name__ == "__main__":
    main()
