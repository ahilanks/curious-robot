"""Roll out a trained policy in the SO-ARM101 env and save overhead + wrist videos.

Checkpoints are pulled from the HF Hub by default (the trainer uploads them as
<name>/ckpt_<step>.pt under $HF_UPLOAD_REPO_ID); pass --ckpt to use a local file.

    # latest checkpoint of run "baseline" from the default HF repo
    python src/play_policy.py --name baseline
    # a specific step, from an explicit repo
    python src/play_policy.py --name baseline --step 5000 --hf-repo a5ilank/curious-robot
    # a local file
    python src/play_policy.py --ckpt runs/baseline/ckpt_0005000.pt
"""
import argparse
import os
import platform
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw" if platform.system() == "Darwin" else "egl")

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.state_encoder import WorldModel, pred_dims_from_args       # noqa: E402
from env.mujoco_env import MujocoSO101Env         # noqa: E402
from src.train import Actor, load_actor_state, record_rollout, resolve_ckpt   # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Play a trained Curious Robot policy (HF or local ckpt).")
    p.add_argument("--name", default="baseline", help="run name (HF folder) to pull from")
    p.add_argument("--step", type=int, default=None, help="checkpoint step (default: latest)")
    p.add_argument("--hf-repo", default=None, help="HF repo id (else $HF_UPLOAD_REPO_ID)")
    p.add_argument("--ckpt", default=None, help="local .pt path (overrides HF fetch)")
    p.add_argument("--out-dir", default="play", help="where to write the mp4s")
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--seed", type=int, default=123)
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
    ca = ckpt["args"]
    n_dof = 6

    wm = WorldModel(n_dof=n_dof, action_block=ca["action_block"],
                    history_size=ca["history_size"], **pred_dims_from_args(ca)).to(device)
    wm.load_state_dict(ckpt["wm"]); wm.eval()
    actor = Actor(wm.z_dim, n_dof * ca["action_block"]).to(device)
    load_actor_state(actor, ckpt["actor"]); actor.eval()

    env = MujocoSO101Env(action_max=ca["action_max"],
                         safety_delta=ca["safety_delta"], seed=args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.name if args.step is None else f"{args.name}_{args.step:07d}"
    paths = record_rollout(actor, wm, env, ca["action_block"], n_dof, device,
                           out_dir, tag, args.steps, args.fps)
    env.close()
    if paths:
        for cam, p_ in paths.items():
            print(f"[play] wrote {cam}: {p_}")
    else:
        print("[play] imageio missing — pip install 'imageio[ffmpeg]'")


if __name__ == "__main__":
    main()
