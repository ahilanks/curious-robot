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

from model.state_encoder import WorldModel       # noqa: E402
from env.mujoco_env import MujocoSO101Env         # noqa: E402
from src.train import Actor, record_rollout       # noqa: E402


def resolve_ckpt(args) -> str:
    """Local --ckpt if given, else download <name>/ckpt_<step>.pt (or the latest
    step for that run) from the HF Hub."""
    if args.ckpt:
        return args.ckpt
    repo = args.hf_repo or os.environ.get("HF_UPLOAD_REPO_ID")
    if not repo:
        raise SystemExit("no --ckpt and no HF repo (set --hf-repo or HF_UPLOAD_REPO_ID in .env)")
    from huggingface_hub import HfApi, hf_hub_download
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    files = [f for f in api.list_repo_files(repo)
             if f.startswith(f"{args.name}/") and f.endswith(".pt")]
    if not files:
        raise SystemExit(f"no checkpoints for run '{args.name}' in {repo}")
    target = (f"{args.name}/ckpt_{args.step:07d}.pt" if args.step is not None
              else sorted(files)[-1])          # zero-padded step -> lexical sort = latest
    if target not in files:
        raise SystemExit(f"{target} not found in {repo}; available: {sorted(files)}")
    print(f"[hf] downloading {target} from {repo}", flush=True)
    return hf_hub_download(repo_id=repo, filename=target, token=token)


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
    ckpt = torch.load(resolve_ckpt(args), map_location=device, weights_only=False)
    ca = ckpt["args"]
    n_dof = 6

    wm = WorldModel(n_dof=n_dof, action_block=ca["action_block"],
                    history_size=ca["history_size"]).to(device)
    wm.load_state_dict(ckpt["wm"]); wm.eval()
    actor = Actor(wm.z_dim, n_dof * ca["action_block"]).to(device)
    actor.load_state_dict(ckpt["actor"]); actor.eval()

    env = MujocoSO101Env(action_max=ca["action_max"], dq_max=ca["dq_max"],
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
