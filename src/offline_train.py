"""Offline fine-tuning of the Curious Robot WM+SAC on saved replay buffers (no env).

One round of the hardware -> cloud adaptation loop:
  1. collect real transitions on the Mac with a frozen policy
     (train.py --env-backend hardware --frozen-policy ... --save-buffer)
  2. upload the dump to the HF Hub:
     huggingface-cli upload <repo> runs/<run>/buffer_*.npz buffers/<run>/buffer.npz
  3. fine-tune offline on a GPU pod (this script), warm-started from the deployed run:
     python src/offline_train.py --resume-name safe15 --resume-step 100000 \
         --buffer buffers/hw_round_1/buffer.npz --name round_1 --steps 4000
  4. redeploy: train.py --env-backend hardware --resume-name round_1 ...

Buffers are the npz dumps written by train.py's save_buffer (--save-buffer /
--frozen-policy). --buffer takes a local path or a path inside the HF repo and
repeats to mix files (e.g. real + sim against forgetting). Architecture is derived
from the buffer (a_dim, prop_dim, image size); history_size comes from the
warm-start checkpoint (it sizes the predictor's positional embedding). Stored
rewards are replayed as-is, never recomputed (only z is re-encoded each sample, as
online); adaptation therefore shows in the NEXT round's collected r_safe, not in
mid-offline metrics. Optimizer state is not checkpointed, so Adam restarts cold
each round -> keep LRs low (defaults here are fine-tune values, below train.py's).
Checkpoints carry the same 8 keys as train.py, so --resume-name round_N works for
both redeploy (train.py) and the next fine-tune.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lewm.module import SIGReg                            # noqa: E402
from model.state_encoder import WorldModel                # noqa: E402
from src.train import (Actor, ReplayBuffer, TwinQ, collapse_metrics,   # noqa: E402
                       resolve_ckpt, sac_update, save_and_upload, wm_update)

try:
    import wandb
except ImportError:
    wandb = None


def fetch_buffer(spec, hf_repo):
    """Resolve a --buffer spec: an existing local path is used as-is; anything else is
    treated as a path inside the HF repo (--hf-repo, else $HF_UPLOAD_REPO_ID), e.g.
    buffers/hw_round_1/buffer.npz as pushed by `huggingface-cli upload`."""
    if Path(spec).exists():
        return spec
    repo = hf_repo or os.environ.get("HF_UPLOAD_REPO_ID")
    if not repo:
        raise SystemExit(f"--buffer {spec}: not a local file and no HF repo "
                         "(set --hf-repo or HF_UPLOAD_REPO_ID in .env)")
    from huggingface_hub import hf_hub_download
    print(f"[hf] downloading {spec} from {repo}", flush=True)
    return hf_hub_download(repo_id=repo, filename=spec, token=os.environ.get("HF_TOKEN"))


def load_buffer(paths, history_size, h_fwd_max, device):
    """Inverse of train.save_buffer: rebuild a ReplayBuffer from npz dumps. Each
    env_lengths entry (from every file) becomes its OWN env row -- that keeps WM
    windows / SAC pairs inside one contiguous stream without reconstructing the
    original n_envs, and lets real + sim files mix in one buffer. cap_per_env leaves
    slack above the longest stream so count < C always holds (no ring-seam branches);
    rows are padded to the longest stream, fine for the small per-round dumps this is
    meant for. npz carries no priorities -> uniform 1.0 cold start (--per-priority td
    re-adapts them after the first pass)."""
    streams = []
    for p in paths:
        z = np.load(p)
        lengths = z["env_lengths"]
        offs = np.concatenate([[0], np.cumsum(lengths)])
        arrs = {k: z[k] for k in ("pixels", "proprio", "action", "r", "d", "is_start")}
        for j in range(len(lengths)):
            streams.append({k: v[offs[j]:offs[j + 1]] for k, v in arrs.items()})
        print(f"[buffer] {p}: {len(lengths)} stream(s), {int(lengths.sum())} transitions, "
              f"r mean/min/max = {z['r'].mean():.2f}/{z['r'].min():.2f}/{z['r'].max():.2f}",
              flush=True)
    a_dim = streams[0]["action"].shape[-1]
    prop_dim = streams[0]["proprio"].shape[-1]
    img_hw = streams[0]["pixels"].shape[1]
    for s in streams:
        assert (s["action"].shape[-1], s["proprio"].shape[-1], s["pixels"].shape[1]) \
            == (a_dim, prop_dim, img_hw), "mixed buffers disagree on action/proprio/image dims"
    max_len = max(len(s["r"]) for s in streams)
    cap_per_env = max_len + (history_size + h_fwd_max + 8)
    buf = ReplayBuffer(len(streams), cap_per_env, img_hw, a_dim, prop_dim, device)
    for j, s in enumerate(streams):
        n = len(s["r"])
        for k in ("pixels", "proprio", "action", "r", "d", "is_start"):
            getattr(buf, k)[j, :n] = s[k]
        buf.prio[j, :n] = 1.0
        buf.head[j] = n
        buf.count[j] = n
    return buf


# -------------------------------------------------------------------------- main
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[device] {device}", flush=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass

    # --- warm-start ckpt FIRST: its history_size sizes the predictor pos-emb ---
    ck, src_args = None, {}
    if args.init_ckpt or args.resume_name:
        path = resolve_ckpt(args.init_ckpt, args.resume_name, args.resume_step, args.hf_repo)
        ck = torch.load(path, map_location=device, weights_only=False)
        src_args = dict(ck.get("args", {}))
        print(f"[resume] {path} (saved step {ck.get('step', '?')})", flush=True)
    if src_args:
        H = int(src_args.get("history_size", 3))
        if args.history_size is not None and args.history_size != H:
            raise SystemExit(f"--history-size {args.history_size} != checkpoint's {H} "
                             "(it sizes the predictor pos-emb; omit the flag when resuming)")
    else:
        H = args.history_size if args.history_size is not None else 3
    wm_dropout = (args.wm_dropout if args.wm_dropout is not None
                  else float(src_args.get("wm_dropout", 0.1)))
    h_fwd = int(ck["h_fwd"]) if ck is not None else args.h_fwd_start  # carry the curriculum

    # --- buffer(s): architecture (a_dim/prop_dim/img) is derived from the data ---
    paths = [fetch_buffer(s, args.hf_repo) for s in args.buffer]
    buf = load_buffer(paths, H, args.h_fwd_max, device)
    a_dim, prop_dim = buf.action.shape[-1], buf.proprio.shape[-1]
    n_dof = prop_dim // 3
    action_block = a_dim // n_dof
    if src_args and "action_block" in src_args and int(src_args["action_block"]) != action_block:
        raise SystemExit(f"buffer action_block={action_block} != checkpoint's "
                         f"{src_args['action_block']} -- mismatched run")
    print(f"[buffer] total: {buf.n_envs} stream(s) x cap {buf.C} = {buf.total} transitions "
          f"(a_dim={a_dim}, prop_dim={prop_dim}, H={H}, h_fwd={h_fwd})", flush=True)

    # --- models + optimizers, exactly as train.py builds them ---
    wm = WorldModel(n_dof=n_dof, action_block=action_block, history_size=H,
                    dropout=wm_dropout).to(device)
    wm.eval()                                  # train() only inside wm_update
    sigreg = SIGReg(knots=17, num_proj=1024).to(device)
    actor = Actor(wm.z_dim, a_dim).to(device)
    critic = TwinQ(wm.z_dim, a_dim).to(device)
    critic_tgt = TwinQ(wm.z_dim, a_dim).to(device)
    critic_tgt.load_state_dict(critic.state_dict())
    for p in critic_tgt.parameters():
        p.requires_grad_(False)
    log_alpha = torch.tensor(float(np.log(args.alpha)), device=device)
    if ck is not None:
        wm.load_state_dict(ck["wm"])
        actor.load_state_dict(ck["actor"])
        critic.load_state_dict(ck["critic"])
        critic_tgt.load_state_dict(ck["critic_tgt"])
        log_alpha = ck["log_alpha"].to(device)
        print(f"[resume] loaded wm+actor+critic+critic_tgt+log_alpha (h_fwd={h_fwd})", flush=True)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    wm_opt = torch.optim.AdamW([p for p in wm.parameters() if p.requires_grad],
                               lr=args.wm_lr, weight_decay=1e-3)

    # sac_update reads these from args: no warmup offline; per_beta anneals over --steps
    args.start_steps = 0
    args.total_steps = args.steps

    run_name = args.name
    out_dir = Path(args.out_dir) if args.out_dir else Path("runs") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    repo = args.hf_repo or os.environ.get("HF_UPLOAD_REPO_ID")
    # checkpoint args: keep the source run's env params (action_max, safety_delta, ...)
    # for play_policy/redeploy, overlay this run's knobs, then pin the derived arch.
    saved_args = {**src_args, **vars(args), "history_size": H,
                  "action_block": action_block, "wm_dropout": wm_dropout}

    run = None
    if not args.no_wandb and wandb is not None and os.environ.get("WANDB_API_KEY"):
        run = wandb.init(project=args.wandb_project or os.environ.get("WANDB_PROJECT", "curious-robot"),
                         entity=os.environ.get("WANDB_ENTITY"), name=run_name, group=run_name,
                         dir=str(out_dir),
                         config={**saved_args, "z_dim": wm.z_dim, "a_dim": a_dim,
                                 "device": str(device), "offline": True})
        print(f"[wandb] {run.url}", flush=True)

    def ckpt_state(step):   # the EXACT 8 keys train.py saves -> redeploy loads cleanly
        return {"step": step, "wm": wm.state_dict(), "actor": actor.state_dict(),
                "critic": critic.state_dict(), "critic_tgt": critic_tgt.state_dict(),
                "log_alpha": log_alpha.detach().cpu(), "h_fwd": h_fwd, "args": saved_args}

    # --- env-free learner loop: WM before SAC each step (SAC re-encodes under the
    #     fresh encoder); h_fwd stays pinned (no flatline curriculum offline) ---
    last_wm = last_sac = last_zb = None
    n_wm = n_sac = 0
    t0 = time.time()
    for step in range(args.steps):
        if step % args.wm_update_every == 0:
            batch = buf.sample_wm(args.wm_batch_size, H + h_fwd)
            if batch is not None:
                wm.train()
                last_wm = wm_update(wm, sigreg, wm_opt, batch, H, h_fwd,
                                    args.gamma_wm, args.sigreg_weight, device)
                wm.eval()
                n_wm += 1
        res = sac_update(buf, wm, actor, critic, critic_tgt, actor_opt, critic_opt,
                         log_alpha, args, step, device)
        if res is not None:
            last_sac = (res["critic_loss"], res["actor_loss"])
            last_zb = res["zb"]
            n_sac += 1

        if step % args.log_every == 0:
            d = {"buffer/transitions": buf.total, "wm/h_fwd": h_fwd,
                 "perf/steps_per_sec": (step + 1) / (time.time() - t0),
                 "sac/alpha": float(log_alpha.exp().item())}
            if last_wm is not None:
                d.update({"wm/pred_loss": last_wm[0], "wm/sigreg": last_wm[1],
                          "wm/identity_baseline": last_wm[2]})
            if last_sac is not None:
                d.update({"sac/critic_loss": last_sac[0], "sac/actor_loss": last_sac[1]})
            if last_zb is not None:
                z_std, eff_rank, feat_corr = collapse_metrics(last_zb)
                d.update({"encoder/z_std": z_std, "encoder/eff_rank": eff_rank,
                          "encoder/feat_corr": feat_corr})
            if run is not None:
                run.log(d, step=step)
            with open(out_dir / "metrics.jsonl", "a") as f:
                f.write(json.dumps({"step": step, **{k: float(v) for k, v in d.items()}}) + "\n")
            print(f"[step {step}] pred={d.get('wm/pred_loss', float('nan')):.4f} "
                  f"sigreg={d.get('wm/sigreg', float('nan')):.3f} "
                  f"critic={d.get('sac/critic_loss', float('nan')):.3f} "
                  f"actor={d.get('sac/actor_loss', float('nan')):.3f} "
                  f"eff_rank={d.get('encoder/eff_rank', float('nan')):.1f} "
                  f"sps={d['perf/steps_per_sec']:.1f}", flush=True)

        if args.save_every > 0 and step > 0 and step % args.save_every == 0:
            save_and_upload(ckpt_state(step), out_dir, step, repo, run_name,
                            not args.no_hf, args.keep_local_ckpts)

    save_and_upload(ckpt_state(args.steps), out_dir, args.steps, repo, run_name,
                    not args.no_hf, args.keep_local_ckpts)
    if n_wm == 0:
        print(f"[warn] wm_update NEVER fired: every stream is shorter than "
              f"history+h_fwd+1={H + h_fwd + 1} or too few windows for "
              f"--wm-batch-size {args.wm_batch_size}", flush=True)
    if n_sac == 0:
        print(f"[warn] sac_update NEVER fired: valid (t,t+1) pairs < --batch-size "
              f"{args.batch_size} (single-stream pairs = len-1); lower --batch-size "
              "or collect more transitions", flush=True)
    print(f"[done] {args.steps} steps: {n_wm} wm updates, {n_sac} sac update steps", flush=True)
    if run is not None:
        run.finish()


def parse_args():
    p = argparse.ArgumentParser(description="Curious Robot: offline WM+SAC fine-tuning "
                                            "on saved buffer npz dumps (no env)")
    p.add_argument("--buffer", action="append", required=True,
                   help="buffer npz: local path or path in the HF repo "
                        "(e.g. buffers/hw_round_1/buffer.npz); repeat to mix files")
    # warm-start (cold start if neither given -- only useful for smokes)
    p.add_argument("--init-ckpt", default=None, help="local .pt to warm-start from")
    p.add_argument("--resume-name", default=None, help="HF run name to warm-start from (e.g. safe15)")
    p.add_argument("--resume-step", type=int, default=None, help="ckpt step (default: latest)")
    p.add_argument("--name", default="offline", help="this run's name (W&B, runs/<name>/, HF folder)")
    p.add_argument("--out-dir", default=None, help="local dir (default: runs/<name>)")
    p.add_argument("--steps", type=int, default=4000, help="offline learner steps")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--keep-local-ckpts", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    # world model (defaults = train.py's except the fine-tune LR; history/dropout
    # default to the warm-start checkpoint's values)
    p.add_argument("--history-size", type=int, default=None,
                   help="H_bwd; ONLY for cold starts (warm starts take the checkpoint's)")
    p.add_argument("--h-fwd-start", type=int, default=1, help="rollout horizon for cold starts")
    p.add_argument("--h-fwd-max", type=int, default=1, help="buffer slack sizing only")
    p.add_argument("--gamma-wm", type=float, default=0.95)
    p.add_argument("--sigreg-weight", type=float, default=0.3, help="beta, pinned (anti-collapse)")
    p.add_argument("--wm-batch-size", type=int, default=128)
    p.add_argument("--wm-lr", type=float, default=1e-5,
                   help="fine-tune default (train.py uses 5e-5); optimizers restart cold")
    p.add_argument("--wm-update-every", type=int, default=4)
    p.add_argument("--wm-dropout", type=float, default=None)
    # SAC (defaults = train.py's except the fine-tune LRs)
    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument("--alpha", type=float, default=0.2, help="fixed entropy temperature (cold starts)")
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--actor-lr", type=float, default=1e-4,
                   help="fine-tune default (train.py uses 3e-4)")
    p.add_argument("--critic-lr", type=float, default=1e-4,
                   help="fine-tune default (train.py uses 3e-4)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--updates-per-step", type=int, default=1)
    p.add_argument("--per-alpha", type=float, default=0.6)
    p.add_argument("--per-beta-start", type=float, default=0.4)
    p.add_argument("--per-priority", choices=["curiosity", "td"], default="curiosity",
                   help="npz has no priorities -> uniform cold start; 'td' re-adapts "
                        "them from |TD error| after the first sampling pass")
    # logging / HF
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--no-hf", action="store_true", help="disable HF checkpoint upload")
    p.add_argument("--hf-repo", default=None, help="HF repo id (else $HF_UPLOAD_REPO_ID)")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
