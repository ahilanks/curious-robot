"""Curious Robot — from-scratch JEPA+SIGReg world model co-trained with SAC under
an intrinsic curiosity reward, on the MuJoCo SO-ARM101 (README is the spec).

Per decision step (action_block env steps), for every parallel env:
  1. encode z_t from (wrist image, proprio) with the current world model
  2. sample a_t ~ pi(.|z_t); apply it (delta-target PD) over action_block env steps
  3. curiosity   r_cur = ||f(z_{t-H+1:t}, a_t) - z_{t+1}||^2     (1-step pred error)
     reward      r_t   = sum_k r_safe_k + lambda_cur * symlog(r_cur)
  4. store (o_t, q_t, qdot_t, a_t, r_t, o_{t+1}) with PER priority = r_cur
Periodically: co-train the WM (autoregressive MSE rollout loss + beta*SIGReg, with
an H_fwd curriculum), run SAC updates (PER), log to W&B, checkpoint to the HF Hub.

The `?` constants in the README (beta, lambda_cur, delta, ...) are CLI flags here with
working defaults, meant to be swept; they are intentionally NOT pinned in the README.
"""
import os
import platform
os.environ.setdefault("MUJOCO_GL", "glfw" if platform.system() == "Darwin" else "osmesa")

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lewm.module import SIGReg                       # noqa: E402
from model.state_encoder import WorldModel           # noqa: E402
from env.parallel_env import VectorMujocoEnv, SubprocVectorMujocoEnv   # noqa: E402
from src.probe import load_probe_hf                  # noqa: E402

try:
    import wandb
except ImportError:
    wandb = None
try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def to_norm_pixel(px_uint8, device):
    """uint8 (...,H,W,3) -> normalized float (...,3,H,W)."""
    t = torch.as_tensor(np.ascontiguousarray(px_uint8), device=device)
    perm = list(range(t.ndim - 3)) + [t.ndim - 1, t.ndim - 3, t.ndim - 2]
    t = t.permute(*perm).float() / 255.0
    shp = [1] * (t.ndim - 3) + [3, 1, 1]
    return (t - IMAGENET_MEAN.to(device).view(shp)) / IMAGENET_STD.to(device).view(shp)


# --------------------------------------------------------------------------- SAC
class Actor(nn.Module):
    def __init__(self, z_dim, a_dim, hidden=256, log_std_range=(-5, 2)):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(z_dim, hidden), nn.ReLU(),
                                   nn.Linear(hidden, hidden), nn.ReLU())
        self.mean = nn.Linear(hidden, a_dim)
        self.log_std = nn.Linear(hidden, a_dim)
        self.lo, self.hi = log_std_range

    def forward(self, z):
        h = self.trunk(z)
        return self.mean(h), self.log_std(h).clamp(self.lo, self.hi)

    def sample(self, z):
        mean, log_std = self(z)
        normal = torch.distributions.Normal(mean, log_std.exp())
        u = normal.rsample()
        a = torch.tanh(u)
        logp = (normal.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)).sum(-1, keepdim=True)
        return a, logp, torch.tanh(mean)


class TwinQ(nn.Module):
    def __init__(self, z_dim, a_dim, hidden=256):
        super().__init__()
        mk = lambda: nn.Sequential(nn.Linear(z_dim + a_dim, hidden), nn.ReLU(),
                                   nn.Linear(hidden, hidden), nn.ReLU(),
                                   nn.Linear(hidden, 1))
        self.q1, self.q2 = mk(), mk()

    def forward(self, z, a):
        za = torch.cat([z, a], -1)
        return self.q1(za), self.q2(za)


# ------------------------------------------------------------------------ buffer
class ReplayBuffer:
    """Per-env ring buffers (so WM windows stay inside one env's contiguous stream)
    with global PER sampling for SAC. Stores raw obs; z is encoded fresh from the
    current WM at sample time, so the co-trained encoder can drift without staleness.
    PER priority = curiosity surprise (raw 1-step prediction error)."""

    def __init__(self, n_envs, cap_per_env, img_hw, a_dim, prop_dim, device):
        self.n_envs, self.C, self.device = n_envs, cap_per_env, device
        s = (n_envs, cap_per_env)
        self.pixels = np.zeros((*s, img_hw, img_hw, 3), np.uint8)
        self.proprio = np.zeros((*s, prop_dim), np.float32)
        self.action = np.zeros((*s, a_dim), np.float32)
        self.r = np.zeros(s, np.float32)
        self.d = np.zeros(s, np.float32)
        self.is_start = np.zeros(s, bool)
        self.prio = np.zeros(s, np.float64)
        self.head = np.zeros(n_envs, np.int64)
        self.count = np.zeros(n_envs, np.int64)

    def add(self, pixels, proprio, action, r, d, is_start, prio):
        for e in range(self.n_envs):
            i = self.head[e]
            self.pixels[e, i] = pixels[e]
            self.proprio[e, i] = proprio[e]
            self.action[e, i] = action[e]
            self.r[e, i] = r[e]
            self.d[e, i] = d[e]
            self.is_start[e, i] = is_start[e]
            self.prio[e, i] = prio[e]
            self.head[e] = (i + 1) % self.C
            self.count[e] = min(self.count[e] + 1, self.C)

    @property
    def total(self):
        return int(self.count.sum())

    def _valid_pairs(self):
        """(e, i) transitions whose next slot (e, i+1) is written and valid."""
        es, isx = [], []
        for e in range(self.n_envs):
            n = int(self.count[e])
            if n < 2:
                continue
            if n == self.C:
                forbid = (int(self.head[e]) - 1) % self.C   # newest, no next yet
                idx = np.delete(np.arange(self.C), forbid)
            else:
                idx = np.arange(n - 1)
            es.append(np.full(len(idx), e)); isx.append(idx)
        if not es:
            return None
        return np.concatenate(es), np.concatenate(isx)

    def sample_sac(self, batch, per_alpha, per_beta):
        vp = self._valid_pairs()
        if vp is None or len(vp[0]) < batch:
            return None
        e_all, i_all = vp
        pr = (self.prio[e_all, i_all] + 1e-6) ** per_alpha
        probs = pr / pr.sum()
        sel = np.random.choice(len(e_all), size=batch, p=probs)
        e, i = e_all[sel], i_all[sel]
        ni = (i + 1) % self.C
        w = (len(e_all) * probs[sel]) ** (-per_beta)
        w = (w / w.max()).astype(np.float32)
        t = lambda x: torch.as_tensor(x, device=self.device)
        return {
            "px": self.pixels[e, i], "prop": self.proprio[e, i],
            "px_n": self.pixels[e, ni], "prop_n": self.proprio[e, ni],
            "a": t(self.action[e, i]), "r": t(self.r[e, i])[:, None],
            "d": t(self.d[e, i])[:, None], "w": t(w)[:, None],
            "e": e, "i": i,                          # for optional TD-error priority writeback
        }

    def update_priorities(self, e, i, prio):
        """Overwrite priorities of sampled transitions (used by --per-priority td)."""
        self.prio[e, i] = np.maximum(np.asarray(prio, np.float64), 1e-6)

    def sample_wm(self, batch, T):
        """Sample (px, proprio, action) windows of length T contiguous within one
        env (no episode-start crossing, no ring-seam crossing)."""
        starts = []
        for e in range(self.n_envs):
            n = int(self.count[e])
            if n < T + 1:
                continue
            if n < self.C:
                lo, hi, head = 0, n - T, -1            # linear region, no wrap
            else:
                lo, hi, head = 0, self.C - T, int(self.head[e])
            for _ in range(8 * batch // max(self.n_envs, 1) + 4):
                s = np.random.randint(lo, hi + 1)
                if head >= 0 and s < head <= s + T - 1:   # straddles the time seam
                    continue
                if self.is_start[e, s + 1:s + T].any():    # crosses an episode reset
                    continue
                starts.append((e, s))
                if len(starts) >= batch:
                    break
            if len(starts) >= batch:
                break
        if len(starts) < max(batch // 4, 1):
            return None
        e = np.array([p[0] for p in starts]); s = np.array([p[1] for p in starts])
        idx = s[:, None] + np.arange(T)[None, :]
        ee = e[:, None].repeat(T, 1)
        return (self.pixels[ee, idx], self.proprio[ee, idx],
                torch.as_tensor(self.action[ee, idx], device=self.device))


# ------------------------------------------------------------------- WM helpers
@torch.no_grad()
def encode_obs(wm, px_uint8, proprio_np, device):
    return wm.encode(to_norm_pixel(px_uint8, device),
                     torch.as_tensor(proprio_np, device=device).float())


@torch.no_grad()
def curiosity_reward(wm, hist_z, hist_a, z_next):
    """r_cur = mean_d (f(z_{t-H+1:t}, a_t)[-1] - z_{t+1})^2, per env: the PER-DIM MEAN
    squared 1-step prediction error (same normalization as the WM loss). Keeps r_cur
    O(0.1-1) so symlog operates in its sensitive region (not the saturated tail of the
    d_z-summed version) -> a more discriminative curiosity reward. Returns (B,)."""
    z_ctx = hist_z.transpose(0, 1)                       # (B, H, D)
    a_emb = wm.action_encoder(hist_a.transpose(0, 1))    # (B, H, A_emb)
    pred = wm.predict(z_ctx, a_emb)[:, -1]               # (B, D)
    return (pred - z_next).pow(2).mean(-1)


def wm_update(wm, sigreg, opt, batch, H_bwd, h, gamma_wm, beta, device):
    """One AdamW step on L_wm = discounted plain-MSE autoregressive rollout + beta*SIGReg
    (LeWM-style: mean squared error per step over batch+feature dims, no symlog)."""
    px, prop, ac = batch
    B, T = px.shape[:2]
    px_n = to_norm_pixel(px, device).reshape(B * T, 3, px.shape[2], px.shape[3])
    prop_t = torch.as_tensor(prop, device=device).float().reshape(B * T, -1)
    emb = wm.encode(px_n, prop_t).reshape(B, T, -1)       # (B,T,D) WITH grad
    a_emb = wm.action_encoder(ac)
    ctx_z, ctx_a = emb[:, :H_bwd], a_emb[:, :H_bwd]
    cur = wm.predict(ctx_z, ctx_a)                         # cur[:, -1] = zhat_{t+1}
    roll, roll_a = ctx_z, ctx_a
    pred_loss, wsum = 0.0, 0.0
    for k in range(1, h + 1):
        zhat = cur[:, -1]
        z_k = emb[:, H_bwd - 1 + k]                        # real z_{t+k}
        g = gamma_wm ** k
        mse = (zhat - z_k).pow(2).mean()                   # LeWM plain MSE over batch+feature dims (no symlog)
        pred_loss = pred_loss + g * mse
        wsum += g
        if k < h:
            roll = torch.cat([roll[:, 1:], zhat.unsqueeze(1)], 1)
            roll_a = torch.cat([roll_a[:, 1:], a_emb[:, H_bwd - 1 + k].unsqueeze(1)], 1)
            cur = wm.predict(roll, roll_a)
    pred_loss = pred_loss / wsum
    with torch.no_grad():   # persistence baseline on the SAME discounted h-step schedule (same MSE metric)
        z_last = emb[:, H_bwd - 1]
        idl = sum((gamma_wm ** k) * (z_last - emb[:, H_bwd - 1 + k]).pow(2).mean()
                  for k in range(1, h + 1)) / wsum
    sig = sigreg(emb.transpose(0, 1))                      # (T,B,D)
    loss = pred_loss + beta * sig
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in wm.parameters() if p.requires_grad], 1.0)
    opt.step()
    return float(pred_loss.item()), float(sig.item()), float(idl)


@torch.no_grad()
def collapse_metrics(z):
    """Encoder-collapse diagnostics on a batch of latents z (B, D), computed on CPU
    (linalg has gaps on MPS): mean per-dim std (->0 collapsed), participation-ratio
    effective rank of the feature covariance (large/<=min(B,D) when healthy, ->1
    collapsed), and mean |off-diagonal feature correlation| (->1 collapsed)."""
    z = z.detach().float().cpu()
    B, D = z.shape
    std = z.std(0)
    zc = z - z.mean(0, keepdim=True)
    lam = torch.linalg.svdvals(zc) ** 2                       # covariance eigenvalues
    eff_rank = (lam.sum() ** 2 / (lam.pow(2).sum() + 1e-12)).item()
    denom = std.clamp_min(1e-6)
    corr = (zc.t() @ zc) / (B * denom[:, None] * denom[None, :])
    off = corr.abs().sum() - corr.diagonal().abs().sum()
    feat_corr = (off / (D * (D - 1) + 1e-9)).item()
    return float(std.mean()), float(eff_rank), float(feat_corr)


# ----------------------------------------------------------------- checkpointing
def save_and_upload(state, out_dir, step, repo_id, run_name, enable_hf, keep_local):
    """Save a checkpoint, upload to HF under <run_name>/ckpt_<step>.pt, then (unless
    keep_local) delete the local copy once the upload succeeds -- so disk stays
    bounded over a long run. On upload failure the local file is kept as a fallback."""
    path = out_dir / f"ckpt_{step:07d}.pt"
    torch.save(state, path)
    uploaded = False
    if enable_hf and repo_id and os.environ.get("HF_TOKEN"):
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=os.environ["HF_TOKEN"])
            api.create_repo(repo_id, repo_type="model", exist_ok=True)
            api.upload_file(path_or_fileobj=str(path), repo_id=repo_id,
                            path_in_repo=f"{run_name}/ckpt_{step:07d}.pt")
            uploaded = True
            print(f"[hf] uploaded {run_name}/ckpt_{step:07d}.pt -> {repo_id}", flush=True)
        except Exception as ex:
            print(f"[hf] upload failed (non-fatal, keeping local): {ex}", flush=True)
    if uploaded and not keep_local:
        try:
            path.unlink()
        except OSError:
            pass
        return None
    return path


def resolve_ckpt(ckpt=None, name="baseline", step=None, hf_repo=None):
    """Return a local checkpoint path: the explicit `ckpt` if given, else download
    <name>/ckpt_<step>.pt (or the latest step for that run) from the HF Hub
    (hf_repo, else $HF_UPLOAD_REPO_ID). Shared by play_policy.py / eval_predictor.py."""
    if ckpt:
        return ckpt
    repo = hf_repo or os.environ.get("HF_UPLOAD_REPO_ID")
    if not repo:
        raise SystemExit("no --ckpt and no HF repo (set --hf-repo or HF_UPLOAD_REPO_ID in .env)")
    from huggingface_hub import HfApi, hf_hub_download
    token = os.environ.get("HF_TOKEN")
    files = [f for f in HfApi(token=token).list_repo_files(repo)
             if f.startswith(f"{name}/") and f.endswith(".pt")]
    if not files:
        raise SystemExit(f"no checkpoints for run '{name}' in {repo}")
    target = f"{name}/ckpt_{step:07d}.pt" if step is not None else sorted(files)[-1]
    if target not in files:
        raise SystemExit(f"{target} not found in {repo}; available: {sorted(files)}")
    print(f"[hf] downloading {target} from {repo}", flush=True)
    return hf_hub_download(repo_id=repo, filename=target, token=token)


def tile_frames(imgs):
    """Tile (n_envs, H, W, 3) per-env camera frames into one (rows*H, cols*W, 3) grid,
    so a single train-video clip shows all parallel envs at once."""
    n, H, W, C = imgs.shape
    cols = int(np.ceil(np.sqrt(n))); rows = int(np.ceil(n / cols))
    grid = np.zeros((rows * H, cols * W, C), imgs.dtype)
    for i in range(n):
        r, c = divmod(i, cols)
        grid[r * H:(r + 1) * H, c * W:(c + 1) * W] = imgs[i]
    return grid


# -------------------------------------------------------------------------- main
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[device] {device}  MUJOCO_GL={os.environ.get('MUJOCO_GL')}", flush=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass

    VecEnv = SubprocVectorMujocoEnv if args.env_backend == "subproc" else VectorMujocoEnv
    env = VecEnv(n_envs=args.n_envs, frame_skip=args.frame_skip,
                 action_max=args.action_max, dq_max=args.dq_max,
                 safety_delta=args.safety_delta, seed=args.seed,
                 threads=args.env_threads)
    n_dof = env.n_dof
    a_dim = n_dof * args.action_block
    prop_dim = 3 * n_dof
    H = args.history_size

    wm = WorldModel(n_dof=n_dof, action_block=args.action_block,
                    history_size=H, dropout=args.wm_dropout).to(device)
    if args.wm_grad_checkpoint:  # off by default: ViT-tiny encode activations are sub-GB vs 80GB free,
        try:                     # so recompute-on-backward is pure slowdown here (the H_fwd rollout is in latent space)
            wm.encoder.vit.gradient_checkpointing_enable()
        except Exception as ex:
            print(f"[wm] grad checkpoint not enabled: {ex}", flush=True)
    wm.eval()                                  # train() only inside wm_update
    sigreg = SIGReg(knots=17, num_proj=1024).to(device)
    z_dim = wm.z_dim

    actor = Actor(z_dim, a_dim).to(device)
    critic = TwinQ(z_dim, a_dim).to(device)
    critic_tgt = TwinQ(z_dim, a_dim).to(device)
    critic_tgt.load_state_dict(critic.state_dict())
    for p in critic_tgt.parameters():
        p.requires_grad_(False)
    log_alpha = torch.tensor(float(np.log(args.alpha)), device=device, requires_grad=args.auto_alpha)
    target_entropy = args.target_entropy if args.target_entropy is not None else -float(a_dim)

    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    alpha_opt = torch.optim.Adam([log_alpha], lr=args.alpha_lr) if args.auto_alpha else None
    if args.auto_alpha:
        print(f"[alpha] learnable (target_entropy={target_entropy:.1f}, lr={args.alpha_lr})", flush=True)
    wm_opt = torch.optim.AdamW([p for p in wm.parameters() if p.requires_grad],
                               lr=args.wm_lr, weight_decay=1e-3)

    cap = int(np.clip(args.buffer_frac * args.total_steps, 1000, 50_000))
    cap_per_env = max(cap // args.n_envs, args.history_size + args.h_fwd_max + 8)
    buf = ReplayBuffer(args.n_envs, cap_per_env, env.wrist_resolution, a_dim, prop_dim, device)
    print(f"[buffer] {args.n_envs} x {cap_per_env} = {args.n_envs * cap_per_env} transitions", flush=True)

    run_name = args.name
    out_dir = Path(args.out_dir) if args.out_dir else Path("runs") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    n_params = sum(p.numel() for p in wm.parameters())
    print(f"[run] name={run_name}  out_dir={out_dir}  wm_params={n_params/1e6:.2f}M", flush=True)

    run = None
    if not args.no_wandb and wandb is not None and os.environ.get("WANDB_API_KEY"):
        # name = the short keyword; every constant/variable goes in the config table.
        run = wandb.init(project=args.wandb_project or os.environ.get("WANDB_PROJECT", "curious-robot"),
                         entity=os.environ.get("WANDB_ENTITY"), name=run_name, group=run_name,
                         dir=str(out_dir),
                         config={**vars(args), "z_dim": z_dim, "a_dim": a_dim,
                                 "n_params_wm": n_params, "device": str(device)})
        print(f"[wandb] {run.url}", flush=True)

    def wlog(d, step):
        if run is not None:
            run.log(d, step=step)

    # --- live per-env state (history is (H, n_envs, .) so resets touch one row) ---
    obs = env.reset()
    z = encode_obs(wm, obs["image"], obs["proprio"], device)        # (n_envs, z_dim)
    hist_z = z.unsqueeze(0).repeat(H, 1, 1)
    hist_a = torch.zeros(H, args.n_envs, a_dim, device=device)
    is_start = np.ones(args.n_envs, bool)
    ep_len = np.zeros(args.n_envs, np.int64)
    ep_ret = np.zeros(args.n_envs, np.float32)

    h_fwd = args.h_fwd_start                          # curriculum horizon
    pred_hist = deque(maxlen=args.flatline_window)    # for the flatline bump trigger
    updates_at_stage = 0
    recent = {k: deque(maxlen=400) for k in
              ("r_cur", "r_safe", "cur_contrib", "contacts", "table_contacts",
               "motion", "ret", "frac_block", "frac_table")}
    recent_mse = {k: deque(maxlen=2000) for k in ("mse_block", "mse_table", "mse_none")}
    t0 = time.time()
    last_wm = last_sac = None
    last_zb = None
    video_on = imageio is not None and args.video_every > 0
    wrist_buf = deque(maxlen=args.video_steps)      # train-video clips (per-env frames, tiled)
    over_buf = deque(maxlen=args.video_steps)
    probe_px = probe_prop = None                    # fixed diverse probe set for encoder/eff_rank_probe
    probe_buf = []                                  # warmup-rollout fallback if the HF probe is unavailable
    if args.probe_size > 0:                         # prefer the canonical uniform-pose probe cached on HF
        loaded = (load_probe_hf(args.hf_repo or os.environ.get("HF_UPLOAD_REPO_ID"), args.probe_id)
                  if not args.no_hf else None)
        if loaded is not None:
            probe_px, probe_prop = loaded[0][:args.probe_size], loaded[1][:args.probe_size]
            print(f"[probe] loaded {len(probe_px)} uniform-pose obs from HF ({args.probe_id})", flush=True)
        else:
            print(f"[probe] HF probe '{args.probe_id}' unavailable; falling back to warmup-rollout probe",
                  flush=True)

    def learner_updates(step, h_fwd):
        """SAC + periodic WM gradient steps on buffered (past) data; returns the
        possibly-bumped h_fwd. Called between env.step_block_async/step_block_wait so
        these GPU updates overlap the env workers rendering the next decision. Update
        count and schedule are identical to the serial loop; they just see the buffer
        minus the single in-flight transition (added after wait) -- negligible off-policy."""
        nonlocal last_wm, last_sac, last_zb, updates_at_stage
        # --- world-model co-training: autoregressive MSE rollout + beta*SIGReg ---
        if step >= args.start_steps and step % args.wm_update_every == 0:
            batch = buf.sample_wm(args.wm_batch_size, H + h_fwd)
            if batch is not None:
                wm.train()
                last_wm = wm_update(wm, sigreg, wm_opt, batch, H, h_fwd,
                                    args.gamma_wm, args.sigreg_weight, device)
                wm.eval()
                pred_hist.append(last_wm[0]); updates_at_stage += 1
                # curriculum: bump H_fwd when pred loss flatlines over the last window
                if (h_fwd < args.h_fwd_max and len(pred_hist) == pred_hist.maxlen
                        and updates_at_stage >= pred_hist.maxlen):
                    arr = np.asarray(pred_hist); half = len(arr) // 2
                    older, newer = arr[:half].mean(), arr[half:].mean()
                    if abs((older - newer) / max(abs(older), 1e-9)) < args.flatline_tol:
                        h_fwd += 1; updates_at_stage = 0; pred_hist.clear()
                        print(f"[curriculum] step={step} H_fwd -> {h_fwd}", flush=True)
        # --- SAC updates (PER; encoder is frozen w.r.t. SAC, z encoded under no_grad) ---
        if step >= args.start_steps and buf.total >= args.batch_size:
            per_beta = min(1.0, args.per_beta_start
                           + (1 - args.per_beta_start) * step / max(args.total_steps, 1))
            for _ in range(args.updates_per_step):
                b = buf.sample_sac(args.batch_size, args.per_alpha, per_beta)
                if b is None:
                    break
                alpha = log_alpha.exp().detach()     # const for critic/actor; tuned separately below
                zb = encode_obs(wm, b["px"], b["prop"], device)
                znb = encode_obs(wm, b["px_n"], b["prop_n"], device)
                with torch.no_grad():
                    an, logpn, _ = actor.sample(znb)
                    q1n, q2n = critic_tgt(znb, an)
                    y = b["r"] + (1 - b["d"]) * args.gamma * (torch.min(q1n, q2n) - alpha * logpn)
                q1, q2 = critic(zb, b["a"])
                critic_loss = (b["w"] * ((q1 - y).pow(2) + (q2 - y).pow(2))).mean()
                critic_opt.zero_grad(); critic_loss.backward(); critic_opt.step()
                if args.per_priority == "td":   # ablation: replace curiosity priority with |TD error|
                    td = (0.5 * (q1 + q2) - y).abs().detach().squeeze(1).cpu().numpy()
                    buf.update_priorities(b["e"], b["i"], td)
                ap, logpp, _ = actor.sample(zb)
                q1p, q2p = critic(zb, ap)
                actor_loss = (alpha * logpp - torch.min(q1p, q2p)).mean()
                actor_opt.zero_grad(); actor_loss.backward(); actor_opt.step()
                if alpha_opt is not None:            # SAC auto-temperature: drive entropy -> target
                    alpha_loss = -(log_alpha * (logpp + target_entropy).detach()).mean()
                    alpha_opt.zero_grad(); alpha_loss.backward(); alpha_opt.step()
                with torch.no_grad():
                    for p, pt in zip(critic.parameters(), critic_tgt.parameters()):
                        pt.mul_(1 - args.tau).add_(args.tau * p)
                last_zb = zb.detach()
                last_sac = (float(critic_loss.item()), float(actor_loss.item()), float(logpp.mean().item()))
        return h_fwd

    for step in range(args.total_steps):
        cur_px, cur_prop = obs["image"], obs["proprio"]          # o_t (before acting)

        # --- act ---
        with torch.no_grad():
            if step < args.start_steps:
                a = torch.rand(args.n_envs, a_dim, device=device) * 2 - 1   # warmup
            else:
                a, _, _ = actor.sample(z)
        hist_a = torch.cat([hist_a[1:], a.unsqueeze(0)], 0)
        a_env = a.detach().cpu().numpy().reshape(args.n_envs, args.action_block, n_dof)

        # --- async actor-learner: launch the env rollout for this decision, then run
        #     the GPU updates on buffered data WHILE the workers render -> overlap CPU/GPU ---
        env.step_block_async(a_env)
        h_fwd = learner_updates(step, h_fwd)
        obs, sub_infos = env.step_block_wait()

        # --- accumulate safety reward + interaction stats over the action_block ---
        r_safe = np.zeros(args.n_envs, np.float32)
        contacts = np.zeros(args.n_envs, np.float32)
        table_contacts = np.zeros(args.n_envs, np.float32)
        motion = np.zeros(args.n_envs, np.float32)
        for info in sub_infos:
            r_safe += info["safety_reward"]
            contacts += info["object_contacts"].astype(np.float32)
            table_contacts += info["table_contacts"].astype(np.float32)
            motion += info["object_motion"]
        r_safe /= args.action_block      # one r_safe per decision (README: Env(a_t) -> r_safe_t)

        # freeze a FIXED diverse probe set (early warmup obs) so encoder/eff_rank_probe
        # measures encoder health independent of how narrow the policy's behavior gets.
        if args.probe_size > 0 and probe_px is None:
            probe_buf.append((obs["image"].copy(), obs["proprio"].copy()))
            if len(probe_buf) * args.n_envs >= args.probe_size:
                probe_px = np.concatenate([p for p, _ in probe_buf])[:args.probe_size]
                probe_prop = np.concatenate([q for _, q in probe_buf])[:args.probe_size]
                probe_buf = None
                print(f"[probe] froze {len(probe_px)} obs for encoder/eff_rank_probe", flush=True)

        z_next = encode_obs(wm, obs["image"], obs["proprio"], device)
        r_cur = curiosity_reward(wm, hist_z, hist_a, z_next).cpu().numpy()        # (n_envs,) >= 0
        cur_term = args.lambda_cur * np.log1p(r_cur)         # lambda_cur * symlog(r_cur)  (r_cur>=0)
        reward = args.lambda_safe * r_safe + cur_term        # r = lambda_safe*r_safe + lambda_cur*symlog(r_cur)

        # contact-conditioned curiosity MSE (r_cur = ||zhat-z||^2): is the model more
        # surprised poking a block than scraping the table or moving in free space?
        # buckets are exclusive, classified block > table > neither (the user's order).
        touch_block = contacts > 0
        touch_table = (~touch_block) & (table_contacts > 0)
        for mask, key in ((touch_block, "mse_block"), (touch_table, "mse_table"),
                          (~(touch_block | touch_table), "mse_none")):
            if mask.any():
                recent_mse[key].extend(r_cur[mask].tolist())

        # --- store transition (truncation-as-done time limit) ---
        ep_len += 1
        ep_ret += reward
        done = (ep_len >= args.max_episode_steps).astype(np.float32)
        buf.add(pixels=cur_px, proprio=cur_prop, action=a.detach().cpu().numpy(),
                r=reward, d=done, is_start=is_start.copy(), prio=r_cur)
        for key, val in (("r_cur", r_cur), ("r_safe", r_safe), ("cur_contrib", cur_term),
                         ("contacts", contacts), ("table_contacts", table_contacts),
                         ("motion", motion), ("ret", reward),
                         ("frac_block", touch_block), ("frac_table", touch_table)):
            recent[key].append(float(np.mean(val)))

        # --- advance latent + history; reset timed-out envs ---
        z = z_next
        hist_z = torch.cat([hist_z[1:], z_next.unsqueeze(0)], 0)
        is_start = done > 0                                       # next o_t is a start where we reset
        done_envs = np.where(done > 0)[0]
        if len(done_envs):
            for e in done_envs:
                o_e = env.reset_one(int(e))
                obs["image"][e] = o_e["image"]
                obs["proprio"][e] = o_e["proprio"]
            z_reset = encode_obs(wm, obs["image"][done_envs], obs["proprio"][done_envs], device)
            z[done_envs] = z_reset
            for j, e in enumerate(done_envs):
                hist_z[:, e] = z_reset[j]
                hist_a[:, e] = 0.0
            if run is not None:
                wlog({"episode/return": float(ep_ret[done_envs].mean()),
                      "episode/len": float(ep_len[done_envs].mean())}, step)
            ep_len[done_envs] = 0
            ep_ret[done_envs] = 0.0

        # --- train videos: buffer per-env frames in the window before each save.
        #     wrist (what the policy sees) is free (already rendered as the obs);
        #     overhead is rendered from the training envs only inside the window
        #     (parallel across workers) -> ~video_steps renders per video_every. ---
        if video_on and 0 < step % args.video_every \
                and step % args.video_every >= args.video_every - args.video_steps:
            wrist_buf.append(tile_frames(obs["image"]))
            over_buf.append(tile_frames(env.render_overhead()))

        # --- logging ---
        if step % args.log_every == 0:
            sps = (step + 1) * args.n_envs / (time.time() - t0)
            safe_m, cur_m = np.mean(recent["r_safe"]), np.mean(recent["cur_contrib"])
            d = {"reward/r_cur": np.mean(recent["r_cur"]),
                 "reward/r_safe": safe_m,
                 "reward/cur_contrib": cur_m,                 # lambda_cur * symlog(r_cur)
                 "reward/safe_cur_ratio": abs(safe_m) / max(abs(cur_m), 1e-6),
                 "reward/total": np.mean(recent["ret"]),
                 "interact/contacts_per_step": np.mean(recent["contacts"]),
                 "interact/table_contacts_per_step": np.mean(recent["table_contacts"]),
                 "interact/object_motion": np.mean(recent["motion"]),
                 "interact/frac_touch_block": np.mean(recent["frac_block"]),
                 "interact/frac_touch_table": np.mean(recent["frac_table"]),
                 "buffer/transitions": buf.total, "perf/steps_per_sec": sps,
                 "wm/h_fwd": h_fwd}
            for key in ("mse_block", "mse_table", "mse_none"):   # curiosity MSE by contact type
                if recent_mse[key]:
                    d[f"wm/{key}"] = float(np.mean(recent_mse[key]))
            if last_zb is not None:
                z_std, eff_rank, feat_corr = collapse_metrics(last_zb)
                d.update({"encoder/z_std": z_std, "encoder/eff_rank": eff_rank,
                          "encoder/feat_corr": feat_corr})
            if probe_px is not None:                          # encoder health on a FIXED diverse probe set
                p_std, p_eff, p_corr = collapse_metrics(encode_obs(wm, probe_px, probe_prop, device))
                d.update({"encoder/eff_rank_probe": p_eff, "encoder/z_std_probe": p_std,
                          "encoder/feat_corr_probe": p_corr})
            if last_wm is not None:
                d.update({"wm/pred_loss": last_wm[0], "wm/sigreg": last_wm[1],
                          "wm/identity_baseline": last_wm[2]})
            if last_sac is not None:
                d.update({"sac/critic_loss": last_sac[0], "sac/actor_loss": last_sac[1],
                          "sac/alpha": float(log_alpha.exp().item()),
                          "sac/policy_entropy": -last_sac[2], "sac/target_entropy": target_entropy})
            wlog(d, step)
            with open(out_dir / "metrics.jsonl", "a") as f:    # local metrics record (esp. when --no-wandb)
                f.write(json.dumps({"step": step, **{k: float(v) for k, v in d.items()}}) + "\n")
            print(f"[step {step}] r_safe={safe_m:.3f} cur_contrib={cur_m:.3f} "
                  f"safe:cur={d['reward/safe_cur_ratio']:.2f} "
                  f"contacts/s={d['interact/contacts_per_step']:.2f} "
                  f"mse[blk/tbl/none]={d.get('wm/mse_block', float('nan')):.2f}/"
                  f"{d.get('wm/mse_table', float('nan')):.2f}/{d.get('wm/mse_none', float('nan')):.2f} "
                  f"h_fwd={h_fwd} sps={sps:.1f}", flush=True)

        # --- checkpoint: upload to HF then clear from disk (bounded disk over a long run) ---
        if args.save_every > 0 and step > 0 and step % args.save_every == 0:
            state = {"step": step, "wm": wm.state_dict(), "actor": actor.state_dict(),
                     "critic": critic.state_dict(), "critic_tgt": critic_tgt.state_dict(),
                     "log_alpha": log_alpha.detach().cpu(), "h_fwd": h_fwd, "args": vars(args)}
            save_and_upload(state, out_dir, step,
                            args.hf_repo or os.environ.get("HF_UPLOAD_REPO_ID"),
                            run_name, not args.no_hf, args.keep_local_ckpts)

        # --- train videos: save the buffered wrist + overhead clips (every video_every) ---
        if video_on and step > 0 and step % args.video_every == 0:
            roll_dir = out_dir / "rollouts"; roll_dir.mkdir(exist_ok=True)
            for tag, frames in (("wrist", wrist_buf), ("overhead", over_buf)):
                if not frames:
                    continue
                vp = roll_dir / f"train_{tag}_{step:07d}.mp4"
                try:
                    imageio.mimsave(vp, list(frames), fps=args.video_fps)
                    if run is not None:
                        wlog({f"train/{tag}": wandb.Video(str(vp), format="mp4")}, step)
                        vp.unlink(missing_ok=True)   # in W&B now; keep local disk clean
                except Exception as ex:
                    print(f"[video] {tag} failed (non-fatal): {ex}", flush=True)
            wrist_buf.clear(); over_buf.clear()

    # --- final checkpoint ---
    state = {"step": args.total_steps, "wm": wm.state_dict(), "actor": actor.state_dict(),
             "critic": critic.state_dict(), "critic_tgt": critic_tgt.state_dict(),
             "log_alpha": log_alpha.detach().cpu(), "h_fwd": h_fwd, "args": vars(args)}
    save_and_upload(state, out_dir, args.total_steps,
                    args.hf_repo or os.environ.get("HF_UPLOAD_REPO_ID"),
                    run_name, not args.no_hf, args.keep_local_ckpts)
    env.close()
    if run is not None:
        run.finish()
    print("[done]", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description="Curious Robot: JEPA+SIGReg WM + SAC curiosity on SO-ARM101")
    # schedule / infra
    p.add_argument("--total-steps", type=int, default=200_000)
    p.add_argument("--start-steps", type=int, default=1000, help="random-action warmup (decision steps)")
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--env-threads", type=int, default=0,
                   help=">0 steps envs on a thread pool (inproc backend only)")
    p.add_argument("--env-backend", choices=("subproc", "inproc"), default="subproc",
                   help="subproc: each env in a CUDA-free worker process, needed on "
                        "GPU+EGL to avoid the MuJoCo-render/CUDA SIGABRT; "
                        "inproc: envs in this process (sequential or --env-threads)")
    p.add_argument("--frame-skip", type=int, default=6)
    p.add_argument("--max-episode-steps", type=int, default=200, help="decision steps before truncation-as-done")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default="baseline",
                   help="short run keyword; drives the W&B run name, runs/<name>/, and HF "
                        "<name>/ckpt_*.pt. Keep it short and identifiable -- every constant/var "
                        "lives in the W&B config table, not the name.")
    p.add_argument("--out-dir", default=None, help="local dir (default: runs/<name>)")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=1000,
                   help="HF-upload-then-clear-disk period (decision steps)")
    p.add_argument("--keep-local-ckpts", action="store_true",
                   help="keep the local .pt after a successful upload (default: delete to bound disk)")
    p.add_argument("--video-every", type=int, default=1000,
                   help="train-video period (decision steps): save a wrist + overhead clip every N; 0 disables")
    p.add_argument("--video-steps", type=int, default=60,
                   help="frames per train-video clip (window of decision steps before each save)")
    p.add_argument("--video-fps", type=int, default=20)
    p.add_argument("--probe-size", type=int, default=256,
                   help="size of the fixed probe set for encoder/eff_rank_probe (isolates encoder health "
                        "from behavioral diversity); 0 disables")
    p.add_argument("--probe-id", default="probe_v1",
                   help="HF probe artifact id (probe/<id>.npz): canonical uniform-pose probe; "
                        "falls back to a warmup-rollout probe if unavailable")
    # action / actuation (README)
    p.add_argument("--action-block", type=int, default=5)
    p.add_argument("--action-max", type=float, default=0.3)
    p.add_argument("--dq-max", type=float, default=100.0, help="README dq^max (~inf; clip rarely binds)")
    # world model (README; the '?' values below are sweepable, not pinned in README)
    p.add_argument("--history-size", type=int, default=3, help="H_bwd")
    p.add_argument("--h-fwd-start", type=int, default=1)
    p.add_argument("--h-fwd-max", type=int, default=1,
                   help="max forward rollout horizon; ==start (1) pins the WM to 1-step-ahead "
                        "prediction and disables the H_fwd curriculum")
    p.add_argument("--gamma-wm", type=float, default=0.95)
    p.add_argument("--sigreg-weight", type=float, default=0.3,
                   help="beta: SIGReg (isotropic-Gaussian) weight, pinned at 0.3")
    p.add_argument("--wm-batch-size", type=int, default=128)
    p.add_argument("--wm-lr", type=float, default=5e-5)
    p.add_argument("--wm-update-every", type=int, default=4)
    p.add_argument("--wm-dropout", type=float, default=0.1)
    p.add_argument("--wm-grad-checkpoint", action="store_true",
                   help="enable ViT gradient checkpointing in the WM update (default off; trades ~10-15ms "
                        "recompute for memory — only worth it if WM-update activation memory is tight)")
    p.add_argument("--flatline-window", type=int, default=200)
    p.add_argument("--flatline-tol", type=float, default=0.03)
    # reward ('?' values; sweepable)
    p.add_argument("--lambda-safe", type=float, default=0.0,
                   help="weight on the safety penalty r_safe in the reward; 0 (default) ablates safety "
                        "(reward = pure curiosity), set 1.0 for the README r = r_safe + lambda_cur*symlog(r_cur)")
    p.add_argument("--lambda-cur", type=float, default=1.0,
                   help="curiosity weight on symlog(r_cur). r_cur is the per-dim MEAN squared pred error "
                        "(~O(0.1-1)), so lambda_cur~1 keeps cur_term O(1) in symlog's sensitive region. README '?'")
    p.add_argument("--safety-delta", type=float, default=0.05, help="delta deadband (README '?'; sweep)")
    # SAC (README)
    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument("--alpha", type=float, default=0.2, help="entropy temperature (initial value if --auto-alpha)")
    p.add_argument("--auto-alpha", action="store_true",
                   help="learn the entropy temperature (standard SAC auto-tuning toward --target-entropy)")
    p.add_argument("--alpha-lr", type=float, default=3e-4, help="lr for the learnable log-alpha")
    p.add_argument("--target-entropy", type=float, default=None,
                   help="SAC target entropy (default: -action_dim = -n_dof*action_block)")
    p.add_argument("--tau", type=float, default=0.005, help="Polyak rate")
    p.add_argument("--actor-lr", type=float, default=3e-4)
    p.add_argument("--critic-lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--updates-per-step", type=int, default=1)
    p.add_argument("--per-alpha", type=float, default=0.6)
    p.add_argument("--per-beta-start", type=float, default=0.4)
    p.add_argument("--per-priority", choices=["curiosity", "td"], default="curiosity",
                   help="SAC replay priority. 'curiosity' (default) = 1-step pred error r_cur; "
                        "'td' (ablation) = |TD error|, sign-agnostic so it also replays unsafe "
                        "transitions the critic mispredicts. See results.md.")
    p.add_argument("--buffer-frac", type=float, default=0.1, help="cap = clip(frac*total, 1e3, 5e4)")
    # logging backends (keys from .env)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--no-hf", action="store_true", help="disable HF checkpoint upload")
    p.add_argument("--hf-repo", default=None, help="HF repo id (else $HF_UPLOAD_REPO_ID)")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
