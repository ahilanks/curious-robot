"""Curious Robot — from-scratch JEPA+SIGReg world model co-trained with SAC under
an intrinsic curiosity reward, on the MuJoCo SO-ARM101 (README is the spec).

Per decision step (action_block env steps), for every parallel env:
  1. encode z_t from (wrist image, proprio) with the current world model
  2. sample a_t ~ pi(.|z_t); apply it (delta-target PD) over action_block env steps
  3. curiosity   r_cur = ||f(z_{t-H+1:t}, a_t) - z_{t+1}||^2     (1-step pred error)
     reward      r_t   = sum_k r_safe_k + lambda_cur * symlog(r_cur)
  4. store (o_t, q_t, qdot_t, a_t, r_t, o_{t+1}) with PER priority = r_cur
Periodically: co-train the WM (autoregressive symlog rollout loss + beta*SIGReg, with
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
from model.proprio import symlog                     # noqa: E402
from env.parallel_env import (VectorMujocoEnv,       # noqa: E402
                              SubprocVectorMujocoEnv, SubprocSingleEnv)
from env.mujoco_env import MujocoSO101Env            # noqa: E402

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


class RunningMeanStd:
    """EMA running mean/std (bias-corrected, like Adam's moments) for normalizing the
    intrinsic reward. The README scales curiosity with symlog (lambda_cur*log1p(r_cur)),
    but at the z_dim-driven r_cur floor (~250) symlog sits on its flat tail: it crushes
    the *per-state spread* (a +/-30 swing in r_cur -> ~+/-0.12 in symlog), so SAC sees a
    near-constant curiosity term and stops exploring. Standardizing by a running mean/std
    restores that spread at any scale; EMA (vs infinite-count Welford) tracks the drift as
    the WM learns and r_cur falls. Opt-in via --normalize-curiosity; default off keeps the
    exact README reward. update() consumes one (n_envs,) batch per decision step."""

    def __init__(self, momentum=0.999, eps=1e-8):
        self.m, self.eps = momentum, eps
        self._mean = 0.0; self._sq = 0.0; self._t = 0

    def update(self, x):
        x = np.asarray(x, np.float64)
        self._mean = self.m * self._mean + (1 - self.m) * x.mean()
        self._sq = self.m * self._sq + (1 - self.m) * (x * x).mean()
        self._t += 1

    @property
    def mean(self):
        c = 1 - self.m ** self._t if self._t else 1.0      # bias correction
        return self._mean / max(c, self.eps)

    @property
    def std(self):
        c = 1 - self.m ** self._t if self._t else 1.0
        mean = self._mean / max(c, self.eps)
        var = max(self._sq / max(c, self.eps) - mean * mean, 0.0)
        return float(np.sqrt(var) + self.eps)


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
    """r_cur = ||f(z_{t-H+1:t}, a_t)[-1] - z_{t+1}||^2, per env. hist_z/hist_a are
    (H, B, .) tensors; returns (B,) squared L2 prediction error."""
    z_ctx = hist_z.transpose(0, 1)                       # (B, H, D)
    a_emb = wm.action_encoder(hist_a.transpose(0, 1))    # (B, H, A_emb)
    pred = wm.predict(z_ctx, a_emb)[:, -1]               # (B, D)
    return (pred - z_next).pow(2).sum(-1)


def wm_update(wm, sigreg, opt, batch, H_bwd, h, gamma_wm, beta, device):
    """One AdamW step on L_wm = discounted symlog autoregressive rollout + beta*SIGReg."""
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
        sq = (zhat - z_k).pow(2).sum(-1)                   # ||.||^2 per sample
        pred_loss = pred_loss + g * symlog(sq).mean()
        wsum += g
        if k < h:
            roll = torch.cat([roll[:, 1:], zhat.unsqueeze(1)], 1)
            roll_a = torch.cat([roll_a[:, 1:], a_emb[:, H_bwd - 1 + k].unsqueeze(1)], 1)
            cur = wm.predict(roll, roll_a)
    pred_loss = pred_loss / wsum
    with torch.no_grad():   # persistence baseline on the SAME discounted h-step schedule
        z_last = emb[:, H_bwd - 1]
        idl = sum((gamma_wm ** k) * symlog((z_last - emb[:, H_bwd - 1 + k]).pow(2).sum(-1)).mean()
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


@torch.no_grad()
def record_rollout(actor, wm, eval_env, action_block, n_dof, device, out_dir, tag, n_steps, fps):
    """Policy rollout in eval_env, written as TWO videos: the overhead cam (what we
    watch) and the wrist cam (what the policy actually sees). Returns {cam: path}
    for whatever got written ({} if imageio is unavailable)."""
    if imageio is None:
        return {}
    obs = eval_env.reset()
    over = [eval_env.render_overhead()]
    wrist = [obs["image"]]
    for _ in range(n_steps):
        z = encode_obs(wm, obs["image"][None], obs["proprio"][None], device)
        a, _, _ = actor.sample(z)
        a = a.squeeze(0).cpu().numpy().reshape(action_block, n_dof)
        for k in range(action_block):
            obs, _ = eval_env.step(a[k])
            over.append(eval_env.render_overhead())
            wrist.append(obs["image"])
    out_dir = Path(out_dir)
    paths = {}
    for name, frames in (("overhead", over), ("wrist", wrist)):
        p = out_dir / f"{name}_{tag}.mp4"
        imageio.mimsave(p, frames, fps=fps)
        paths[name] = p
    return paths


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
    EvalEnv = SubprocSingleEnv if args.env_backend == "subproc" else MujocoSO101Env
    eval_env = EvalEnv(action_max=args.action_max, dq_max=args.dq_max,
                       safety_delta=args.safety_delta, seed=args.seed + 9999)
    n_dof = env.n_dof
    a_dim = n_dof * args.action_block
    prop_dim = 3 * n_dof
    H = args.history_size

    wm = WorldModel(n_dof=n_dof, action_block=args.action_block,
                    history_size=H, dropout=args.wm_dropout).to(device)
    try:  # bound WM-update activation memory so the H_fwd curriculum can grow
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
    # entropy temperature: fixed --alpha (default), or auto-tuned to --target-entropy (--learn-alpha)
    log_alpha = torch.tensor(float(np.log(args.alpha)), device=device, requires_grad=args.learn_alpha)
    target_entropy = args.target_entropy if args.target_entropy is not None else -float(a_dim)
    alpha_opt = torch.optim.Adam([log_alpha], lr=args.alpha_lr) if args.learn_alpha else None

    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
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
    rms_cur = RunningMeanStd(momentum=args.cur_norm_momentum)   # curiosity reward normalizer (opt-in)
    snap = None                 # active env-0 training-snapshot capture window (started at --video-every)
    t0 = time.time()
    last_wm = last_sac = None
    last_zb = None

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

        # --- action_block env steps; accumulate safety reward + interaction stats ---
        r_safe = np.zeros(args.n_envs, np.float32)
        contacts = np.zeros(args.n_envs, np.float32)
        table_contacts = np.zeros(args.n_envs, np.float32)
        motion = np.zeros(args.n_envs, np.float32)
        for k in range(args.action_block):
            obs, info = env.step(a_env[:, k])
            r_safe += info["safety_reward"]
            contacts += info["object_contacts"].astype(np.float32)
            table_contacts += info["table_contacts"].astype(np.float32)
            motion += info["object_motion"]
            if snap is not None:                  # training-run snapshot: env 0, every frame
                snap["overhead"].append(env.render_overhead_one(0))
                snap["wrist"].append(obs["image"][0])
        r_safe /= args.action_block      # one r_safe per decision (README: Env(a_t) -> r_safe_t)
        if snap is not None:                      # close the snapshot window after video_steps decisions
            snap["left"] -= 1
            if snap["left"] <= 0:
                if imageio is not None and run is not None:
                    sdir = out_dir / "rollouts"; sdir.mkdir(exist_ok=True)
                    for cam in ("overhead", "wrist"):
                        try:
                            vp = sdir / f"train_{cam}_{snap['tag']}.mp4"
                            imageio.mimsave(vp, snap[cam], fps=args.video_fps)
                            wlog({f"train/{cam}": wandb.Video(str(vp), format="mp4")}, step)
                            vp.unlink(missing_ok=True)
                        except Exception as ex:
                            print(f"[train-video] failed (non-fatal): {ex}", flush=True)
                snap = None

        z_next = encode_obs(wm, obs["image"], obs["proprio"], device)
        r_cur = curiosity_reward(wm, hist_z, hist_a, z_next).cpu().numpy()        # (n_envs,) >= 0
        if args.raw_curiosity:          # no transform at all: cur_term = lambda_cur * r_cur (raw, unbounded)
            cur_term = args.lambda_cur * r_cur
        elif args.normalize_curiosity:  # de-saturate: standardize r_cur by a running mean/std (clip a la RND)
            rms_cur.update(r_cur)
            cur_term = args.lambda_cur * np.clip((r_cur - rms_cur.mean) / rms_cur.std, -5.0, 5.0)
        else:
            cur_term = args.lambda_cur * np.log1p(r_cur)     # README default: lambda_cur * symlog(r_cur)  (r_cur>=0)
        reward = args.safety_weight * r_safe + cur_term      # README: r = r_safe + lambda_cur*symlog(r_cur) (safety_weight=1)

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

        # --- world-model co-training: autoregressive symlog rollout + beta*SIGReg ---
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
                alpha = log_alpha.exp().detach()      # recompute: auto-tuned α can change each update
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
                if args.learn_alpha:                  # SAC dual: auto-tune α toward target_entropy
                    alpha_loss = (-log_alpha * (logpp.detach() + target_entropy)).mean()
                    alpha_opt.zero_grad(); alpha_loss.backward(); alpha_opt.step()
                    if args.min_alpha > 0.0:
                        with torch.no_grad():
                            log_alpha.clamp_(min=float(np.log(args.min_alpha)))
                with torch.no_grad():
                    for p, pt in zip(critic.parameters(), critic_tgt.parameters()):
                        pt.mul_(1 - args.tau).add_(args.tau * p)
                last_zb = zb.detach()
                last_sac = (float(critic_loss.item()), float(actor_loss.item()), float((-logpp).mean().item()))

        # --- logging ---
        if step % args.log_every == 0:
            sps = (step + 1) * args.n_envs / (time.time() - t0)
            safe_m, cur_m = np.mean(recent["r_safe"]), np.mean(recent["cur_contrib"])
            d = {"reward/r_cur": np.mean(recent["r_cur"]),
                 "reward/r_safe": safe_m,
                 "reward/cur_contrib": cur_m,                 # lambda_cur * symlog(r_cur)
                 "reward/safe_cur_ratio": abs(args.safety_weight * safe_m) / max(abs(cur_m), 1e-6),
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
            if args.normalize_curiosity:    # watch de-saturation: running r_cur mean/std
                d["reward/cur_norm_mean"] = rms_cur.mean
                d["reward/cur_norm_std"] = rms_cur.std
            if last_zb is not None:
                z_std, eff_rank, feat_corr = collapse_metrics(last_zb)
                d.update({"encoder/z_std": z_std, "encoder/eff_rank": eff_rank,
                          "encoder/feat_corr": feat_corr})
            if last_wm is not None:
                d.update({"wm/pred_loss": last_wm[0], "wm/sigreg": last_wm[1],
                          "wm/identity_baseline": last_wm[2]})
            if last_sac is not None:
                d.update({"sac/critic_loss": last_sac[0], "sac/actor_loss": last_sac[1],
                          "sac/alpha": float(log_alpha.exp().item()), "sac/policy_entropy": last_sac[2]})
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

        # --- periodic policy rollout videos (overhead + wrist cams) ---
        if args.video_every > 0 and step > 0 and step % args.video_every == 0:
            roll_dir = out_dir / "rollouts"; roll_dir.mkdir(exist_ok=True)
            try:
                paths = record_rollout(actor, wm, eval_env, args.action_block, n_dof, device,
                                       roll_dir, f"step_{step:07d}", args.video_steps, args.video_fps)
                for cam, vp in paths.items():
                    if run is not None:
                        wlog({f"rollout/{cam}": wandb.Video(str(vp), format="mp4")}, step)
                        vp.unlink(missing_ok=True)   # in W&B now; keep local disk clean
            except Exception as ex:
                print(f"[video] failed (non-fatal): {ex}", flush=True)
            if snap is None:        # also capture an env-0 training-run snapshot over the next decisions
                snap = {"overhead": [], "wrist": [], "left": args.video_steps, "tag": f"step_{step:07d}"}

    # --- final checkpoint ---
    state = {"step": args.total_steps, "wm": wm.state_dict(), "actor": actor.state_dict(),
             "critic": critic.state_dict(), "critic_tgt": critic_tgt.state_dict(),
             "log_alpha": log_alpha.detach().cpu(), "h_fwd": h_fwd, "args": vars(args)}
    save_and_upload(state, out_dir, args.total_steps,
                    args.hf_repo or os.environ.get("HF_UPLOAD_REPO_ID"),
                    run_name, not args.no_hf, args.keep_local_ckpts)
    env.close(); eval_env.close()
    if run is not None:
        run.finish()
    print("[done]", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description="Curious Robot: JEPA+SIGReg WM + SAC curiosity on SO-ARM101")
    # schedule / infra
    p.add_argument("--total-steps", type=int, default=200_000)
    p.add_argument("--start-steps", type=int, default=200,
                   help="random-action warmup (decision steps) before WM/SAC updates start. The PER "
                        "buffer is small (fills in ~30 steps), so 200 fills it many times over while "
                        "avoiding a long GPU-idle warmup; raise it for more pre-training random data.")
    p.add_argument("--n-envs", type=int, default=32,
                   help="parallel envs. CPU/render-bound: more envs = more parallel data per step, "
                        "NOT a faster fixed-length run (per-step cost grows). Keep BELOW CPU "
                        "saturation (~cores/4.5) or env renders compete & it slows sharply; 32 suits "
                        "a ~224-core box (~144 cores), ~48 nears its edge (~216). "
                        "Run `python src/hw_config.py` for a hardware-specific recommendation.")
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
    p.add_argument("--video-every", type=int, default=500,
                   help="overhead+wrist rollout video period (decision steps)")
    p.add_argument("--video-steps", type=int, default=60)
    p.add_argument("--video-fps", type=int, default=20)
    # action / actuation (README)
    p.add_argument("--action-block", type=int, default=5)
    p.add_argument("--action-max", type=float, default=0.3)
    p.add_argument("--dq-max", type=float, default=100.0, help="README dq^max (~inf; clip rarely binds)")
    # world model (README; the '?' values below are sweepable, not pinned in README)
    p.add_argument("--history-size", type=int, default=3, help="H_bwd")
    p.add_argument("--h-fwd-start", type=int, default=1)
    p.add_argument("--h-fwd-max", type=int, default=1,
                   help="max WM rollout horizon. Default 1 = single-step prediction, NO horizon "
                        "curriculum (matches the le-wm reference, which trains with num_preds=1). "
                        "Raise it (e.g. --h-fwd-max 20) to enable the flatline-triggered "
                        "autoregressive curriculum that grows H_fwd from --h-fwd-start up to this cap.")
    p.add_argument("--gamma-wm", type=float, default=0.95)
    p.add_argument("--sigreg-weight", type=float, default=0.9, help="beta (README '?'; sweep)")
    p.add_argument("--wm-batch-size", type=int, default=128)
    p.add_argument("--wm-lr", type=float, default=5e-5)
    p.add_argument("--wm-update-every", type=int, default=4)
    p.add_argument("--wm-dropout", type=float, default=0.1)
    p.add_argument("--flatline-window", type=int, default=200)
    p.add_argument("--flatline-tol", type=float, default=0.03)
    # reward ('?' values; sweepable)
    p.add_argument("--lambda-cur", type=float, default=15.0,
                   help="curiosity weight; ~15 puts safety:curiosity ~0.5:1 at observed magnitudes "
                        "(|r_safe|~40, symlog(r_cur)~5.6). README '?'; sweep & watch reward/safe_cur_ratio")
    p.add_argument("--safety-delta", type=float, default=0.05, help="delta deadband (README '?'; sweep)")
    p.add_argument("--safety-weight", type=float, default=1.0,
                   help="multiplier on r_safe in the reward (r = safety_weight*r_safe + lambda_cur*cur). "
                        "Default 1.0 = README. Lower it (<1) to de-emphasize the jerk penalty so curiosity "
                        "can drive more interaction. (delta barely affects firing — this is the real knob.)")
    p.add_argument("--normalize-curiosity", action="store_true",
                   help="standardize the curiosity reward by a running (EMA) mean/std instead of "
                        "lambda_cur*symlog(r_cur). De-saturates curiosity (symlog crushes per-state "
                        "spread at the r_cur floor) so SAC keeps exploring. Default off = exact README reward.")
    p.add_argument("--cur-norm-momentum", type=float, default=0.999,
                   help="EMA retention for the curiosity running mean/std (only with --normalize-curiosity)")
    p.add_argument("--raw-curiosity", action="store_true",
                   help="use the RAW r_cur as the curiosity reward (cur_term = lambda_cur*r_cur): no symlog, "
                        "no normalization, no clip. Preserves the full per-state spread but keeps the large "
                        "r_cur DC floor (~240) and is unbounded — pair with a small --lambda-cur (and watch "
                        "sac/policy_entropy: at this reward scale a fixed --alpha is negligible -> near-greedy "
                        "policy). Takes precedence over --normalize-curiosity.")
    # SAC (README)
    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument("--alpha", type=float, default=0.2, help="fixed entropy temperature (used when not --learn-alpha)")
    p.add_argument("--learn-alpha", action="store_true",
                   help="auto-tune entropy temperature α toward --target-entropy (SAC dual). Default off "
                        "= fixed --alpha. Keeps the policy from collapsing to deterministic (more exploration).")
    p.add_argument("--alpha-lr", type=float, default=3e-4, help="learning rate for log α (with --learn-alpha)")
    p.add_argument("--target-entropy", type=float, default=None,
                   help="target policy entropy for --learn-alpha (default: -action_dim)")
    p.add_argument("--min-alpha", type=float, default=0.0, help="floor on α when auto-tuning (0 = no floor)")
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
