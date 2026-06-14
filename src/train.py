"""Curious Robot — from-scratch JEPA+SIGReg world model co-trained with SAC under
an intrinsic curiosity reward, on the MuJoCo SO-ARM101 (README is the spec).

Per decision step (action_block env steps), for every parallel env:
  1. encode z_t from (wrist image, proprio) with the current world model
  2. act a_t = pi(z_t) (deterministic tanh policy); apply it (delta-target PD) over action_block env steps
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
from src.probe import load_probe_hf                  # noqa: E402
# Env backends are imported lazily in main() so the `hardware` backend does not require mujoco.

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


# ----------------------------------------------------- actor-critic (deterministic)
class Actor(nn.Module):
    """Deterministic policy a = tanh(MLP(z)). The Gaussian head, sampling and entropy
    machinery were removed 2026-06-12 (user decision: deterministic policy only);
    exploration comes from the curiosity reward, not action noise. `mean` keeps its
    name so pre-removal checkpoints load via load_actor_state."""

    def __init__(self, z_dim, a_dim, hidden=256):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(z_dim, hidden), nn.ReLU(),
                                   nn.Linear(hidden, hidden), nn.ReLU())
        self.mean = nn.Linear(hidden, a_dim)

    def forward(self, z):
        return torch.tanh(self.mean(self.trunk(z)))


def load_actor_state(actor, sd):
    """Load an actor state_dict, tolerating pre-2026-06-12 stochastic-SAC checkpoints:
    their log_std.* head is dropped; everything else must match exactly."""
    actor.load_state_dict({k: v for k, v in sd.items() if not k.startswith("log_std")})


class TwinQ(nn.Module):
    """Twin Q critics. n_out=1 is the scalar critic; n_out=K (--multihead-q) gives one
    head per weighted reward component, each trained on its own per-component TD target
    with the SHARED next action — the actor still maximizes the sum over heads, so the
    optimum matches the scalar critic; the heads exist to make the value decomposition
    inspectable (which reward component is steering the policy)."""

    def __init__(self, z_dim, a_dim, hidden=256, n_out=1):
        super().__init__()
        mk = lambda: nn.Sequential(nn.Linear(z_dim + a_dim, hidden), nn.ReLU(),
                                   nn.Linear(hidden, hidden), nn.ReLU(),
                                   nn.Linear(hidden, n_out))
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

    def __init__(self, n_envs, cap_per_env, img_hw, a_dim, prop_dim, device, n_comp=0):
        self.n_envs, self.C, self.device = n_envs, cap_per_env, device
        s = (n_envs, cap_per_env)
        self.pixels = np.zeros((*s, img_hw, img_hw, 3), np.uint8)
        self.proprio = np.zeros((*s, prop_dim), np.float32)
        self.action = np.zeros((*s, a_dim), np.float32)
        self.r = np.zeros(s, np.float32)
        self.rc = np.zeros((*s, n_comp), np.float32) if n_comp else None   # per-component rewards (--multihead-q)
        self.d = np.zeros(s, np.float32)
        self.is_start = np.zeros(s, bool)
        self.prio = np.zeros(s, np.float64)
        self.head = np.zeros(n_envs, np.int64)
        self.count = np.zeros(n_envs, np.int64)

    def add(self, pixels, proprio, action, r, d, is_start, prio, rc=None):
        for e in range(self.n_envs):
            i = self.head[e]
            self.pixels[e, i] = pixels[e]
            self.proprio[e, i] = proprio[e]
            self.action[e, i] = action[e]
            self.r[e, i] = r[e]
            if self.rc is not None:
                self.rc[e, i] = rc[e]
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
        out = {
            "px": self.pixels[e, i], "prop": self.proprio[e, i],
            "px_n": self.pixels[e, ni], "prop_n": self.proprio[e, ni],
            "a": t(self.action[e, i]), "r": t(self.r[e, i])[:, None],
            "d": t(self.d[e, i])[:, None], "w": t(w)[:, None],
            "e": e, "i": i,                          # for optional TD-error priority writeback
        }
        if self.rc is not None:
            out["rc"] = t(self.rc[e, i])             # (B, K) per-component rewards
        return out

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


REWARD_COMPONENTS = ("cur", "safe", "rate", "energy")   # --multihead-q head order


def scrub_torque_obs(obs, n_dof):
    """--no-torque-obs: zero the u^app slice of proprio in place (obs -> [q, qd, 0]).
    Keeps the proprio/encoder shapes (old ckpts still load) while removing the channel
    that is a near-constant saturated sign bit on hardware (2026-06-11: 96-97% of
    joint-samples pegged at +/-3.35 under the kp-law recompute) and mostly saturated
    in sim too — the main sim->real obs-distribution mismatch."""
    obs["proprio"][..., 2 * n_dof:3 * n_dof] = 0.0
    return obs


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


def grad_caps_temporal_loss(traj, eps, valid=None):
    """Grad-CAPS displacement-normalized temporal-smoothness penalty (actor-only).

    traj: (B, L, n_dof) — consecutive policy sub-actions along time (the real applied
    path; here L = 2*action_block, pi(z_t) concatenated with pi(z_{t+1})). For each
    interior triple (s_{k-1}, s_k, s_{k+1}):

        acc_k  = ||s_{k-1} - 2 s_k + s_{k+1}||_2     # == ||Da_t - Da_{t+1}||_2 (curvature)
        disp_k = ||s_{k+1} - s_{k-1}||_2             # net displacement across the window
        L_k    = acc_k * tanh( 1 / (disp_k + eps) )

    A smooth ramp has acc≈0 -> ≈0 loss at ANY speed (wide motion is free); an in-place
    zigzag has large acc and tiny disp -> tanh(1/eps)≈1 -> curvature paid in full. disp is
    a SCALAR magnitude per window (norm over joints), NOT a per-dim reciprocal: a parked
    joint's 1/eps would otherwise dominate ||1/(d+eps)|| and saturate every window. eps caps
    the 1/disp blow-up; tanh keeps the factor in [0,1). Norms carry +1e-12 so the gradient
    is finite at acc=0 / disp=0. `valid` (B, L-2) masks windows (e.g. the join straddling an
    episode reset) before the mean."""
    s0, s1, s2 = traj[:, :-2], traj[:, 1:-1], traj[:, 2:]            # (B, L-2, n_dof)
    acc = ((s0 - 2.0 * s1 + s2).pow(2).sum(-1) + 1e-12).sqrt()       # (B, L-2) curvature
    disp = ((s2 - s0).pow(2).sum(-1) + 1e-12).sqrt()                 # (B, L-2) net displacement
    per_window = acc * torch.tanh(1.0 / (disp + eps))               # (B, L-2)
    if valid is None:
        return per_window.mean()
    return (per_window * valid).sum() / valid.sum().clamp_min(1.0)


def sac_update(buf, wm, actor, critic, critic_tgt, actor_opt, critic_opt,
               args, step, device):
    """Run args.updates_per_step deterministic actor-critic gradient steps on PER
    samples from buf (TD3-style: twin critics + Polyak target net, but no entropy
    term and no action sampling — stochasticity removed 2026-06-12; the name stays
    for caller/W&B-key continuity). The encoder is frozen w.r.t. these updates: z is
    encoded under no_grad from the CURRENT wm.
    Returns {"critic_loss", "actor_loss", "zb"} from the last completed update, or
    None if the gate is closed (warmup / buffer below batch) or sampling came up dry.
    Shared by the online loop here and offline fine-tuning (offline_train.py)."""
    if step < args.start_steps or buf.total < args.batch_size:
        return None
    per_beta = min(1.0, args.per_beta_start
                   + (1 - args.per_beta_start) * step / max(args.total_steps, 1))
    out = None
    for _ in range(args.updates_per_step):
        b = buf.sample_sac(args.batch_size, args.per_alpha, per_beta)
        if b is None:
            break
        zb = encode_obs(wm, b["px"], b["prop"], device)
        znb = encode_obs(wm, b["px_n"], b["prop_n"], device)
        with torch.no_grad():
            q1n, q2n = critic_tgt(znb, actor(znb))
            # scalar critic: (B,1) target as before. multihead: per-head TD targets from
            # the per-component rewards, min-over-twins applied per head (pessimism per
            # component); the shared next action keeps the heads' sum == the scalar value.
            y = b.get("rc", b["r"]) + (1 - b["d"]) * args.gamma * torch.min(q1n, q2n)
        q1, q2 = critic(zb, b["a"])
        critic_loss = (b["w"] * ((q1 - y).pow(2) + (q2 - y).pow(2)).mean(-1, keepdim=True)).mean()
        critic_opt.zero_grad(); critic_loss.backward(); critic_opt.step()
        if args.per_priority == "td":   # ablation: replace curiosity priority with |TD error|
            td = (0.5 * (q1 + q2) - y).sum(-1).abs().detach().cpu().numpy()
            buf.update_priorities(b["e"], b["i"], td)
        ap = actor(zb)
        q1p, q2p = critic(zb, ap)
        actor_loss = (-torch.min(q1p.sum(-1), q2p.sum(-1))).mean()   # sum heads, then twin-min (== old path at K=1)
        if getattr(args, "actor_rate_reg", 0) > 0:
            # action-rate as an actor-loss regularizer instead of a reward term: penalize
            # the policy's own within-block sub-action jerk directly — smoothness pressure
            # that never enters r or propagates through Q (the curiosity balance untouched).
            sub = ap.view(ap.shape[0], args.action_block, -1)
            actor_loss = actor_loss + args.actor_rate_reg * sub.diff(dim=1).pow(2).mean()
        gc_val = 0.0
        if getattr(args, "lambda_temp", 0) > 0:
            # Grad-CAPS temporal smoothness (displacement-normalized): penalize the CURVATURE
            # of the policy's real applied sub-action path across the t->t+1 decision boundary
            # [pi(z_t) | pi(z_{t+1})], scaled by 1/displacement so a smooth wide ramp is free
            # and only low-travel zigzag (the in-place jitter that cheaply satisfies 1-step
            # curiosity) is paid. Actor-only: gradients flow through pi, Q/r untouched.
            nb = args.action_block
            apn = actor(znb)                                          # next-state block (WITH grad)
            traj = torch.cat([ap.view(ap.shape[0], nb, -1),
                              apn.view(apn.shape[0], nb, -1)], dim=1)  # (B, 2*nb, n_dof)
            vmask = torch.ones(traj.shape[0], 2 * nb - 2, device=traj.device)
            done = b["d"].view(-1) > 0.5            # the t->t+1 join spans an episode reset -> not a real path
            if done.any():
                vmask[done, nb - 2] = 0.0; vmask[done, nb - 1] = 0.0
            gc = grad_caps_temporal_loss(traj, args.grad_caps_eps, vmask)
            actor_loss = actor_loss + args.lambda_temp * gc
            gc_val = float(gc.item())
        actor_opt.zero_grad(); actor_loss.backward(); actor_opt.step()
        with torch.no_grad():
            for p, pt in zip(critic.parameters(), critic_tgt.parameters()):
                pt.mul_(1 - args.tau).add_(args.tau * p)
        out = {"critic_loss": float(critic_loss.item()),
               "actor_loss": float(actor_loss.item()), "grad_caps": gc_val, "zb": zb.detach()}
        if q1p.shape[-1] > 1:            # per-head mean Q: which component steers the policy
            out["q_heads"] = (0.5 * (q1p + q2p)).mean(0).detach().cpu().numpy()
    return out


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
def save_buffer(buf, out_dir):
    """Dump the collected transitions (chronological, per env) to out_dir/buffer_<N>.npz
    for offline training (e.g. ship to a RunPod GPU). Reconstructs each env's ring into
    oldest->newest order via start=(head-count)%C; is_start marks episode boundaries so a
    consumer never builds a WM window across a reset."""
    px, prop, act, r, d, isx, lengths = [], [], [], [], [], [], []
    for e in range(buf.n_envs):
        n = int(buf.count[e])
        if n == 0:
            continue
        start = (int(buf.head[e]) - n) % buf.C
        idx = (start + np.arange(n)) % buf.C
        px.append(buf.pixels[e, idx]); prop.append(buf.proprio[e, idx])
        act.append(buf.action[e, idx]); r.append(buf.r[e, idx])
        d.append(buf.d[e, idx]); isx.append(buf.is_start[e, idx]); lengths.append(n)
    if not lengths:
        print("[frozen] buffer empty -> nothing to save", flush=True)
        return None
    path = out_dir / f"buffer_{sum(lengths):07d}.npz"
    np.savez_compressed(path,
                        pixels=np.concatenate(px), proprio=np.concatenate(prop),
                        action=np.concatenate(act), r=np.concatenate(r),
                        d=np.concatenate(d), is_start=np.concatenate(isx),
                        env_lengths=np.asarray(lengths, np.int64))
    print(f"[frozen] saved {sum(lengths)} transitions -> {path} "
          f"({path.stat().st_size / 1e6:.1f} MB)", flush=True)
    return path


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


def load_init_ckpt(args, wm, actor, critic, critic_tgt, device):
    """Warm-start / resume: load WM + actor-critic weights so the online loop CONTINUES a
    prior run instead of starting cold (train.py otherwise only ever SAVES checkpoints,
    never loads). Resolves a local --init-ckpt or an HF --resume-name[/--resume-step].
    Returns h_fwd to overwrite the freshly-initialised value. NOTE: optimizer state is
    not in the checkpoint, so Adam moments restart — lower LRs for bring-up if needed."""
    path = resolve_ckpt(args.init_ckpt, args.resume_name or args.name, args.resume_step, args.hf_repo)
    ck = torch.load(path, map_location=device, weights_only=False)
    wm.load_state_dict(ck["wm"])
    load_actor_state(actor, ck["actor"])     # drops the log_std head of old stochastic ckpts
    critic.load_state_dict(ck["critic"])
    critic_tgt.load_state_dict(ck["critic_tgt"])
    h_fwd = int(ck.get("h_fwd", args.h_fwd_start))
    print(f"[resume] loaded wm+actor+critic from {path} "
          f"(saved step {ck.get('step', '?')}, h_fwd={h_fwd})", flush=True)
    return h_fwd


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

    if args.env_backend == "hardware":
        if args.n_envs != 1:
            print(f"[hardware] forcing n_envs=1 (was {args.n_envs}; one physical arm)", flush=True)
            args.n_envs = 1
        from env.hardware_env import HardwareSO101Env
        VecEnv = HardwareSO101Env
    else:
        from env.parallel_env import VectorMujocoEnv, SubprocVectorMujocoEnv
        VecEnv = SubprocVectorMujocoEnv if args.env_backend == "subproc" else VectorMujocoEnv
    if args.start_steps < 0:                      # default-aware: skip random warmup on a real arm
        args.start_steps = 0 if args.env_backend == "hardware" else 1000
    env = VecEnv(n_envs=args.n_envs, frame_skip=args.frame_skip,
                 action_max=args.action_max,
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
    n_q_out = len(REWARD_COMPONENTS) if args.multihead_q else 1
    critic = TwinQ(z_dim, a_dim, n_out=n_q_out).to(device)
    critic_tgt = TwinQ(z_dim, a_dim, n_out=n_q_out).to(device)
    critic_tgt.load_state_dict(critic.state_dict())
    for p in critic_tgt.parameters():
        p.requires_grad_(False)

    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    wm_opt = torch.optim.AdamW([p for p in wm.parameters() if p.requires_grad],
                               lr=args.wm_lr, weight_decay=1e-3)

    # --- resume / warm-start: load wm+sac weights BEFORE the loop (else cold start) ---
    resume_h_fwd = None
    if args.init_ckpt or args.resume_name:
        resume_h_fwd = load_init_ckpt(args, wm, actor, critic, critic_tgt, device)

    # --- frozen-policy data collection: act + buffer, NO gradient updates ---
    policy_loaded = bool(args.init_ckpt or args.resume_name)
    if args.frozen_policy and not policy_loaded:
        msg = ("--frozen-policy with no --init-ckpt/--resume-name -> a RANDOM-init actor "
               "would drive the arm with no learning to correct it")
        if args.env_backend == "hardware":
            raise SystemExit(f"[frozen] REFUSING on hardware: {msg}. "
                             "Load a policy, e.g. --resume-name safe15 --resume-step 100000.")
        print(f"[frozen] WARNING: {msg}.", flush=True)
    if args.frozen_policy:
        print("[frozen] data-collection mode: NO gradient updates (WM/SAC/curriculum "
              "all skipped); acting + buffering only. Buffer -> out_dir/buffer_<N>.npz on exit.",
              flush=True)

    cap = int(np.clip(args.buffer_frac * args.total_steps, 1000, 50_000))
    cap_per_env = max(cap // args.n_envs, args.history_size + args.h_fwd_max + 8)
    if args.frozen_policy or args.save_buffer:   # collection KEEPS everything -> size the ring to the whole run
        keep = min(args.total_steps, 50_000)     # (the per-env ring otherwise overwrites the oldest in place)
        if args.total_steps > 50_000:
            print(f"[frozen] WARNING: requested {args.total_steps} steps but the buffer holds {keep}/env; "
                  f"oldest will be overwritten. Split into shorter runs to keep all transitions.", flush=True)
        cap_per_env = max(cap_per_env, keep)
    buf = ReplayBuffer(args.n_envs, cap_per_env, env.wrist_resolution, a_dim, prop_dim, device,
                       n_comp=len(REWARD_COMPONENTS) if args.multihead_q else 0)
    print(f"[buffer] {args.n_envs} x {cap_per_env} = {args.n_envs * cap_per_env} transitions", flush=True)

    run_name = args.name
    out_dir = Path(args.out_dir) if args.out_dir else Path("runs") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    n_params = sum(p.numel() for p in wm.parameters())
    print(f"[run] name={run_name}  out_dir={out_dir}  wm_params={n_params/1e6:.2f}M", flush=True)
    if args.env_backend == "hardware":
        loaded = bool(args.init_ckpt or args.resume_name)
        warns = []
        if args.action_max > 0.15:
            warns.append(f"action_max={args.action_max} is large for a real arm")
        if not loaded:
            warns.append("no policy loaded (random actor)")
        print(f"[hardware] SAFETY: n_envs=1, control_dt={env.dt_safe}s, "
              f"action_max={args.action_max} (<= +/-{args.action_max} rad/joint/step), "
              f"start_steps={args.start_steps}, policy={'loaded' if loaded else 'RANDOM'}. "
              f"Keep the e-stop within reach." + ("  [!] " + "; ".join(warns) if warns else ""),
              flush=True)

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
    if args.no_torque_obs:
        scrub_torque_obs(obs, n_dof)
    z = encode_obs(wm, obs["image"], obs["proprio"], device)        # (n_envs, z_dim)
    hist_z = z.unsqueeze(0).repeat(H, 1, 1)
    hist_a = torch.zeros(H, args.n_envs, a_dim, device=device)
    is_start = np.ones(args.n_envs, bool)
    ep_len = np.zeros(args.n_envs, np.int64)
    ep_ret = np.zeros(args.n_envs, np.float32)

    h_fwd = resume_h_fwd if resume_h_fwd is not None else args.h_fwd_start   # curriculum horizon (resumed if warm-started)
    pred_hist = deque(maxlen=args.flatline_window)    # for the flatline bump trigger
    updates_at_stage = 0
    prev_sub_a = np.zeros((args.n_envs, n_dof), np.float64)   # last sub-action of the previous block (action-rate boundary)
    tau_max_arr = np.asarray(env.tau_max, np.float32)
    prev_qpos_dec = None                                      # last decision's final joint pose (for pose_step travel)
    recent_qpos = deque(maxlen=200)                           # rolling final-pose history -> pose_spread / pose_range
    recent = {k: deque(maxlen=400) for k in
              ("r_cur", "r_safe", "cur_contrib", "contacts", "table_contacts",
               "motion", "ret", "frac_block", "frac_table",
               "rate", "rate2", "energy", "qd_mean", "tau_sat", "qd_rev",
               "r_rate", "r_energy", "pose_step")}
    recent_mse = {k: deque(maxlen=2000) for k in ("mse_block", "mse_table", "mse_none")}
    t0 = time.time()
    last_wm = last_sac = None
    last_zb = last_qh = None
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
            if args.no_torque_obs:               # probe obs must match the scrubbed training obs
                probe_prop = probe_prop.copy()
                probe_prop[..., 2 * n_dof:3 * n_dof] = 0.0
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
        nonlocal last_wm, last_sac, last_zb, last_qh, updates_at_stage
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
        res = sac_update(buf, wm, actor, critic, critic_tgt, actor_opt, critic_opt,
                         args, step, device)
        if res is not None:
            last_sac = (res["critic_loss"], res["actor_loss"], res.get("grad_caps", 0.0))
            last_zb = res["zb"]
            last_qh = res.get("q_heads")
        return h_fwd

    # graceful stop for data-collection runs: 1st Ctrl-C finishes the in-flight decision and
    # saves; a 2nd Ctrl-C (default handler restored) force-quits.
    _stop = {"flag": False}
    if args.frozen_policy or args.save_buffer:
        import signal
        def _on_sigint(signum, frame):
            _stop["flag"] = True
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            print("\n[frozen] stop requested -> finishing this decision then saving "
                  "(Ctrl-C again to force-quit).", flush=True)
        signal.signal(signal.SIGINT, _on_sigint)

    for step in range(args.total_steps):
        if _stop["flag"]:
            print(f"[frozen] graceful stop at step {step} ({buf.total} transitions).", flush=True)
            break
        cur_px, cur_prop = obs["image"], obs["proprio"]          # o_t (before acting)

        # --- act (deterministic policy; exploration = curiosity reward, not action noise.
        #     start_steps no longer randomizes acting — it only delays gradient updates,
        #     unless --warmup-random opts the uniform-action warmup back in: data-side
        #     diversity for from-scratch sim runs, still no policy stochasticity) ---
        with torch.no_grad():
            if args.warmup_random and step < args.start_steps:
                a = torch.rand(args.n_envs, a_dim, device=device) * 2 - 1
            else:
                a = actor(z)
                if args.explore_noise > 0:   # TD3-style collection noise; policy itself stays deterministic
                    a = (a + args.explore_noise * torch.randn_like(a)).clamp(-1.0, 1.0)
        hist_a = torch.cat([hist_a[1:], a.unsqueeze(0)], 0)
        a_env = a.detach().cpu().numpy().reshape(args.n_envs, args.action_block, n_dof)

        # --- async actor-learner: launch the env rollout for this decision, then run
        #     the GPU updates on buffered data WHILE the workers render -> overlap CPU/GPU ---
        env.step_block_async(a_env)
        if not args.frozen_policy:                # frozen: skip ALL gradient work (data collection only)
            h_fwd = learner_updates(step, h_fwd)
        obs, sub_infos = env.step_block_wait()
        if args.no_torque_obs:
            scrub_torque_obs(obs, n_dof)

        # --- accumulate safety reward + interaction + smoothness stats over the action_block ---
        r_safe = np.zeros(args.n_envs, np.float32)
        contacts = np.zeros(args.n_envs, np.float32)
        table_contacts = np.zeros(args.n_envs, np.float32)
        motion = np.zeros(args.n_envs, np.float32)
        energy = np.zeros(args.n_envs, np.float32)      # mean_i |tau_i * qd_i| (mechanical power)
        qd_mean = np.zeros(args.n_envs, np.float32)
        tau_sat = np.zeros(args.n_envs, np.float32)
        qd_rev = np.zeros(args.n_envs, np.float32)      # sign-flip fraction of qd between substeps
        prev_qd = None
        for info in sub_infos:
            r_safe += info["safety_reward"]
            contacts += info["object_contacts"].astype(np.float32)
            table_contacts += info["table_contacts"].astype(np.float32)
            motion += info["object_motion"]
            tau, qd = info["applied_torque"], info["qvel"]
            energy += np.abs(tau * qd).mean(-1)
            qd_mean += np.abs(qd).mean(-1)
            tau_sat += (np.abs(tau) > 0.95 * tau_max_arr).mean(-1).astype(np.float32)
            if prev_qd is not None:
                qd_rev += ((qd * prev_qd) < 0).mean(-1).astype(np.float32)
            prev_qd = qd
        r_safe /= args.action_block      # one r_safe per decision (README: Env(a_t) -> r_safe_t)
        energy /= args.action_block
        qd_mean /= args.action_block
        tau_sat /= args.action_block
        qd_rev /= max(args.action_block - 1, 1)

        # --- action-rate (legged_gym-style smoothness): mean per-dim squared delta over
        #     consecutive sub-actions, including the boundary pair with the previous
        #     block's last sub-action. On an episode's first decision the boundary (and
        #     the 2nd-order term spanning it) is masked out — never penalize the
        #     cross-reset jump. Computed for every run (logged); enters the reward only
        #     when --w-action-rate/--w-action-rate2 > 0. ---
        seq = np.concatenate([prev_sub_a[:, None, :], a_env.astype(np.float64)], axis=1)  # (n_envs, 1+B, n_dof)
        d1 = np.diff(seq, axis=1)                          # (n_envs, B, n_dof)
        d2 = np.diff(seq, n=2, axis=1)                     # (n_envs, B-1, n_dof)
        sq1, sq2 = (d1 ** 2).mean(-1), (d2 ** 2).mean(-1)  # per-pair, per-dim mean
        m1, m2 = np.ones_like(sq1), np.ones_like(sq2)
        m1[is_start, 0] = 0.0
        m2[is_start, 0] = 0.0
        rate = ((sq1 * m1).sum(1) / m1.sum(1)).astype(np.float32)
        rate2 = ((sq2 * m2).sum(1) / np.maximum(m2.sum(1), 1.0)).astype(np.float32)
        prev_sub_a = a_env[:, -1].astype(np.float64).copy()

        # --- ACTUAL joint travel (is the arm parked or going somewhere?). pose_step =
        #     how far the joint vector moved THIS decision; ~0 = frozen/dithering in place.
        #     pose_spread/pose_range (logged below) = how much of config space the recent
        #     window covers. These read q directly, so they catch in-place jitter that
        #     qd_mean misses (high qd + ~0 pose_step = oscillating, not exploring). ---
        qpos_dec = sub_infos[-1]["qpos"].astype(np.float64)        # (n_envs, n_dof) final pose of the block
        if prev_qpos_dec is None:
            prev_qpos_dec = qpos_dec
        pose_step = np.where(is_start, 0.0,
                             np.linalg.norm(qpos_dec - prev_qpos_dec, axis=-1)).astype(np.float32)
        prev_qpos_dec = qpos_dec
        recent_qpos.append(qpos_dec.copy())

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
        r_rate = -(args.w_action_rate * rate + args.w_action_rate2 * rate2)       # smoothness penalties (0 unless flagged)
        r_energy = -args.w_energy * energy
        safe_term = args.lambda_safe * r_safe
        reward = safe_term + cur_term + r_rate + r_energy
        comps = np.stack([cur_term, safe_term, r_rate, r_energy], -1).astype(np.float32)  # REWARD_COMPONENTS order

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
                r=reward, d=done, is_start=is_start.copy(), prio=r_cur,
                rc=comps if args.multihead_q else None)
        for key, val in (("r_cur", r_cur), ("r_safe", r_safe), ("cur_contrib", cur_term),
                         ("contacts", contacts), ("table_contacts", table_contacts),
                         ("motion", motion), ("ret", reward),
                         ("frac_block", touch_block), ("frac_table", touch_table),
                         ("rate", rate), ("rate2", rate2), ("energy", energy),
                         ("qd_mean", qd_mean), ("tau_sat", tau_sat), ("qd_rev", qd_rev),
                         ("r_rate", r_rate), ("r_energy", r_energy), ("pose_step", pose_step)):
            recent[key].append(float(np.mean(val)))

        # --- advance latent + history; reset timed-out envs ---
        z = z_next
        hist_z = torch.cat([hist_z[1:], z_next.unsqueeze(0)], 0)
        is_start = done > 0                                       # next o_t is a start where we reset
        done_envs = np.where(done > 0)[0]
        if len(done_envs):
            for e in done_envs:
                o_e = env.reset_one(int(e))
                if args.no_torque_obs:
                    scrub_torque_obs(o_e, n_dof)
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
                 "smooth/action_rate": np.mean(recent["rate"]),
                 "smooth/action_rate2": np.mean(recent["rate2"]),
                 "smooth/energy": np.mean(recent["energy"]),
                 "smooth/qd_mean": np.mean(recent["qd_mean"]),
                 "smooth/tau_sat_frac": np.mean(recent["tau_sat"]),
                 "smooth/qd_reversal_frac": np.mean(recent["qd_rev"]),
                 "reward/r_rate": np.mean(recent["r_rate"]),
                 "reward/r_energy": np.mean(recent["r_energy"]),
                 "explore/pose_step": np.mean(recent["pose_step"]),     # joint travel/decision; ~0 = parked
                 "buffer/transitions": buf.total, "perf/steps_per_sec": sps,
                 "wm/h_fwd": h_fwd}
            if recent_qpos:    # how much of joint space the recent window covers (parked -> ~0)
                qarr = np.stack(recent_qpos)                            # (T, n_envs, n_dof)
                d["explore/pose_spread"] = float(qarr.std(0).mean())    # mean temporal std over joints/envs
                d["explore/pose_range"] = float((qarr.max(0) - qarr.min(0)).mean())  # mean per-joint sweep
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
                          "smooth/grad_caps": last_sac[2]})
            if last_qh is not None:              # --multihead-q: per-component policy value
                d.update({f"sac/q_{k}": float(v) for k, v in zip(REWARD_COMPONENTS, last_qh)})
            wlog(d, step)
            with open(out_dir / "metrics.jsonl", "a") as f:    # local metrics record (esp. when --no-wandb)
                f.write(json.dumps({"step": step, **{k: float(v) for k, v in d.items()}}) + "\n")
            print(f"[step {step}] r_safe={safe_m:.3f} cur_contrib={cur_m:.3f} "
                  f"safe:cur={d['reward/safe_cur_ratio']:.2f} "
                  f"contacts/s={d['interact/contacts_per_step']:.2f} "
                  f"mse[blk/tbl/none]={d.get('wm/mse_block', float('nan')):.2f}/"
                  f"{d.get('wm/mse_table', float('nan')):.2f}/{d.get('wm/mse_none', float('nan')):.2f} "
                  f"rate={d['smooth/action_rate']:.2f} pose_step={d['explore/pose_step']:.3f} "
                  f"pose_range={d.get('explore/pose_range', float('nan')):.2f} "
                  f"h_fwd={h_fwd} sps={sps:.1f}", flush=True)

        # --- checkpoint: upload to HF then clear from disk (bounded disk over a long run) ---
        if args.save_every > 0 and step > 0 and step % args.save_every == 0:
            state = {"step": step, "wm": wm.state_dict(), "actor": actor.state_dict(),
                     "critic": critic.state_dict(), "critic_tgt": critic_tgt.state_dict(),
                     "h_fwd": h_fwd, "args": vars(args)}
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

    # --- final: collected data (frozen / --save-buffer) and/or model checkpoint ---
    if args.frozen_policy or args.save_buffer:
        save_buffer(buf, out_dir)
    if not args.frozen_policy:                    # frozen: weights are unchanged, skip the re-upload
        state = {"step": args.total_steps, "wm": wm.state_dict(), "actor": actor.state_dict(),
                 "critic": critic.state_dict(), "critic_tgt": critic_tgt.state_dict(),
                 "h_fwd": h_fwd, "args": vars(args)}
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
    p.add_argument("--start-steps", type=int, default=-1,
                   help="update warmup (decision steps): no WM/SAC gradient steps before this; acting is "
                        "ALWAYS the deterministic policy (the random-action warmup was removed with the "
                        "rest of the policy stochasticity). Default-aware: 0 on hardware, 1000 in sim")
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--env-threads", type=int, default=0,
                   help=">0 steps envs on a thread pool (inproc backend only)")
    p.add_argument("--env-backend", choices=("subproc", "inproc", "hardware"), default="subproc",
                   help="subproc: each env in a CUDA-free worker process, needed on "
                        "GPU+EGL to avoid the MuJoCo-render/CUDA SIGABRT; "
                        "inproc: envs in this process (sequential or --env-threads); "
                        "hardware: one physical SO-ARM101 via env/hardware_env.py (forces n_envs=1)")
    p.add_argument("--frame-skip", type=int, default=6)
    p.add_argument("--max-episode-steps", type=int, default=200, help="decision steps before truncation-as-done")
    p.add_argument("--seed", type=int, default=0)
    # resume / warm-start (train.py otherwise never loads a checkpoint)
    p.add_argument("--init-ckpt", default=None,
                   help="local .pt to warm-start wm+actor+critic+critic_tgt+h_fwd before the loop")
    p.add_argument("--resume-name", default=None,
                   help="resume from an HF run name (e.g. safe15) instead of a local --init-ckpt")
    p.add_argument("--resume-step", type=int, default=None,
                   help="checkpoint step for --resume-name (default: latest available)")
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
    p.add_argument("--action-max", type=float, default=0.3,
                   help="README dq^max: rad of joint delta per unit tanh action")
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
    p.add_argument("--lambda-safe", type=float, default=2.2,
                   help="weight on the safety penalty r_safe: r = lambda_safe*r_safe + lambda_cur*symlog(r_cur). "
                        "Default 2.2 (2026-06-12, real-arm calibration): under delta=9 benign motion scores "
                        "exactly 0, so lambda_safe scales only genuine events — 2.2 makes one median "
                        "user-labeled-bad substep cancel ~one decision's curiosity term. 0 ablates safety.")
    p.add_argument("--lambda-cur", type=float, default=15.0,
                   help="curiosity weight on symlog(r_cur). r_cur is the per-dim MEAN squared pred error "
                        "(~O(0.1-1)); 15-20 keeps curiosity audible against raw |r_safe|~50 at lambda_safe 0.1. "
                        "Default 15 (2026-06-11; safe15/campaign history ran 20 — old default 1.0 silently "
                        "shrank curiosity 20x and caused one mis-deploy).")
    p.add_argument("--safety-delta", type=float, default=9.0,
                   help="delta: safety-reward deadband on the per-joint -tau*qddot (N*m*rad/s^2). Default 9 "
                        "re-pinned 2026-06-12 from real-arm calibration at P8/D16 (true-dt args: all benign "
                        "motion incl. max-violence reversals <=7.4; user-labeled-bad grabs/blocks/jerks "
                        ">=10.7). SIM runs: use 15 with --lambda-safe 0.1 — measured 2026-06-12 "
                        "(runs/sim_scales/kp499.json): sim's saturated PD torque puts even smooth motion at "
                        "args>9 on 33%% of joint-samples, so the real-arm (9, 2.2) pair freezes sim policies. "
                        "The pre-2026-06 0.05 penalized all motion -> policy froze; the old 15 never fired on hw.")
    # smoothness / transferability experiment knobs (2026-06-12 sim campaign; all default-off)
    p.add_argument("--w-action-rate", type=float, default=0.0,
                   help="weight W on the action-rate penalty -W * mean_dim (a_t - a_{t-1})^2 over consecutive "
                        "sub-actions incl. the block boundary (legged_gym-style; actions already in [-1,1]). "
                        "Episode-start boundary masked. Sim scale ref (kp499.json): dither ~1.42, smooth ~0.09 "
                        "-> W=3 puts dither at -4.3 vs cur_contrib ~+10 while smooth motion pays ~-0.3.")
    p.add_argument("--w-action-rate2", type=float, default=0.0,
                   help="weight on the 2nd-order action-rate term mean_dim (a_t - 2a_{t-1} + a_{t-2})^2 "
                        "(omega^4 rolloff). Off by default; try after the 1st-order term proves out.")
    p.add_argument("--w-energy", type=float, default=0.0,
                   help="weight W on the energy penalty -W * mean_substeps mean_i |tau_i * qd_i| (mechanical "
                        "power; N*m*rad/s). Sim scale ref: dither ~3.3, smooth ~1.4 -> W=1 is a balanced trial. "
                        "Hardware analogue exists since kt=10 current-torque (2026-06-06).")
    p.add_argument("--no-torque-obs", action="store_true",
                   help="zero the u^app slice of proprio (obs -> [q, qd, 0]); shapes unchanged so old ckpts "
                        "load. Removes the obs channel that is ~96%% saturated sign-bit on hw and the main "
                        "sim->real proprio mismatch.")
    p.add_argument("--multihead-q", action="store_true",
                   help="critic outputs one Q head per reward component (cur/safe/rate/energy), each trained "
                        "on its own TD target; the actor maximizes the sum (same optimum as the scalar critic). "
                        "Logs sac/q_<comp> for interpretability.")
    p.add_argument("--actor-rate-reg", type=float, default=0.0,
                   help="action-rate as an ACTOR-LOSS regularizer (vs --w-action-rate's reward term): "
                        "+W * mean (pi(z) sub-action diffs)^2 added to actor_loss. Keeps smoothness "
                        "pressure out of r and Q — the A/B for whether the reward-term variant "
                        "pollutes the curiosity balance.")
    p.add_argument("--lambda-temp", type=float, default=0.0,
                   help="Grad-CAPS temporal-smoothness weight lambda_T on the ACTOR loss: "
                        "+lambda_T * mean_k ||s_{k-1}-2s_k+s_{k+1}|| * tanh(1/(||s_{k+1}-s_{k-1}||+eps)) "
                        "over the real applied sub-action path across the t->t+1 boundary "
                        "[pi(z_t)|pi(z_{t+1})]. Unlike --actor-rate-reg / --w-action-rate (squared "
                        "velocity -> penalizes ALL motion -> parks), this pays only low-travel "
                        "curvature (in-place zigzag), leaving smooth wide ramps free. Q/r untouched.")
    p.add_argument("--grad-caps-eps", type=float, default=1e-2,
                   help="epsilon in the Grad-CAPS 1/(displacement+eps) factor (caps the in-place "
                        "blow-up before tanh; smaller eps = sharper jitter penalty).")
    p.add_argument("--warmup-random", action="store_true",
                   help="act with uniform random actions during start_steps (restores the pre-2026-06-12 "
                        "warmup as an opt-in): buffer diversity for from-scratch sim runs; acting is "
                        "deterministic after warmup either way.")
    p.add_argument("--explore-noise", type=float, default=0.0,
                   help="TD3-style Gaussian noise std added to actions during COLLECTION only (clamped to "
                        "[-1,1]); the policy/eval remain deterministic. Sim-pretrain remedy for the "
                        "two-attractor pathology (bang-bang thrash vs frozen lull) seen in from-scratch "
                        "deterministic runs 2026-06-12; not for hardware.")
    # actor-critic (README; deterministic — entropy/alpha removed 2026-06-12)
    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument("--tau", type=float, default=0.005, help="Polyak rate (SAC-style target critic)")
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
    p.add_argument("--frozen-policy", action="store_true",
                   help="data-collection/eval: act with the loaded policy, NO gradient updates "
                        "(much higher cadence, learner removed). Refuses on hardware without a loaded policy.")
    p.add_argument("--save-buffer", action="store_true",
                   help="dump the replay buffer to out_dir/buffer_<N>.npz on exit (implied by "
                        "--frozen-policy); enables graceful Ctrl-C save.")
    p.add_argument("--hf-repo", default=None, help="HF repo id (else $HF_UPLOAD_REPO_ID)")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
