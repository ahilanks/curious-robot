"""Stage-1 probe v2. Fixes over v1:
  0) HARNESS SANITY ANCHOR: replay real training-buffer windows through this exact
     predict call -- must reproduce training's pred/persist (~0.1-0.5) else the
     harness is buggy and nothing else counts.
  1) Contact generation that works: teleport block 0 to the arm's deterministic
     front spot (0.30, 0.0) after every reset + mild downward sweep -- v1's hard
     bias never reached the block zone (0 contacts in 4200 decisions).
  2) Contact labels from `object_contacts` ONLY (v1's object-motion threshold
     caught settle-jitter/respawn artifacts).
  3) Decode over many short episodes (layout diversity), base-yaw dim reported
     separately (wrist cam can't see it -- v1 discovery).
"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "osmesa")
sys.path.insert(0, "/workspace/curious-robot")

import numpy as np
import torch
from types import SimpleNamespace
from env.mujoco_env import MujocoSO101Env
from model.state_encoder import WorldModel, pred_dims_from_args
from src.train import to_norm_pixel

CKPT = os.environ.get("PROBE_CKPT",
    "/root/.cache/huggingface/hub/models--a5ilank--curious-robot/snapshots/c98490b52e309c802faacc77a2177ccf5ba3a822/arr95_hot5b/ckpt_0059000.pt")
device = torch.device("cuda")
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
a = SimpleNamespace(**ck["args"])
wm = None  # built after env for n_dof

# ---------- 0) harness anchor on real buffer windows ----------
z0 = np.load(os.environ.get("PROBE_BUFFER", "/workspace/curious-robot/runs/arr95_hot5b/state_latest.npz"))
n_dof_buf = 6
Hb = a.history_size
env0 = MujocoSO101Env(frame_skip=a.frame_skip, action_max=a.action_max, encode_cam=a.wm_cam,
                      safety_delta=a.safety_delta, seed=11, fixed_objects=False)
n_dof = env0.n_dof
wm = WorldModel(n_dof=n_dof, action_block=a.action_block, history_size=a.history_size,
                dropout=a.wm_dropout, use_proprio=not a.no_proprio,
                **pred_dims_from_args(a)).to(device)
wm.load_state_dict(ck["wm"]); wm.eval()
[p.requires_grad_(False) for p in wm.parameters()]

@torch.no_grad()
def encode_px(px_batch):
    prop = np.zeros((len(px_batch), 3 * n_dof), np.float32)
    return wm.encode(to_norm_pixel(np.stack(px_batch), device),
                     torch.as_tensor(prop, device=device).float())

@torch.no_grad()
def pred_persist(z_seq, act_seq):
    """z_seq (T+1,D) torch, act_seq (T,a_dim) np -> lists of (pred_mse, persist_mse)."""
    acts = torch.as_tensor(act_seq, device=device).float()
    out = []
    for s in range(z_seq.shape[0] - Hb - 1):
        pred = wm.predict(z_seq[s:s+Hb].unsqueeze(0),
                          wm.action_encoder(acts[s:s+Hb].unsqueeze(0)))[0, -1]
        tgt = z_seq[s+Hb]
        out.append((float((pred - tgt).pow(2).mean()),
                    float((z_seq[s+Hb-1] - tgt).pow(2).mean())))
    return out

print("=== 0) harness anchor: real buffer windows ===", flush=True)
anchor = []
rng = np.random.default_rng(0)
for e in range(4):
    n = int(z0["count"][e]); head = int(z0["head"][e]); C = z0["pixels"].shape[1]
    start = (head - n) % C
    # take 3 random contiguous chunks of 30 transitions, avoiding episode starts
    for _ in range(3):
        s0 = int(rng.integers(0, n - 32))
        idx = (start + s0 + np.arange(31)) % C
        if z0["is_start"][e][(start + s0 + np.arange(1, 31)) % C].any():
            continue
        zc = encode_px(list(z0["pixels"][e, idx]))
        anchor += pred_persist(zc, z0["action"][e, idx[:-1]])
pm = np.mean([x[0] for x in anchor]); pp = np.mean([x[1] for x in anchor])
print(f"buffer windows n={len(anchor)}: pred {pm:.4f} persist {pp:.4f} pred/persist {pm/pp:.2f} "
      f"(training logs ~0.1-0.5; >1 => harness bug)", flush=True)

# ---------- 1+2) contact collection: block teleported into the sweep path ----------
def teleport_front(env):
    addr = env._object_qpos_addrs[0]
    vadr = env._object_qvel_addrs[0]
    z_rest = env._object_resting_z(0)
    env.data.qpos[addr:addr+2] = [0.30, 0.0]
    env.data.qpos[addr+2] = z_rest
    env.data.qvel[vadr:vadr+6] = 0.0
    import mujoco; mujoco.mj_forward(env.model, env.data)

EPS, DEC = 40, 60
eps = []
env = env0
for ep in range(EPS):
    obs = env.reset(); teleport_front(env)
    obs = env._get_obs()
    walk = np.zeros(n_dof, np.float32)
    bias = np.zeros(n_dof, np.float32); bias[1] = -0.25
    rec = {"px": [obs["image"]], "qpos": [obs["proprio"][:n_dof].copy()], "act": [],
           "contact": [], "obj_xy": [env._object_xpos()[:, :2].copy()]}
    for t in range(DEC):
        subs, n_con = [], 0
        for _ in range(a.action_block):
            walk = np.clip(0.8 * walk + 0.2 * rng.standard_normal(n_dof).astype(np.float32), -1, 1)
            sub = np.clip(walk + bias, -1, 1)
            subs.append(sub)
            obs, info = env.step(sub)
            n_con += int(info["object_contacts"])
        rec["px"].append(obs["image"]); rec["qpos"].append(obs["proprio"][:n_dof].copy())
        rec["act"].append(np.concatenate(subs)); rec["contact"].append(n_con)
        rec["obj_xy"].append(env._object_xpos()[:, :2].copy())
    eps.append(rec)
    if ep % 8 == 0:
        print(f"[ep {ep:02d}] contact decisions {sum(np.array(rec['contact'])>0)}/{DEC}", flush=True)
env.close()
tot_c = sum(sum(np.array(r["contact"]) > 0) for r in eps)
print(f"contact decisions total: {tot_c} / {EPS*DEC}", flush=True)

for rec in eps:
    zs = []
    for i in range(0, len(rec["px"]), 256):
        zs.append(encode_px(rec["px"][i:i+256]))
    rec["z"] = torch.cat(zs)

cls = {"free": [], "contact": []}
for rec in eps:
    pairs = pred_persist(rec["z"], np.stack(rec["act"]))
    for s, (pm_, pp_) in enumerate(pairs):
        key = "contact" if sum(rec["contact"][s:s+Hb]) > 0 else "free"
        cls[key].append((pm_, pp_))
print("\n=== A) predictor error by window class (same collection distribution) ===")
for k, v in cls.items():
    if v:
        pm_ = np.mean([x[0] for x in v]); pp_ = np.mean([x[1] for x in v])
        print(f"{k:8s} n={len(v):5d} pred {pm_:.4f} persist {pp_:.4f} pred/persist {pm_/pp_:.2f}")

# ---------- 3) decode with layout diversity ----------
Z = torch.cat([r["z"][:-1] for r in eps]).cpu().numpy()
nb = eps[0]["obj_xy"][0].shape[0]
BXY = np.concatenate([np.stack(r["obj_xy"][:-1]).reshape(-1, nb * 2) for r in eps])
Q = np.concatenate([np.stack(r["qpos"][:-1]) for r in eps])
ep_id = np.concatenate([np.full(len(r["z"]) - 1, i) for i, r in enumerate(eps)])
test = ep_id >= EPS - 8
def ridge_r2(X, Y, lam=1e-1):
    Xm, Ym = X[~test].mean(0), Y[~test].mean(0)
    W = np.linalg.solve((X[~test]-Xm).T @ (X[~test]-Xm) + lam*np.eye(X.shape[1]),
                        (X[~test]-Xm).T @ (Y[~test]-Ym))
    P = (X[test]-Xm) @ W + Ym
    return 1 - ((P-Y[test])**2).sum(0) / (((Y[test]-Y[test].mean(0))**2).sum(0)+1e-12)
r2b, r2q = ridge_r2(Z, BXY), ridge_r2(Z, Q)
print("\n=== B) linear decode (8 held-out episodes/layouts) ===")
print(f"block0 (teleported, always in view) xy R2: {np.round(r2b[:2],3)}")
print(f"other blocks xy R2 mean: {r2b[2:].mean():.3f}")
print(f"arm qpos R2 (dim0=base yaw): {np.round(r2q,3)}")
