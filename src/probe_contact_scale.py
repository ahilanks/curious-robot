"""Stage-2 contact-knowledge probe AT SCALE (the pre-registered acceptance test).

Extends probe_contact_wm.py (stage-1, n=15 contact windows from random-walk
collection) with the ChildPlow rake collector (smoke_child_sweep.py): scripted
radial rakes through the reachable annulus yield ~40-75% contact decisions per
episode, so contact-window n goes from ~15 to ~1k -- the scale the 07-10 stacking
plan pre-registered for the acceptance test ("contact pred/persist << 1 on >=1k
windows").

Key design: collection is CHECKPOINT-INDEPENDENT (scripted actions never consult
the WM), so all checkpoints are evaluated on the IDENTICAL windows -- the
treat-vs-ctrl comparison carries zero collection variance. Window classes:
  walk_free / walk_contact -- v2's OU-walk distribution (continuity + OOD-free base)
  rake_free / rake_contact -- the plow distribution; rake_contact is the headline
Uplift ratios use pred/persist (mean-of-preds over mean-of-persists) with
episode-level bootstrap CIs (windows within an episode are correlated).

Harness anchor (v2's step 0): the FIRST checkpoint with --buffer replays real
training-buffer windows through the exact predict call; pred/persist must land in
the training band (~0.1-0.5) or the harness is buggy and nothing else counts.
"""
import os, sys, json, argparse
os.environ.setdefault("MUJOCO_GL", "osmesa")
sys.path.insert(0, "/workspace/curious-robot")

import numpy as np
import torch
from types import SimpleNamespace
from env.mujoco_env import MujocoSO101Env
from model.state_encoder import WorldModel, pred_dims_from_args
from src.train import to_norm_pixel
from src.smoke_child_sweep import ChildPlow, teleport_front

ap = argparse.ArgumentParser()
ap.add_argument("--ckpts", required=True, help="comma-separated label=path pairs")
ap.add_argument("--buffer", default=None, help="state_latest.npz for the anchor (first ckpt)")
ap.add_argument("--eps-walk", type=int, default=12)
ap.add_argument("--eps-rake", type=int, default=28)
ap.add_argument("--dec", type=int, default=60)
ap.add_argument("--seed", type=int, default=11)
ap.add_argument("--out", default="runs/probe_contact_scale")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
device = torch.device("cuda")

ckpts = [kv.split("=", 1) for kv in args.ckpts.split(",")]
ck0 = torch.load(ckpts[0][1], map_location="cpu", weights_only=False)
a = SimpleNamespace(**ck0["args"])
Hb = a.history_size

# ---------------- 1) collect once (checkpoint-independent) ----------------
cache = os.path.join(args.out, "collection.npz")
if os.path.exists(cache):
    z = np.load(cache)
    px_all, act_all, con_all, kind_all = z["px"], z["act"], z["con"], z["kind"]
    print(f"[collect] reusing {cache}: {px_all.shape[0]} episodes", flush=True)
else:
    env = MujocoSO101Env(frame_skip=a.frame_skip, action_max=a.action_max, encode_cam=a.wm_cam,
                         safety_delta=a.safety_delta, seed=args.seed, fixed_objects=False)
    n_dof = env.n_dof
    rng = np.random.default_rng(0)
    px_eps, act_eps, con_eps, kinds = [], [], [], []
    for ep in range(args.eps_walk + args.eps_rake):
        kind = "walk" if ep < args.eps_walk else "rake"
        obs = env.reset(); teleport_front(env)
        obs = env._get_obs()
        plow = ChildPlow(env) if kind == "rake" else None
        walk = np.zeros(n_dof, np.float32)
        bias = np.zeros(n_dof, np.float32); bias[1] = -0.25
        px, acts, cons = [obs["image"]], [], []
        for dec in range(args.dec):
            subs, n_con = [], 0
            for k in range(a.action_block):
                if kind == "walk":
                    walk = np.clip(0.8 * walk + 0.2 * rng.standard_normal(n_dof).astype(np.float32), -1, 1)
                    sub = np.clip(walk + bias, -1, 1)
                else:
                    q_kf = plow.keyframe()
                    qpos = env.data.qpos[:n_dof].copy()
                    sub = np.clip((q_kf - qpos) / a.action_max, -1, 1).astype(np.float32)
                subs.append(sub)
                obs, info = env.step(sub, render=(k == a.action_block - 1))
                if plow is not None:
                    plow.observe(int(info["object_contacts"]))
                n_con += int(info["object_contacts"])
            px.append(obs["image"]); acts.append(np.concatenate(subs)); cons.append(n_con)
        px_eps.append(np.stack(px)); act_eps.append(np.stack(acts)); con_eps.append(np.array(cons)); kinds.append(kind)
        if ep % 8 == 0 or ep == args.eps_walk:
            print(f"[collect ep {ep:02d} {kind}] contact decisions {int((np.array(cons) > 0).sum())}/{args.dec}", flush=True)
    env.close()
    px_all, act_all = np.stack(px_eps), np.stack(act_eps)
    con_all, kind_all = np.stack(con_eps), np.array(kinds)
    np.savez_compressed(cache, px=px_all, act=act_all, con=con_all, kind=kind_all)
    print(f"[collect] saved {cache}", flush=True)
tot = int((con_all > 0).sum())
print(f"[collect] contact decisions total {tot}/{con_all.size} "
      f"(rake {int((con_all[kind_all == 'rake'] > 0).sum())}/{int((kind_all == 'rake').sum()) * args.dec})", flush=True)

# ---------------- 2) per-checkpoint eval on the identical windows ----------------
def build_wm(ck, n_dof):
    aa = SimpleNamespace(**ck["args"])
    wm = WorldModel(n_dof=n_dof, action_block=aa.action_block, history_size=aa.history_size,
                    dropout=aa.wm_dropout, use_proprio=not aa.no_proprio,
                    **pred_dims_from_args(aa)).to(device)
    wm.load_state_dict(ck["wm"]); wm.eval()
    [p.requires_grad_(False) for p in wm.parameters()]
    return wm

N_DOF = 6

@torch.no_grad()
def encode_px(wm, px_batch):
    prop = np.zeros((len(px_batch), 3 * N_DOF), np.float32)
    return wm.encode(to_norm_pixel(np.stack(px_batch), device),
                     torch.as_tensor(prop, device=device).float())

@torch.no_grad()
def pred_persist(wm, z_seq, act_seq):
    acts = torch.as_tensor(act_seq, device=device).float()
    out = []
    for s in range(z_seq.shape[0] - Hb - 1):
        pred = wm.predict(z_seq[s:s + Hb].unsqueeze(0),
                          wm.action_encoder(acts[s:s + Hb].unsqueeze(0)))[0, -1]
        tgt = z_seq[s + Hb]
        out.append((float((pred - tgt).pow(2).mean()),
                    float((z_seq[s + Hb - 1] - tgt).pow(2).mean())))
    return out

def ratio_ci(rows, n_boot=2000, seed=0):
    """rows: list of (ep_idx, pred, persist). Episode-bootstrap CI of mean(pred)/mean(persist)."""
    if not rows:
        return None
    eps_ids = sorted({r[0] for r in rows})
    by_ep = {e: [(p, q) for (ee, p, q) in rows if ee == e] for e in eps_ids}
    rng = np.random.default_rng(seed)
    point = np.mean([r[1] for r in rows]) / np.mean([r[2] for r in rows])
    boots = []
    for _ in range(n_boot):
        sel = rng.choice(eps_ids, size=len(eps_ids), replace=True)
        pp = [x for e in sel for x in by_ep[e]]
        boots.append(np.mean([x[0] for x in pp]) / np.mean([x[1] for x in pp]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"ratio": float(point), "lo": float(lo), "hi": float(hi), "n": len(rows), "n_eps": len(eps_ids)}

results = {}
for label, path in ckpts:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    wm = build_wm(ck, N_DOF)

    if args.buffer and label == ckpts[0][0]:
        z0 = np.load(args.buffer)
        anchor, rngb = [], np.random.default_rng(0)
        for e in range(min(4, z0["count"].shape[0])):
            n = int(z0["count"][e]); head = int(z0["head"][e]); C = z0["pixels"].shape[1]
            start = (head - n) % C
            for _ in range(3):
                s0 = int(rngb.integers(0, n - 32))
                idx = (start + s0 + np.arange(31)) % C
                if z0["is_start"][e][(start + s0 + np.arange(1, 31)) % C].any():
                    continue
                zc = encode_px(wm, list(z0["pixels"][e, idx]))
                anchor += pred_persist(wm, zc, z0["action"][e, idx[:-1]])
        pm = np.mean([x[0] for x in anchor]); pp = np.mean([x[1] for x in anchor])
        print(f"[anchor {label}] buffer windows n={len(anchor)}: pred/persist {pm / pp:.2f} "
              f"(band 0.1-0.5; >1 => harness bug)", flush=True)
        results["_anchor"] = {"ckpt": label, "ratio": float(pm / pp), "n": len(anchor)}
        del z0

    cls = {"walk_free": [], "walk_contact": [], "rake_free": [], "rake_contact": []}
    for e in range(px_all.shape[0]):
        zs = []
        for i in range(0, px_all.shape[1], 256):
            zs.append(encode_px(wm, list(px_all[e, i:i + 256])))
        z_seq = torch.cat(zs)
        pairs = pred_persist(wm, z_seq, act_all[e])
        for s, (pm_, pp_) in enumerate(pairs):
            key = f"{kind_all[e]}_{'contact' if con_all[e, s:s + Hb].sum() > 0 else 'free'}"
            cls[key].append((e, pm_, pp_))
    results[label] = {k: ratio_ci(v) for k, v in cls.items()}
    print(f"[{label}] " + "  ".join(
        f"{k} {v['ratio']:.2f} [{v['lo']:.2f},{v['hi']:.2f}] n={v['n']}"
        for k, v in results[label].items() if v), flush=True)
    del wm, ck
    torch.cuda.empty_cache()

with open(os.path.join(args.out, "results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"[done] {args.out}/results.json", flush=True)
