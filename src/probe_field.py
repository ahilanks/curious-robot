"""Spatial curiosity autopsy: render the curiosity/safety/value fields over the arm's
3D end-effector workspace from a checkpoint, so we can SEE *where* the curiosity signal
lives (and why the policy freezes) instead of inferring it from the windowed scalars.

First principles (see the diagnosis): exploration only emerges if every link in
    geometry(q) -> obs(o) -> latent(z) -> pred-error(r_cur) -> reward -> value(Q) -> action(pi)
preserves a usable, spatially-varying signal. A single pose sweep on a FIXED scene lets
us probe links 2,3,5 at once and overlay them on the same ee_pos axes:
  - link 2 (encoder): eff_rank / SV-spectrum / PCA(z) coloured by ee_pos, z->ee linear-probe R^2
  - link 3 (predictor): r_cur field + the persistence/mean bake-off on the SAME data
                        (matches the live reward exactly: per-dim MEAN error through wm.predict)
  - link 5 (policy/value): ||pi(z)|| field + Q(z, pi(z)) field + occupancy overlay

The sweep is faithful to the live reward: per pose we set qpos, take the policy's own
action over an action_block, read info["safety_reward"]/contacts/ee_pos, and re-encode the
real next obs for z_next (so r_cur is exactly what SAC would receive at that pose).

One-shot (snapshot a checkpoint):
    python src/probe_field.py --name safe15 --n 512            # latest safe15 ckpt from HF
    python src/probe_field.py --ckpt runs/lcur20/ckpt_0055000.pt --n 512

Watch (re-render the fields as a continual run drops new checkpoints -> a movie over training):
    python src/probe_field.py --watch --run-dir runs/safe15_cont --n 384 --wandb
    python src/probe_field.py --watch --name safe15_cont --n 384        # poll HF instead
"""
import os
import platform
# Render backend BEFORE importing mujoco / src.train (whose setdefault would otherwise pick
# osmesa on linux, which is not installed on the mac). glfw on darwin, egl on the GPU box.
os.environ.setdefault("MUJOCO_GL", "glfw" if platform.system() == "Darwin" else "egl")

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
from mpl_toolkits.mplot3d import Axes3D   # noqa: E402,F401  (registers 3d projection)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import mujoco   # noqa: E402
from env.mujoco_env import MujocoSO101Env, ARM_BASE_RADIUS, TABLE_TOP_Z   # noqa: E402
from model.state_encoder import WorldModel   # noqa: E402
from src.train import (Actor, TwinQ, REWARD_COMPONENTS, encode_obs, curiosity_reward,   # noqa: E402
                       load_actor_state, resolve_ckpt, collapse_metrics)

try:
    import wandb
except ImportError:
    wandb = None


def pick_device(override="auto"):
    if override and override != "auto":
        return torch.device(override)
    return torch.device("cuda" if torch.cuda.is_available()
                        else "mps" if torch.backends.mps.is_available() else "cpu")


def load_models(ckpt_path, device):
    """Reconstruct wm + actor + critic from a train.py checkpoint (same recipe as
    eval_predictor.py, plus the critic for the Q-field)."""
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = dict(ck["args"]);  n_dof = 6
    wm = WorldModel(n_dof=n_dof, action_block=a["action_block"],
                    history_size=a["history_size"]).to(device)
    wm.load_state_dict(ck["wm"]); wm.eval()
    a_dim = n_dof * a["action_block"]
    actor = Actor(wm.z_dim, a_dim).to(device)
    load_actor_state(actor, ck["actor"]); actor.eval()
    n_out = len(REWARD_COMPONENTS) if a.get("multihead_q") else 1
    critic = TwinQ(wm.z_dim, a_dim, n_out=n_out).to(device)
    try:
        critic.load_state_dict(ck["critic"]); critic.eval()
    except Exception as ex:
        print(f"[field] critic load failed ({ex}); Q-field disabled", flush=True)
        critic = None
    return wm, actor, critic, a, int(ck.get("step", 0))


@torch.no_grad()
def batch_encode(wm, px, prop, device, chunk=64):
    zs = [encode_obs(wm, px[s:s + chunk], prop[s:s + chunk], device)
          for s in range(0, len(px), chunk)]
    return torch.cat(zs, 0)


def linear_probe_r2(X, Y, frac=0.7, seed=0):
    """Per-column HELD-OUT R^2 of a least-squares linear decode X->Y (with bias). How legibly
    the latent encodes ee_pos: high test-R^2 = the latent genuinely resolves where the arm is.
    The split is essential: with a 256-d latent, an in-sample fit hits R^2=1 by overfitting
    whenever n_train <= dim, so only the held-out score is honest."""
    n = len(X)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    ntr = max(int(n * frac), 1)
    tr, te = idx[:ntr], idx[ntr:]
    if len(te) < 3:
        return np.full(Y.shape[1], np.nan, np.float64)
    Xa = np.concatenate([X, np.ones((n, 1), X.dtype)], 1)
    W, *_ = np.linalg.lstsq(Xa[tr], Y[tr], rcond=None)
    pred = Xa[te] @ W
    ss_res = ((Y[te] - pred) ** 2).sum(0)
    ss_tot = ((Y[te] - Y[te].mean(0)) ** 2).sum(0) + 1e-9
    return 1.0 - ss_res / ss_tot


@torch.no_grad()
def sweep(env, wm, actor, critic, ck_args, n, device, seed, action_mode="policy"):
    """Sweep n uniform arm poses on a FIXED scene; return the spatial fields. Two passes:
    pass 1 sets each pose statically (mj_forward) and renders the obs -> batch-encode z_t,
    pass 2 re-sets each pose and STEPS the chosen action over an action_block to get a real
    z_next + the live r_safe/contacts (so r_cur matches what SAC receives)."""
    n_dof = env.n_dof
    a_block = ck_args["action_block"]
    H = ck_args["history_size"]
    no_torque = bool(ck_args.get("no_torque_obs"))
    res = env.wrist_resolution
    rng = np.random.default_rng(seed)
    lo = env.model.jnt_range[:n_dof, 0].copy()
    hi = env.model.jnt_range[:n_dof, 1].copy()

    env.reset()                                  # fix ONE scene (objects) for the whole sweep
    obj_xpos = env._object_xpos()                # (n_objects, 3) world positions for the overlay
    qs = rng.uniform(lo, hi, size=(n, n_dof)).astype(np.float64)

    # --- pass 1: static encode (z_t, ee_pos) ---
    px_t = np.zeros((n, res, res, 3), np.uint8)
    prop_t = np.zeros((n, 3 * n_dof), np.float32)
    ee = np.zeros((n, 3), np.float32)
    for i in range(n):
        env.data.qpos[:n_dof] = qs[i]; env.data.qvel[:n_dof] = 0.0
        mujoco.mj_forward(env.model, env.data)
        ee[i] = env.data.xpos[env._ee_body_id]
        o = env._get_obs(render=True)
        px_t[i] = o["image"]; prop_t[i] = o["proprio"]
    if no_torque:
        prop_t[:, 2 * n_dof:3 * n_dof] = 0.0
    z_t = batch_encode(wm, px_t, prop_t, device)               # (n, D)

    # --- action at each pose: the policy's own deterministic mean (behaviourally relevant)
    #     or a fixed small reference action (intrinsic predictability of the pose) ---
    if action_mode == "policy":
        a = actor(z_t)
    else:
        a = torch.full((n, n_dof * a_block), 0.1, device=device)
    a_np = a.cpu().numpy().reshape(n, a_block, n_dof)
    a_mag = np.linalg.norm(a.cpu().numpy(), axis=-1).astype(np.float32)

    if critic is not None:
        q1, q2 = critic(z_t, a)
        Q = torch.min(q1.sum(-1), q2.sum(-1)).cpu().numpy().astype(np.float32)
    else:
        Q = np.full(n, np.nan, np.float32)

    # --- pass 2: step the action_block for z_next + live r_safe / contacts ---
    px_n = np.zeros_like(px_t); prop_n = np.zeros((n, 3 * n_dof), np.float32)
    r_safe = np.zeros(n, np.float32)
    n_obj_c = np.zeros(n, np.float32); n_tab_c = np.zeros(n, np.float32); motion = np.zeros(n, np.float32)
    for i in range(n):
        env.data.qpos[:n_dof] = qs[i]; env.data.qvel[:n_dof] = 0.0
        mujoco.mj_forward(env.model, env.data)
        env._prev_ctrl = qs[i].copy()                          # delta-target baseline = this pose
        env._prev_qvel = np.zeros(n_dof, np.float32)
        rs = 0.0; last = None
        for k in range(a_block):
            o, info = env.step(a_np[i, k], render=(k == a_block - 1))
            rs += info["safety_reward"]
            n_obj_c[i] += int(info["object_contacts"]); n_tab_c[i] += int(info["table_contacts"])
            motion[i] += float(info["object_motion"]); last = o
        r_safe[i] = rs / a_block
        px_n[i] = last["image"]; prop_n[i] = last["proprio"]
    if no_torque:
        prop_n[:, 2 * n_dof:3 * n_dof] = 0.0
    z_next = batch_encode(wm, px_n, prop_n, device)

    # --- r_cur EXACTLY as the live reward: per-dim MEAN error through wm.predict (incl pred_proj).
    #     hist_z = z_t tiled over H (arm sat at this pose), hist_a = zeros then the chosen action ---
    hist_z = z_t.unsqueeze(0).repeat(H, 1, 1)
    hist_a = torch.zeros(H, n, n_dof * a_block, device=device); hist_a[-1] = a
    r_cur = curiosity_reward(wm, hist_z, hist_a, z_next).cpu().numpy().astype(np.float32)

    # persistence & constant-mean bake-off baselines on the SAME data (matched .mean(-1))
    zt_np, zn_np = z_t.cpu().numpy(), z_next.cpu().numpy()
    persist = ((zt_np - zn_np) ** 2).mean(-1)                  # predict z_next == z_t
    mean_bl = ((zt_np.mean(0, keepdims=True) - zn_np) ** 2).mean(-1)

    bucket = np.where(n_obj_c > 0, 0, np.where(n_tab_c > 0, 1, 2)).astype(np.int64)  # 0 blk,1 tbl,2 none
    return dict(qs=qs.astype(np.float32), ee=ee, z=zt_np, r_cur=r_cur, r_safe=r_safe,
                persist=persist.astype(np.float32), mean_bl=mean_bl.astype(np.float32),
                a_mag=a_mag, Q=Q, bucket=bucket, obj_xpos=obj_xpos.astype(np.float32),
                n_cubes=int(env.n_cubes), contacts=n_obj_c, table_contacts=n_tab_c, motion=motion,
                action_mode=action_mode)


def summarize(f):
    """Headline diagnostics, computed from the sweep arrays."""
    r = f["r_cur"]
    bk = {0: "block", 1: "table", 2: "none"}
    by = {bk[b]: float(r[f["bucket"] == b].mean()) if (f["bucket"] == b).any() else float("nan")
          for b in (0, 1, 2)}
    z_std, eff_rank, feat_corr = collapse_metrics(torch.as_tensor(f["z"]))
    r2 = linear_probe_r2(f["z"], f["ee"])
    pred_mse = float(r.mean()); persist_mse = float(f["persist"].mean()); mean_mse = float(f["mean_bl"].mean())
    return {
        "r_cur_mean": pred_mse, "r_cur_std": float(r.std()),
        "r_cur_cv": float(r.std() / max(abs(pred_mse), 1e-9)),       # spatial variability of curiosity
        "mse_block": by["block"], "mse_table": by["table"], "mse_none": by["none"],
        "mse_contact_gap": by["none"] - by["block"],                 # >0 = blocks LESS novel (the inversion)
        "pred_mse": pred_mse, "persist_mse": persist_mse, "mean_mse": mean_mse,
        "pred_vs_persist": pred_mse / max(persist_mse, 1e-9),        # ~1.0 = persistence-collapse
        "pred_vs_mean": pred_mse / max(mean_mse, 1e-9),              # ~1.0 = mean-collapse
        "eff_rank": eff_rank, "z_std": z_std, "feat_corr": feat_corr,
        "z_to_ee_R2": [round(float(v), 3) for v in r2], "z_to_ee_R2_mean": float(np.mean(r2)),
        "a_mag_mean": float(np.nanmean(f["a_mag"])), "Q_mean": float(np.nanmean(f["Q"])),
        "frac_touch_block": float((f["bucket"] == 0).mean()),
        "frac_touch_table": float((f["bucket"] == 1).mean()),
        "n_poses": int(len(r)), "action_mode": f["action_mode"],
    }


# ------------------------------------------------------------------ plotting
def _scatter3d(ax, ee, c, title, cmap="viridis"):
    sc = ax.scatter(ee[:, 0], ee[:, 1], ee[:, 2], c=c, cmap=cmap, s=8, alpha=0.85)
    ax.set_title(title, fontsize=10); ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    return sc


def _overlay_objects(ax, f):
    o = f["obj_xpos"]
    ax.scatter(o[:f["n_cubes"], 0], o[:f["n_cubes"], 1], o[:f["n_cubes"], 2],
               marker="X", c="k", s=60, label="cubes")


def make_plots(f, summ, out, step, tag):
    out.mkdir(parents=True, exist_ok=True)
    saved = []

    # 1) twin 3D fields: curiosity vs safety on identical ee_pos axes (the literal ask)
    try:
        fig = plt.figure(figsize=(13, 5.5))
        ax1 = fig.add_subplot(121, projection="3d")
        sc1 = _scatter3d(ax1, f["ee"], f["r_cur"], "r_cur (curiosity) over ee_pos", "viridis")
        _overlay_objects(ax1, f); fig.colorbar(sc1, ax=ax1, shrink=0.6)
        ax2 = fig.add_subplot(122, projection="3d")
        sc2 = _scatter3d(ax2, f["ee"], f["r_safe"], "r_safe (safety) over ee_pos", "magma")
        _overlay_objects(ax2, f); fig.colorbar(sc2, ax=ax2, shrink=0.6)
        fig.suptitle(f"{tag}  step={step}  | curiosity CV={summ['r_cur_cv']:.3f}  "
                     f"contact_gap(none-block)={summ['mse_contact_gap']:+.3f}", fontsize=11)
        p = out / "field_3d.png"; fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig); saved.append(p)
    except Exception as ex:
        print(f"[field] field_3d plot failed: {ex}", flush=True)

    # 2) contact-bucketed r_cur histogram (the inversion, made visible)
    try:
        fig, ax = plt.subplots(figsize=(7, 4))
        for b, name, col in ((2, "none", "tab:blue"), (1, "table", "tab:orange"), (0, "block", "tab:green")):
            v = f["r_cur"][f["bucket"] == b]
            if len(v):
                ax.hist(v, bins=40, alpha=0.55, color=col, density=True,
                        label=f"{name} (n={len(v)}, mean={v.mean():.3f})")
        ax.set_xlabel("r_cur"); ax.set_ylabel("density")
        ax.set_title(f"curiosity by contact — gap(none-block)={summ['mse_contact_gap']:+.3f} "
                     f"({'INVERTED: blocks less novel' if summ['mse_contact_gap'] > 0 else 'blocks more novel'})")
        ax.legend(fontsize=8)
        p = out / "contact_hist.png"; fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig); saved.append(p)
    except Exception as ex:
        print(f"[field] contact_hist plot failed: {ex}", flush=True)

    # 3) encoder: SV spectrum + PCA(z) coloured by r_cur and by ee-as-RGB
    try:
        z = f["z"]; zc = z - z.mean(0)
        U, S, Vt = np.linalg.svd(zc, full_matrices=False)
        emb = zc @ Vt[:2].T
        ee = f["ee"]; rgb = (ee - ee.min(0)) / (ee.max(0) - ee.min(0) + 1e-9)
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
        axs[0].semilogy((S ** 2) / (S[0] ** 2 + 1e-12), ".-")
        axs[0].set_title(f"latent SV spectrum\neff_rank={summ['eff_rank']:.1f}  z_std={summ['z_std']:.3f}  "
                         f"feat_corr={summ['feat_corr']:.2f}")
        axs[0].set_xlabel("component"); axs[0].set_ylabel("eigenvalue (norm.)")
        s1 = axs[1].scatter(emb[:, 0], emb[:, 1], c=f["r_cur"], cmap="viridis", s=10)
        axs[1].set_title("PCA(z) coloured by r_cur"); fig.colorbar(s1, ax=axs[1], shrink=0.7)
        axs[2].scatter(emb[:, 0], emb[:, 1], c=rgb, s=10)
        axs[2].set_title(f"PCA(z) coloured by ee_pos (RGB)\nz->ee R2={summ['z_to_ee_R2_mean']:.2f}")
        for a in axs[1:]:
            a.set_xlabel("PC1"); a.set_ylabel("PC2")
        p = out / "latent.png"; fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig); saved.append(p)
    except Exception as ex:
        print(f"[field] latent plot failed: {ex}", flush=True)

    # 4) policy/value: ||pi(z)|| and Q(z, pi(z)) fields (the freeze basin)
    try:
        fig = plt.figure(figsize=(13, 5.5))
        ax1 = fig.add_subplot(121, projection="3d")
        sc1 = _scatter3d(ax1, f["ee"], f["a_mag"], f"||pi(z)|| (action mag, mean={summ['a_mag_mean']:.3f})", "plasma")
        fig.colorbar(sc1, ax=ax1, shrink=0.6)
        ax2 = fig.add_subplot(122, projection="3d")
        sc2 = _scatter3d(ax2, f["ee"], f["Q"], "Q(z, pi(z)) (critic value)", "cividis")
        fig.colorbar(sc2, ax=ax2, shrink=0.6)
        p = out / "policy_value.png"; fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig); saved.append(p)
    except Exception as ex:
        print(f"[field] policy_value plot failed: {ex}", flush=True)

    # 5) top-down (x,y) r_cur heatmaps in z-bands with object footprint overlay
    try:
        ee = f["ee"]; bands = [(ee[:, 2].min(), 0.10), (0.10, 0.25), (0.25, ee[:, 2].max() + 1e-3)]
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
        for ax, (z0, z1) in zip(axs, bands):
            m = (ee[:, 2] >= z0) & (ee[:, 2] < z1)
            if m.sum() < 4:
                ax.set_title(f"z∈[{z0:.2f},{z1:.2f}) — too few"); continue
            H_, xe, ye = np.histogram2d(ee[m, 0], ee[m, 1], bins=16, weights=f["r_cur"][m])
            C_, _, _ = np.histogram2d(ee[m, 0], ee[m, 1], bins=[xe, ye])
            with np.errstate(invalid="ignore"):
                im = ax.imshow((H_ / np.maximum(C_, 1)).T, origin="lower",
                               extent=[xe[0], xe[-1], ye[0], ye[-1]], aspect="auto", cmap="viridis")
            o = f["obj_xpos"]
            ax.scatter(o[:f["n_cubes"], 0], o[:f["n_cubes"], 1], marker="X", c="r", s=40)
            ax.add_patch(plt.Circle((0, 0), ARM_BASE_RADIUS, fill=False, color="w", ls="--"))
            ax.set_title(f"mean r_cur, z∈[{z0:.2f},{z1:.2f})"); ax.set_xlabel("x"); ax.set_ylabel("y")
            fig.colorbar(im, ax=ax, shrink=0.7)
        p = out / "topdown.png"; fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig); saved.append(p)
    except Exception as ex:
        print(f"[field] topdown plot failed: {ex}", flush=True)

    return saved


# ------------------------------------------------------------------ driver
def process(ckpt_path, env_holder, args, device, wb):
    wm, actor, critic, ck_args, step = load_models(ckpt_path, device)
    if env_holder.get("env") is None:        # build the env once (reuse renderer across ckpts)
        env_holder["env"] = MujocoSO101Env(action_max=ck_args["action_max"],
                                            safety_delta=ck_args["safety_delta"], seed=args.seed)
    t0 = time.time()
    f = sweep(env_holder["env"], wm, actor, critic, ck_args, args.n, device, args.seed, args.action_mode)
    summ = summarize(f)
    out = Path(args.out) / args.tag / f"step_{step:08d}"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "field.npz", **{k: v for k, v in f.items() if isinstance(v, np.ndarray)})
    (out / "summary.json").write_text(json.dumps({"step": step, **summ}, indent=2))
    saved = make_plots(f, summ, out, step, args.tag)
    print(f"[field] step={step} ({time.time()-t0:.1f}s)  CV={summ['r_cur_cv']:.3f}  "
          f"contact_gap={summ['mse_contact_gap']:+.3f}  eff_rank={summ['eff_rank']:.1f}  "
          f"pred/persist={summ['pred_vs_persist']:.3f}  pred/mean={summ['pred_vs_mean']:.3f}  "
          f"z->ee R2={summ['z_to_ee_R2_mean']:.2f}  ||a||={summ['a_mag_mean']:.3f}\n"
          f"        -> {out}", flush=True)
    if wb is not None:
        log = {f"field/{k}": v for k, v in summ.items() if isinstance(v, (int, float))}
        for p in saved:
            log[f"img/{p.stem}"] = wandb.Image(str(p))
        wb.log(log, step=step)
    return step


def discover_local(run_dir):
    out = {}
    for p in Path(run_dir).glob("ckpt_*.pt"):
        m = re.search(r"ckpt_(\d+)\.pt", p.name)
        if m:
            out[int(m.group(1))] = p
    return out


def discover_hf(name, repo):
    from huggingface_hub import HfApi
    fs = HfApi(token=os.environ.get("HF_TOKEN")).list_repo_files(repo)
    out = {}
    for fn in fs:
        m = re.match(rf"{re.escape(name)}/ckpt_(\d+)\.pt$", fn)
        if m:
            out[int(m.group(1))] = fn
    return out


def main():
    p = argparse.ArgumentParser(description="Spatial curiosity/safety/value field probe over the ee workspace.")
    p.add_argument("--name", default=None, help="HF run name (folder) to pull checkpoints from")
    p.add_argument("--step", type=int, default=None, help="checkpoint step (default: latest)")
    p.add_argument("--ckpt", default=None, help="explicit local .pt (overrides HF)")
    p.add_argument("--hf-repo", default=None, help="HF repo id (else $HF_UPLOAD_REPO_ID)")
    p.add_argument("--run-dir", default=None, help="local run dir to watch for ckpt_*.pt (needs --keep-local-ckpts on the trainer)")
    p.add_argument("--n", type=int, default=512, help="number of swept poses (smaller = faster)")
    p.add_argument("--seed", type=int, default=12345, help="FIXED across checkpoints -> identical scene+poses -> comparable fields over training")
    p.add_argument("--action-mode", choices=("policy", "fixed"), default="policy",
                   help="policy: r_cur under pi(z) (what the policy harvests); fixed: a small reference action (intrinsic pose predictability)")
    p.add_argument("--out", default="runs/field", help="output root (PNGs + npz + summary.json per step)")
    p.add_argument("--tag", default=None, help="subfolder/label (default: name or ckpt stem)")
    p.add_argument("--watch", action="store_true", help="poll for new checkpoints and render each as it appears")
    p.add_argument("--poll", type=int, default=120, help="watch poll interval (s)")
    p.add_argument("--wandb", action="store_true", help="log fields + scalars to a '<tag>_field' W&B run")
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto",
                   help="torch device. Use 'cpu' on the GPU box if EGL-render + CUDA in one process "
                        "SIGABRTs (the reason training uses --env-backend subproc); the ViT-tiny "
                        "encode is cheap and rendering dominates anyway.")
    args = p.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass
    repo = args.hf_repo or os.environ.get("HF_UPLOAD_REPO_ID")
    args.tag = args.tag or args.name or (Path(args.ckpt).parent.name if args.ckpt else "field")
    device = pick_device(args.device)
    print(f"[field] device={device} MUJOCO_GL={os.environ.get('MUJOCO_GL')} tag={args.tag}", flush=True)

    wb = None
    if args.wandb and wandb is not None and os.environ.get("WANDB_API_KEY"):
        wb = wandb.init(project=os.environ.get("WANDB_PROJECT", "curious-robot"),
                        entity=os.environ.get("WANDB_ENTITY"), name=f"{args.tag}_field",
                        group=args.tag, config=vars(args))
        print(f"[field] wandb {wb.url}", flush=True)

    env_holder = {"env": None}
    try:
        if not args.watch:
            ckpt = args.ckpt or resolve_ckpt(None, args.name, args.step, repo)
            process(ckpt, env_holder, args, device, wb)
            return
        # --- watch: render each NEW checkpoint as the continual run produces it ---
        print(f"[field] watching {'dir ' + args.run_dir if args.run_dir else 'HF ' + str(args.name)} "
              f"every {args.poll}s (Ctrl-C to stop)", flush=True)
        done = set()
        while True:
            try:
                avail = discover_local(args.run_dir) if args.run_dir else discover_hf(args.name, repo)
                for step in sorted(k for k in avail if k not in done):
                    ref = avail[step]
                    path = ref if args.run_dir else resolve_ckpt(None, args.name, step, repo)
                    process(str(path), env_holder, args, device, wb)
                    done.add(step)
            except Exception as ex:
                print(f"[field] poll error (will retry): {ex}", flush=True)
            time.sleep(args.poll)
    finally:
        if env_holder["env"] is not None:
            env_holder["env"].close()
        if wb is not None:
            wb.finish()


if __name__ == "__main__":
    main()
