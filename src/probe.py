"""Fixed, diverse probe set for encoder-collapse diagnostics.

Uniform-random arm poses across the FULL joint limits + freshly randomized object
layouts, rendered once and cached on the HF Hub. Computing encoder/eff_rank_probe on
this canonical set means it's (a) identical across every run -> directly comparable
run-to-run, and (b) decoupled from the policy's (possibly narrowing) behavior. Diversity
is GUARANTEED by construction (uniform over joint limits), not hoped for; build_probe_set
also returns an INPUT-space coverage report (measuring it via the encoder would be
circular -- that's the thing under test). CPU/OSMesa render only; no torch/GPU.
"""
from __future__ import annotations

import json
import os

import numpy as np

PROBE_DIR_IN_REPO = "probe"


def build_probe_set(n: int = 256, wrist_resolution: int = 224, seed: int = 12345):
    """Construct `n` probe observations: each a uniform-random arm pose (within the
    joint limits) over a fresh randomized object layout, rendered. Returns
    (pixels uint8 (n,H,W,3), proprio float32 (n,3*n_dof), report dict)."""
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    import mujoco
    from env.mujoco_env import MujocoSO101Env

    env = MujocoSO101Env(wrist_resolution=wrist_resolution, seed=seed)
    nd = env.n_dof
    lo = env.model.jnt_range[:nd, 0].copy()
    hi = env.model.jnt_range[:nd, 1].copy()
    rng = np.random.default_rng(seed)
    px, prop = [], []
    for _ in range(n):
        env.reset()                                   # fresh randomized object layout
        env.data.qpos[:nd] = rng.uniform(lo, hi)      # uniform arm pose across joint limits
        env.data.qvel[:nd] = 0.0
        mujoco.mj_forward(env.model, env.data)
        obs = env._get_obs()
        px.append(obs["image"].copy())
        prop.append(obs["proprio"].copy())
    env.close()

    px = np.stack(px)
    prop = np.stack(prop).astype(np.float32)
    qpos = prop[:, :nd]                                # the joint angles we sampled
    cover = (qpos.max(0) - qpos.min(0)) / (hi - lo + 1e-9)   # fraction of each joint's range spanned
    report = {
        "n": int(n), "wrist_resolution": int(wrist_resolution), "seed": int(seed),
        "mean_joint_coverage": round(float(cover.mean()), 3),
        "per_joint_coverage": [round(float(c), 3) for c in cover],
        "note": "coverage is over qpos (the sampled joints); qvel=0 and visual diversity comes from the pose render",
    }
    return px, prop, report


def save_probe_local(path, px, prop, report) -> None:
    np.savez_compressed(path, pixels=px, proprio=prop, report=np.array(json.dumps(report)))


def upload_probe_hf(path, repo_id: str, probe_id: str, token: str | None = None) -> None:
    from huggingface_hub import HfApi
    api = HfApi(token=token or os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    api.upload_file(path_or_fileobj=str(path), repo_id=repo_id,
                    path_in_repo=f"{PROBE_DIR_IN_REPO}/{probe_id}.npz")


def load_probe_hf(repo_id: str, probe_id: str, token: str | None = None):
    """Download + load the cached probe. Returns (pixels, proprio) or None if unavailable."""
    if not repo_id:
        return None
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(repo_id=repo_id, filename=f"{PROBE_DIR_IN_REPO}/{probe_id}.npz",
                            repo_type="model", token=token or os.environ.get("HF_TOKEN"))
        d = np.load(p)
        return d["pixels"], d["proprio"].astype(np.float32)
    except Exception as ex:
        print(f"[probe] HF load failed ({ex})", flush=True)
        return None
