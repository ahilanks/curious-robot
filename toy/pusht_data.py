"""The 206 human Push-T demonstrations as a held-out test set.

Source: diffusion_policy's pusht_cchi_v7_replay.zarr (the data lerobot/pusht was converted from:
identical agent positions, actions and episode ends), which unlike the lerobot parquet carries the
T pose. 206 episodes, 25,650 frames at 10 Hz; actions are absolute PD targets (mouse teleop).
Downloaded once (31 MB) to toy/data/ and cached as npz.

The recorded state has no agent velocity; the recorder starts from rest, so the velocity follows
exactly from the kinematic agent's PD map (push_t.pd_matrices): one-step agent error with the
rebuilt velocity is ~1e-5 px vs ~2 px from the 5-D state alone.

    python toy/pusht_data.py        # download + convert + replay check in the simulator
"""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from push_t import Sim, keypoints, pd_matrices

URL = "https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip"
DATA = Path(__file__).resolve().parent / "data"
NPZ = DATA / "pusht_demos.npz"


def _convert():
    import zarr
    DATA.mkdir(parents=True, exist_ok=True)
    zpath = DATA / "pusht" / "pusht_cchi_v7_replay.zarr"
    if not zpath.exists():
        print(f"[pusht_data] downloading {URL}", flush=True)
        with urllib.request.urlopen(URL, timeout=120) as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extractall(DATA)
    z = zarr.open(str(zpath), mode="r")
    np.savez(NPZ, state=z["data/state"][:].astype(np.float64), action=z["data/action"][:].astype(np.float64),
             episode_ends=z["meta/episode_ends"][:].astype(np.int64))


def load_demos():
    """List of episodes: s7 (L, 7) = agent xy, agent v (rebuilt), T xy, T angle; target (L, 2)."""
    if not NPZ.exists():
        _convert()
    d = np.load(NPZ)
    S5, A, ends = d["state"], d["action"], d["episode_ends"]
    M, N = pd_matrices()
    eps = []
    for b, e in zip(np.r_[0, ends[:-1]], ends):
        s5, tg = S5[b:e], A[b:e]
        v = np.zeros((len(s5), 2))
        for t in range(len(s5) - 1):
            v[t + 1] = M[1, 0] * s5[t, :2] + M[1, 1] * v[t] + N[1] * tg[t]
        eps.append({"s7": np.concatenate([s5[:, :2], v, s5[:, 2:5]], -1), "target": tg})
    return eps


def replay_check(eps, horizons=(1, 10, 50, 100)):
    """Open-loop replay of every episode in the simulator from its first recorded state."""
    sim, err = Sim(), {h: [] for h in horizons}
    for ep in eps:
        s7, tg = ep["s7"], ep["target"]
        sim.set(s7[0])
        for t in range(1, min(len(s7), max(horizons) + 1)):
            sim.step(tg[t - 1])
            if t in err:
                d = np.linalg.norm(keypoints(sim.state()[4:7]) - keypoints(s7[t, 4:7]), axis=-1).mean()
                err[t].append(d)
    for h, v in err.items():
        v = np.array(v)
        print(f"  replay h={h:3d}: n={len(v):3d}  T-keypoint error median {np.median(v):.2e} px, "
              f"p95 {np.percentile(v, 95):.2f} px, max {v.max():.2f} px")
    return err


if __name__ == "__main__":
    eps = load_demos()
    L = np.array([len(e["s7"]) for e in eps])
    print(f"[pusht_data] {len(eps)} episodes, {L.sum()} frames, length min/median/max {L.min()}/{int(np.median(L))}/{L.max()}")
    replay_check(eps)
