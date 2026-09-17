"""Critic-quality diagnostic for the RP1 planner experiment (2026-09-17).

Does the learned quasimetric cost-to-go V(z, z*) (--plan-cost value, fitted on the replay
buffer's own latents) track TRUE steps-to-goal along the buffer's trajectories better than
latent L2 (the campaign's planning cost)? Reads a run checkpoint (WM + the planner critic's
EMA teacher) and a state snapshot, rebuilds the BufferLatentCache exactly as the run did, and
reports, on in-episode pairs (z_t, z_{t+d}):

  * V and ||z_t - z_{t+d}|| vs the true offset d (mean +- std per d)
  * Spearman rank-correlation of V and of L2 with d (random in-episode pairs, d in [1, dmax])
  * cross-episode (unrelated-state) pairs: the scale V / L2 assign to "not reachable soon"
  * the quadrant that decides the experiment: pairs CLOSE in L2 but FAR in time -- does V
    call them far? and pairs FAR in L2 but CLOSE in time -- does V call them near?

  python src/diag_vcritic.py --ckpt runs/rp1x_val/ckpt_0001000.pt --state runs/wr_sleepret2/state_latest.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from model.state_encoder import WorldModel, pred_dims_from_args      # noqa: E402
from src.train import ReplayBuffer, load_state_snapshot, encode_obs   # noqa: E402
from src.rp1_planner import BufferLatentCache                          # noqa: E402
from rp1.critic import make_critic                                     # noqa: E402


def spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--dmax", type=int, default=60)
    ap.add_argument("--pairs", type=int, default=20000)
    ap.add_argument("--n-dof", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(a.seed)

    ck = torch.load(a.ckpt, map_location=device, weights_only=False)
    args = argparse.Namespace(**ck["args"])
    wm = WorldModel(n_dof=a.n_dof, action_block=args.action_block, history_size=args.history_size,
                    dropout=args.wm_dropout, use_proprio=not args.no_proprio, **pred_dims_from_args(args)).to(device)
    wm.load_state_dict(ck["wm"]); wm.eval()
    if "planner" not in ck:
        raise SystemExit("checkpoint has no planner state (run with --plan-cost value / --planner rp1)")
    D = wm.z_dim
    critic = make_critic(D, 256, 128, 2).to(device)
    critic.load_state_dict(ck["planner"].get("critic_teacher", ck["planner"]["critic"])); critic.eval()

    snap = np.load(a.state)
    n_envs, cap = int(snap["head"].shape[0]), int(snap["count"].max())
    img_hw, prop_dim, a_dim = snap["pixels"].shape[2], snap["proprio"].shape[-1], snap["action"].shape[-1]
    buf = ReplayBuffer(n_envs, cap, img_hw, a_dim, prop_dim, device, goal_explore=True)
    load_state_snapshot(buf, a.state)

    def encode_rows(px, pr):
        with torch.no_grad():
            return encode_obs(wm, px, pr, device).float().cpu().numpy()
    cache = BufferLatentCache(buf, encode_rows, device, args.history_size)
    z, ep_len, ep_off = cache.z, cache.ep_len, cache.ep_off
    print(f"cache: {cache.rows} rows, {cache.n_ep} episodes (len min/med/max {ep_len.min()}/{int(np.median(ep_len))}/{ep_len.max()})")

    @torch.no_grad()
    def V(i, j):
        return critic(z[torch.as_tensor(i, device=device)], z[torch.as_tensor(j, device=device)]).cpu().numpy()

    @torch.no_grad()
    def L2(i, j):
        return (z[torch.as_tensor(i, device=device)] - z[torch.as_tensor(j, device=device)]).norm(dim=-1).cpu().numpy()

    # --- 1. V and L2 vs true offset d --------------------------------------------------------------
    print("\n[1] in-episode pairs (z_t, z_{t+d}):   d    V mean+-sd     L2 mean+-sd")
    rows_d = {}
    for d in [1, 2, 3, 5, 8, 12, 16, 20, 30, 40, 60]:
        if d > a.dmax:
            break
        ok = np.flatnonzero(ep_len > d + 1)
        e = rng.choice(ok, 2000)
        t = (rng.random(2000) * (ep_len[e] - 1 - d)).astype(np.int64)
        i, j = ep_off[e] + t, ep_off[e] + t + d
        v, l = V(i, j), L2(i, j)
        rows_d[d] = (v.mean(), v.std(), l.mean(), l.std())
        print(f"   {d:3d}   {v.mean():6.2f} +- {v.std():5.2f}   {l.mean():6.2f} +- {l.std():5.2f}")

    # --- 2. rank correlation with d over random pairs ------------------------------------------------
    ok = np.flatnonzero(ep_len > 3)
    e = rng.choice(ok, a.pairs)
    dmax_e = np.minimum(a.dmax, ep_len[e] - 2)
    d = (rng.random(a.pairs) * dmax_e).astype(np.int64) + 1
    t = (rng.random(a.pairs) * (ep_len[e] - 1 - d)).astype(np.int64)
    i, j = ep_off[e] + t, ep_off[e] + t + d
    v, l = V(i, j), L2(i, j)
    print(f"\n[2] Spearman(., true d) over {a.pairs} in-episode pairs, d in [1, {a.dmax}]:  V {spearman(v, d):.3f}   L2 {spearman(l, d):.3f}"
          f"   (V vs L2: {spearman(v, l):.3f})")

    # --- 3. cross-episode pairs ----------------------------------------------------------------------
    e1, e2 = rng.choice(cache.n_ep, a.pairs), rng.choice(cache.n_ep, a.pairs)
    i = ep_off[e1] + (rng.random(a.pairs) * ep_len[e1]).astype(np.int64)
    j = ep_off[e2] + (rng.random(a.pairs) * ep_len[e2]).astype(np.int64)
    vx, lx = V(i, j), L2(i, j)
    print(f"[3] cross-episode pairs: V {vx.mean():.2f} +- {vx.std():.2f}   L2 {lx.mean():.2f} +- {lx.std():.2f}"
          f"   (random-pair diameter sqrt(2D) = {np.sqrt(2 * D):.1f})")

    # --- 4. the deciding quadrants -----------------------------------------------------------------
    lq = np.quantile(l, [0.25, 0.75]); dq = np.quantile(d, [0.25, 0.75])
    near_l_far_d = (l <= lq[0]) & (d >= dq[1])
    far_l_near_d = (l >= lq[1]) & (d <= dq[0])
    near_l_near_d = (l <= lq[0]) & (d <= dq[0])
    far_l_far_d = (l >= lq[1]) & (d >= dq[1])
    print(f"\n[4] quadrants (L2 quartiles {lq[0]:.1f}/{lq[1]:.1f}, d quartiles {dq[0]:.0f}/{dq[1]:.0f} decisions):")
    for name, m in [("L2-near & time-near", near_l_near_d), ("L2-near & time-FAR ", near_l_far_d),
                    ("L2-FAR  & time-near", far_l_near_d), ("L2-FAR  & time-FAR ", far_l_far_d)]:
        if m.any():
            print(f"   {name}: n={int(m.sum()):5d}  V {v[m].mean():6.2f}  L2 {l[m].mean():6.2f}  true d {d[m].mean():5.1f}")
    print("   -> a useful critic separates the two 'L2-near' rows (V small vs V large) and the two 'time-near' rows.")


if __name__ == "__main__":
    main()
