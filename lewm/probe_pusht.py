"""Representation read-outs for Push-T LeWM checkpoints (offline replication, released, from-scratch arms).

For each checkpoint (frozen, eval mode, its own BatchNorm running statistics) on two frame sets:
  expert   held-out episodes of pusht_expert_train (probe fit on episodes [0, 2000), scored on [17000, 18685))
  uniform  stock-env renders of uniformly random states (agent [22, 489]^2, T centre [80, 432]^2, angle U[0, 2pi)),
           fit / score split 50/50 - a probe set no model was trained toward
probes from z (192-d) to  agent xy | T xy | T angle (as cos, sin):
  linear  ridge regression on standardised z (closed form, lambda 1e-3 * trace / dim)
  mlp     192 -> 512 -> 512 -> 6, Adam 1e-3, 3000 steps, batch 1024
reported per quantity: R^2, Pearson r, and the error in physical units (px for positions, deg for the angle).
Plus open-loop prediction on held-out expert windows (3 context frames + k predicted blocks, expert actions in each
model's own action units): the predicted latent decoded by the linear expert probe, error in px / deg at k = 1..5,
next to the same probe applied to the true latents (probe floor) and to "copy the last context latent".

    .venv/bin/python probe_pusht.py --ckpt released/lewm_pusht --ckpt pusht_rep_s3072/weights_epoch_1.pt \
        --ckpt pusht_fs_pred_s0/weights_step_13933.pt:uniform --out runs/probe_ep1.json
  (':uniform' marks a model trained with the fixed U[-1, 1] action scaler; default = expert statistics.)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

LEWM = Path(__file__).resolve().parent
sys.path.insert(0, str(LEWM))


_RD: dict = {}


def _reader_init(h5_path):
    import h5py
    import hdf5plugin  # noqa: F401
    _RD["f"] = h5py.File(h5_path, "r")


def _read_chunk(job):
    """One compressed chunk range of the pixels column, decompressed once; selected rows into the memmap."""
    out_path, rows, out_idx = job
    lo, hi = int(rows.min()), int(rows.max()) + 1
    block = _RD["f"]["pixels"][lo:hi]
    out = np.load(out_path, mmap_mode="r+")
    out[out_idx] = block[rows - lo]
    out.flush()
    return len(out_idx)


def read_rows(h5_path, rows, out_path, chunk=100, workers=16):
    """pixels[rows] for sorted rows -> .npy at out_path (uint8, memmapped), decompressed in parallel processes."""
    import multiprocessing as mp
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.uint8, shape=(len(rows), 224, 224, 3))
    del out
    cid = rows // chunk
    jobs = []
    for c in np.unique(cid):
        idx = np.nonzero(cid == c)[0]
        jobs.append((str(out_path), rows[idx], idx))
    with mp.get_context("fork").Pool(workers, initializer=_reader_init, initargs=(str(h5_path),)) as pool:
        n = sum(pool.imap_unordered(_read_chunk, jobs, chunksize=4))
    assert n == len(rows)
    return np.load(out_path, mmap_mode="r")


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", action="append", required=True, help="path under $STABLEWM_HOME/checkpoints[:uniform]")
    p.add_argument("--n-fit", type=int, default=20000)
    p.add_argument("--n-test", type=int, default=10000)
    p.add_argument("--n-uniform", type=int, default=16000)
    p.add_argument("--n-pred", type=int, default=3000, help="held-out expert windows for open-loop prediction")
    p.add_argument("--mlp-steps", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    args = parse()
    import h5py
    import hdf5plugin  # noqa: F401
    import torch
    import torch.nn as nn
    import stable_worldmodel as swm
    from stable_worldmodel.wm.utils import load_pretrained

    torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.device("cuda")
    rng = np.random.default_rng(args.seed)
    home = Path(swm.data.utils.get_cache_dir())
    h5_path = home / "datasets" / "pusht_expert_train.h5"
    f = h5py.File(h5_path, "r")
    ep_off, ep_len = f["ep_offset"][:], f["ep_len"][:]
    state_all = f["state"][:]
    act_all = f["action"][:]
    expert_std = act_all.std(0)
    expert_mean = act_all.mean(0)

    cache = home / "probe_cache" / f"seed{args.seed}_fit{args.n_fit}_test{args.n_test}_pred{args.n_pred}_u{args.n_uniform}"
    cache.mkdir(parents=True, exist_ok=True)

    def frames_from(tag, eps, n, rng_s):
        """Every 5th frame of each episode, then a random subset of n. The pixels sit in 100-frame compressed
        chunks, so reads are spread over worker processes (each decompresses whole chunks) and the decoded set is
        cached under $STABLEWM_HOME/probe_cache for every later run."""
        if (cache / f"{tag}_px.npy").exists():
            return np.load(cache / f"{tag}_px.npy", mmap_mode="r"), np.load(cache / f"{tag}_st.npy")
        rows = np.concatenate([ep_off[e] + np.arange(0, ep_len[e], 5) for e in eps])
        keep = np.sort(rng_s.choice(len(rows), size=min(n, len(rows)), replace=False))
        sel = rows[keep]
        px = read_rows(h5_path, sel, cache / f"{tag}_px.npy")
        st = state_all[sel]
        np.save(cache / f"{tag}_st.npy", st)
        return px, st

    fit_px, fit_st = frames_from("fit", range(0, 2000), args.n_fit, np.random.default_rng(args.seed))
    test_px, test_st = frames_from("test", range(17000, len(ep_off)), args.n_test, np.random.default_rng(args.seed + 1))

    # uniform random states rendered by the stock env
    import pusht_sim as ps
    env = ps.make_env()
    U = args.n_uniform
    rng = np.random.default_rng(args.seed + 2)   # uniform states: drawn identically whether or not frames are cached
    us = np.zeros((U, ps.STATE_DIM))
    us[:, 0:2] = rng.uniform(ps.LO, ps.HI, (U, 2))
    us[:, 4:6] = rng.uniform(80, 432, (U, 2))
    us[:, 6] = rng.uniform(0, 2 * np.pi, U)
    if (cache / "uniform_px.npy").exists():
        upx = np.load(cache / "uniform_px.npy", mmap_mode="r")
    else:
        upx = np.empty((U,) + ps.FRAME, np.uint8)
        for i in range(U):
            ps.set_state(env, us[i])
            upx[i] = env.render()
        np.save(cache / "uniform_px.npy", upx)
    u_st = np.concatenate([us[:, 0:2], us[:, 4:6], us[:, 6:7]], 1)    # same layout as the dataset's state[:5]

    # held-out expert windows for open-loop prediction: frames t, t+5, ..., t+40 (3 context + 5 targets)
    rng = np.random.default_rng(args.seed + 3)   # held-out windows
    win_rows = []
    for e in rng.permutation(np.arange(17000, len(ep_off))):
        if ep_len[e] >= 45:
            s0 = rng.integers(0, ep_len[e] - 44)
            win_rows.append(ep_off[e] + s0)
        if len(win_rows) >= args.n_pred:
            break
    win_rows = np.sort(np.array(win_rows))
    flat = (win_rows[:, None] + np.arange(0, 41, 5)[None]).reshape(-1)    # (W * 9,) sorted (windows disjoint)
    if (cache / "win_px.npy").exists():
        wflat = np.load(cache / "win_px.npy", mmap_mode="r")
    else:
        wflat = read_rows(h5_path, flat, cache / "win_px.npy")
    wpx = np.asarray(wflat).reshape(len(win_rows), 9, *ps.FRAME)          # (W, 9, H, W, 3)
    wst = state_all[flat].reshape(len(win_rows), 9, -1)
    wact = np.stack([act_all[r:r + 40] for r in win_rows])                # (W, 40, 2)

    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    @torch.no_grad()
    def encode(model, px, bs=512):
        out = []
        for i in range(0, len(px), bs):
            x = torch.as_tensor(px[i:i + bs], device=dev).permute(0, 3, 1, 2).float().div_(255)
            x = (x - mean) / std
            h = model.encoder(x, interpolate_pos_encoding=True).last_hidden_state[:, 0]
            out.append(model.projector(h).float())
        return torch.cat(out)

    def targets(st):  # agent xy, T xy, cos, sin
        return np.concatenate([st[:, 0:4], np.cos(st[:, 4:5]), np.sin(st[:, 4:5])], 1).astype(np.float64)

    def ridge(Zf, Yf):
        mu, sd = Zf.mean(0), Zf.std(0) + 1e-6
        X = (Zf - mu) / sd
        X1 = np.concatenate([X, np.ones((len(X), 1))], 1)
        lam = 1e-3 * np.trace(X1.T @ X1) / X1.shape[1]
        W = np.linalg.solve(X1.T @ X1 + lam * np.eye(X1.shape[1]), X1.T @ Yf)
        return lambda Z: np.concatenate([(Z - mu) / sd, np.ones((len(Z), 1))], 1) @ W

    def mlp(Zf, Yf):
        mu, sd = Zf.mean(0), Zf.std(0) + 1e-6
        ym, ys = Yf.mean(0), Yf.std(0) + 1e-9
        X = torch.as_tensor((Zf - mu) / sd, dtype=torch.float32, device=dev)
        Y = torch.as_tensor((Yf - ym) / ys, dtype=torch.float32, device=dev)
        torch.manual_seed(args.seed)
        net = nn.Sequential(nn.Linear(X.shape[1], 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(),
                            nn.Linear(512, Y.shape[1])).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        for _ in range(args.mlp_steps):
            idx = torch.randint(0, len(X), (1024,), device=dev)
            loss = (net(X[idx]) - Y[idx]).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()

        def pred(Z):
            with torch.no_grad():
                out = net(torch.as_tensor((Z - mu) / sd, dtype=torch.float32, device=dev)).cpu().numpy()
            return out * ys + ym
        return pred

    def metrics(Yp, Yt):
        r2 = lambda a, b: 1 - ((a - b) ** 2).sum(0) / ((b - b.mean(0)) ** 2).sum(0)
        pr = lambda a, b: np.array([np.corrcoef(a[:, j], b[:, j])[0, 1] for j in range(a.shape[1])])
        R2, PR = r2(Yp, Yt), pr(Yp, Yt)
        ang_p = np.arctan2(Yp[:, 5], Yp[:, 4])
        ang_t = np.arctan2(Yt[:, 5], Yt[:, 4])
        dang = np.abs((ang_p - ang_t + np.pi) % (2 * np.pi) - np.pi)
        return {"agent_r2": float(R2[0:2].mean()), "block_r2": float(R2[2:4].mean()), "angle_r2": float(R2[4:6].mean()),
                "agent_r": float(PR[0:2].mean()), "block_r": float(PR[2:4].mean()), "angle_r": float(PR[4:6].mean()),
                "agent_err_px": float(np.linalg.norm(Yp[:, 0:2] - Yt[:, 0:2], axis=1).mean()),
                "block_err_px": float(np.linalg.norm(Yp[:, 2:4] - Yt[:, 2:4], axis=1).mean()),
                "angle_err_deg": float(np.degrees(dang).mean())}

    results = {}
    for spec in args.ckpt:
        path, _, norm = spec.partition(":")
        model = load_pretrained(path).to(dev).eval()
        model.requires_grad_(False)
        res = {"action_norm": norm or "expert"}
        Zf = encode(model, fit_px).cpu().numpy()
        Zt = encode(model, test_px).cpu().numpy()
        Yf, Yt = targets(fit_st), targets(test_st)
        lin = ridge(Zf, Yf)
        res["expert_linear"] = metrics(lin(Zt), Yt)
        res["expert_mlp"] = metrics(mlp(Zf, Yf)(Zt), Yt)
        Zu = encode(model, upx).cpu().numpy()
        Yu = targets(u_st)
        h = U // 2
        res["uniform_linear"] = metrics(ridge(Zu[:h], Yu[:h])(Zu[h:]), Yu[h:])
        res["uniform_mlp"] = metrics(mlp(Zu[:h], Yu[:h])(Zu[h:]), Yu[h:])
        # latent statistics on expert test frames
        C = np.cov(Zt.T)
        ev = np.clip(np.linalg.eigvalsh(C), 0, None)
        res["latent"] = {"eff_rank": float(np.exp(-(ev / ev.sum() * np.log(ev / ev.sum() + 1e-12)).sum())),
                         "mean_std": float(Zt.std(0).mean())}
        # open-loop prediction through the linear expert probe
        if norm == "uniform":
            a_n = wact / (1 / math.sqrt(3))
        else:
            a_n = (wact - expert_mean) / expert_std
        A = torch.as_tensor(a_n.reshape(len(win_rows), 8, 10), dtype=torch.float32, device=dev)
        Zw = encode(model, wpx.reshape(-1, *wpx.shape[2:])).view(len(win_rows), 9, -1)
        ctx = Zw[:, :3].clone()
        acts = A[:, :3]
        errs = {"pred": [], "true": [], "copy": []}
        with torch.no_grad():
            for k in range(1, 6):
                ae = model.action_encoder(acts)
                nxt = model.predict(ctx[:, -3:], ae[:, -3:])[:, -1:]
                ctx = torch.cat([ctx, nxt], 1)
                acts = torch.cat([acts, A[:, 2 + k:3 + k]], 1)
                t_idx = 2 + k
                Yk = targets(wst[:, t_idx])
                for tag, z in (("pred", nxt[:, 0]), ("true", Zw[:, t_idx]), ("copy", Zw[:, 2])):
                    m_ = metrics(lin(z.cpu().numpy()), Yk)
                    errs[tag].append({"k": k, "block_err_px": m_["block_err_px"], "agent_err_px": m_["agent_err_px"],
                                      "angle_err_deg": m_["angle_err_deg"]})
        res["openloop_linear_probe"] = errs
        results[spec] = res
        print(json.dumps({spec: {k: v for k, v in res.items() if k != "openloop_linear_probe"}}, indent=1), flush=True)
        for tag in ("pred", "true", "copy"):
            print(f"  open-loop {tag:4s} block px k=1..5: " +
                  " ".join(f"{e['block_err_px']:.1f}" for e in errs[tag]), flush=True)
        del model
        torch.cuda.empty_cache()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
