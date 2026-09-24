"""Why do the explorers go where they go? Reconstruct, from a saved run, the reward landscape each
policy faced over time (the learner's error map, the pool LP map, the RND novelty map) and draw
the trajectory on top, next to the map of STORED rewards SAC actually trained on.

    python diagnose.py --runs runs --seed 0     # -> runs/why.png + numbers
The reconstruction replays the saved visit sequence through the same learner / predictor with the
same seeds, so the maps are exact, not approximations.
"""
from __future__ import annotations

import argparse
import copy
import json
import os

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from plot import COL, INK, INK2, LABEL  # noqa: E402
from toy_signals import METHODS, SAC, init_learner, mlp, random_target  # noqa: E402


def rebuild(d):
    """Re-create f, g (and RND nets for rnd) exactly as run() did for this saved run."""
    a = argparse.Namespace(**json.loads(str(d["args"])))
    seed, method = a.seed, a.method
    gen = torch.Generator().manual_seed(seed)
    f = random_target(a.f_hidden, a.f_depth, a.f_gain_in, a.f_gain_h, gen, a.f_act)
    torch.manual_seed(seed * 7919 + METHODS.index(method))
    g = init_learner(a.g_hidden, a.g_depth, seed)
    _sac = SAC(a.amax, a.sac_hidden, a.gamma, a.tau, a.sac_lr)          # consumes the global RNG as run() did
    rnd = None
    if method == "rnd":
        rnd_t = mlp([1, a.rnd_hidden, a.rnd_hidden, a.rnd_out])
        with torch.no_grad():
            first = [m for m in rnd_t if isinstance(m, torch.nn.Linear)][0]
            first.weight.normal_(0.0, a.rnd_gain_in, generator=gen)
            first.bias.uniform_(-a.rnd_gain_in, a.rnd_gain_in, generator=gen)
        for p in rnd_t.parameters():
            p.requires_grad_(False)
        rnd_p = mlp([1, a.rnd_hidden, a.rnd_hidden, a.rnd_out])
        rnd = (rnd_t, rnd_p, torch.optim.Adam(rnd_p.parameters(), lr=a.rnd_lr))
    return a, f, g, rnd


def landscapes(d, every=25):
    """Replay the visit sequence; return (times, map[T, G]) of the method's reward landscape over the grid."""
    a, f, g, rnd = rebuild(d)
    grid = torch.as_tensor(d["grid"]).unsqueeze(1)
    with torch.no_grad():
        fg = f(grid)
    xs = d["xs"]; N = len(xs)
    opt_g = torch.optim.Adam(g.parameters(), lr=a.g_lr)
    pool_x = torch.zeros(N, 1); pool_f = torch.zeros(N, 1); pool_ema = torch.zeros(N, 1); n = 0
    times, maps = [], []
    for t in range(N):
        xt = torch.tensor([[float(xs[t])]])
        with torch.no_grad():
            ft = f(xt)
            if t % every == 0:
                if a.method == "rnd":
                    m = (rnd[1](grid) - rnd[0](grid)).pow(2).mean(1)
                elif a.method == "lp":
                    if n > 0:
                        phi = (g(pool_x[:n]) - pool_f[:n]).pow(2)
                        lp = (phi - pool_ema[:n]).abs()                          # LP of each pool point NOW
                        w = torch.exp(-0.5 * ((pool_x[:n].T - grid) / a.lp_bw) ** 2)   # (G, n)
                        m = (w * lp.T).sum(1) / w.sum(1).clamp_min(1e-6)
                    else:
                        m = torch.zeros(len(grid))
                else:
                    m = (g(grid) - fg).pow(2).squeeze(1)
                times.append(t); maps.append(m.numpy().copy())
            if a.method == "lp" and n > 0:                                     # the run's per-step re-scoring
                phi = (g(pool_x[:n]) - pool_f[:n]).pow(2)
                pool_ema[:n] += a.lp_alpha * (phi - pool_ema[:n])
            if a.method == "lp":
                pool_x[n] = xt; pool_f[n] = ft; pool_ema[n] = float((g(xt) - ft).pow(2)); n += 1
        if getattr(a, "g_replay", 0) and t > 0:
            ridx = np.random.default_rng(0).integers(0, t, a.g_replay)      # not bit-exact under replay
            bx = torch.cat([xt, torch.as_tensor(xs[ridx]).unsqueeze(1)])
            with torch.no_grad():
                bf = f(bx)
            loss = (g(bx) - bf).pow(2).mean()
        else:
            loss = (g(xt) - ft).pow(2).mean()
        opt_g.zero_grad(); loss.backward(); opt_g.step()
        if rnd is not None:
            lr_ = (rnd[1](xt) - rnd[0](xt)).pow(2).mean()
            rnd[2].zero_grad(); lr_.backward(); rnd[2].step()
    return np.array(times), np.stack(maps)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--every", type=int, default=25)
    a = p.parse_args()
    ms = ["mse", "rnd", "lp"]
    fig, axs = plt.subplots(3, 2, figsize=(13, 9.5), gridspec_kw={"width_ratios": [4, 1.3]})
    nb = 40; edges = np.linspace(-1, 1, nb + 1); centers = (edges[:-1] + edges[1:]) / 2
    for i, m in enumerate(ms):
        d = np.load(os.path.join(a.runs, f"{m}_s{a.seed}.npz"))
        times, M = landscapes(d, a.every)
        xs, rn, rr = d["xs"], d["r_norm"], d["r_raw"]; N = len(xs); grid = d["grid"]
        ax = axs[i, 0]
        Z = np.log10(M.T + 1e-6)
        ax.imshow(Z, origin="lower", aspect="auto", extent=[0, N, -1, 1], cmap="Blues",
                  vmin=np.percentile(Z, 5), vmax=np.percentile(Z, 99.5))
        ax.plot(np.arange(N), xs, color="#e34948", lw=0.7, alpha=0.9)
        ax.set_ylabel("x"); ax.set_title(f"{LABEL[m]}: reward landscape over time (log colour), trajectory in red",
                                          loc="left", color=INK)
        if i == 2:
            ax.set_xlabel("step")
        # what SAC trains on: mean STORED (normalised) reward per x-bin, all steps vs last 1000
        ax2 = axs[i, 1]
        for lo, lab, alpha in ((0, "all steps", 0.35), (N - 1000, "last 1000", 0.9)):
            idx = np.digitize(xs[lo:], edges) - 1
            mean = np.array([rn[lo:][idx == b].mean() if np.any(idx == b) else np.nan for b in range(nb)])
            ax2.barh(centers, mean, height=(edges[1] - edges[0]) * 0.9, color=COL[m], alpha=alpha, label=lab)
        ax2.set_ylim(-1, 1); ax2.set_title("mean stored reward / sd\nby x (what SAC fits)", loc="left",
                                           color=INK2, fontsize=8)
        ax2.legend(fontsize=7, loc="lower right")
        # numbers
        mode = centers[np.histogram(xs[-1000:], bins=edges)[0].argmax()]
        near = np.abs(xs - mode) < 0.1
        late = np.arange(N) >= N - 2000
        print(f"{m:4s} seed {a.seed}: attractor x={mode:+.2f}; last 2000 steps: raw reward at attractor "
              f"{rr[late & near].mean():.4f} vs elsewhere {rr[late & ~near].mean() if (late & ~near).any() else float('nan'):.4f}; "
              f"landscape at attractor (final) {M[-1][np.abs(grid - mode) < 0.1].mean():.4f} vs grid mean {M[-1].mean():.4f}; "
              f"running-sd share from first 200 steps {(rr[:200].var() * 200) / (rr.var() * N):.2f}")
    fig.tight_layout()
    out = os.path.join(a.runs, "why.png"); fig.savefig(out, dpi=120); print("->", out)


if __name__ == "__main__":
    main()
