"""Animate runs side by side: the learner's fit, the explorer's recent trail, and the visit histogram.

    python toy_signals.py --method mse --seed 0 --eval-every 20 --out runs/video_data   # dense snapshots
    python animate.py --runs runs/video_data --out runs/video.mp4
"""
from __future__ import annotations

import argparse
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FFMpegWriter, FuncAnimation  # noqa: E402

from plot import COL, INK, INK2, LABEL  # noqa: E402  (also applies the rcParams)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True, help="dir holding <method>_s<seed>.npz with dense --eval-every")
    p.add_argument("--methods", nargs="+", default=["mse", "rnd", "lp"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--trail", type=int, default=40, help="recent visited points drawn on f")
    p.add_argument("--title", default="")
    p.add_argument("--preview", action="store_true", help="also save first / middle / last frames as PNG")
    a = p.parse_args()
    out = a.out or os.path.join(a.runs, "video.mp4")
    runs = {m: np.load(os.path.join(a.runs, f"{m}_s{a.seed}.npz")) for m in a.methods}
    steps = runs[a.methods[0]]["eval_steps"]
    grid = runs[a.methods[0]]["grid"]
    N = len(runs[a.methods[0]]["xs"])
    nb = 40
    edges = np.linspace(-1, 1, nb + 1)
    centers = (edges[:-1] + edges[1:]) / 2

    fig, axs = plt.subplots(2, len(a.methods), figsize=(4.4 * len(a.methods), 6.2),
                            gridspec_kw={"height_ratios": [3, 1]}, squeeze=False, sharex=True)
    art = {}
    for j, m in enumerate(a.methods):
        r = runs[m]
        ax, axh = axs[0, j], axs[1, j]
        f = r["f_grid"]
        ax.plot(grid, f, color=INK2, lw=1, label="target f")
        (lg,) = ax.plot(grid, r["g_snaps"][0], color=COL[m], lw=2.2, label="learner g")
        trail = ax.scatter([], [], s=14, color=INK, zorder=3)
        (cur,) = ax.plot([], [], "o", ms=11, mfc="none", mec=COL[m], mew=2, zorder=4)
        ax.set_ylim(f.min() - 0.6, f.max() + 0.6); ax.set_xlim(-1, 1)
        ttl = ax.set_title("", loc="left", color=INK, fontsize=10)
        if j == 0:
            ax.legend(loc="lower left", fontsize=8)
        hmax = np.histogram(r["xs"], bins=edges)[0].max()
        bars = axh.bar(centers, np.zeros(nb), width=(edges[1] - edges[0]) * 0.9, color=COL[m], alpha=0.85)
        axh.set_ylim(0, hmax * 1.05); axh.set_xlabel("x")
        if j == 0:
            axh.set_ylabel("visits so far")
        art[m] = (lg, trail, cur, ttl, bars)
    sup = fig.suptitle("", x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    def update(k):
        t = int(steps[k])
        for m in a.methods:
            r = runs[m]
            lg, trail, cur, ttl, bars = art[m]
            lg.set_ydata(r["g_snaps"][k])
            xs = r["xs"][:t]
            if t > 0:
                tr = xs[max(0, t - a.trail):]
                fy = np.interp(tr, grid, r["f_grid"])
                trail.set_offsets(np.c_[tr, fy])
                alpha = np.linspace(0.15, 0.9, len(tr))
                trail.set_facecolors(np.c_[np.zeros((len(tr), 3)), alpha])
                cur.set_data([tr[-1]], [fy[-1]])
                h = np.histogram(xs, bins=edges)[0]
                for b, v in zip(bars, h):
                    b.set_height(v)
            ttl.set_text(f"{LABEL[m]}   mse {float(r['eval_mse'][k]):.3f}")
        sup.set_text(f"{a.title}   step {t} / {N}")
        return []

    anim = FuncAnimation(fig, update, frames=len(steps), blit=False)
    anim.save(out, writer=FFMpegWriter(fps=a.fps, bitrate=2500), dpi=110)
    print("wrote", out, f"({len(steps)} frames @ {a.fps} fps = {len(steps) / a.fps:.0f}s)")
    if a.preview:
        base = os.path.splitext(out)[0]
        for k in (0, len(steps) // 2, len(steps) - 1):
            update(k); fig.savefig(f"{base}_frame{k}.png", dpi=110)


if __name__ == "__main__":
    main()
