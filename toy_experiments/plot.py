"""Plots + summary table for toy_signals.py runs.

    python plot.py --runs runs/          # -> runs/mse_vs_steps.png, visitation.png, final_fit.png,
                                          #    policy.png, summary.md
"""
from __future__ import annotations

import argparse
import glob
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ORDER = ["mse", "rnd", "lp", "uniform", "walk"]
COL = {"mse": "#2a78d6", "rnd": "#eb6834", "lp": "#1baf7a", "uniform": "#52514e", "walk": "#52514e"}
LS = {"mse": "-", "rnd": "-", "lp": "-", "uniform": "--", "walk": ":"}
LABEL = {"mse": "mse (prediction error)", "rnd": "rnd (novelty)", "lp": "lp (learning progress)",
         "uniform": "uniform sampling", "walk": "random walk"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "legend.frameon": False})


def load(runs_dir):
    by = {}
    for p in sorted(glob.glob(os.path.join(runs_dir, "*_s*.npz"))):
        d = np.load(p)
        by.setdefault(str(d["method"]), []).append(d)
    return {m: by[m] for m in ORDER if m in by}


def smooth(v, k):
    if k <= 1:
        return v
    c = np.cumsum(np.insert(v, 0, 0.0))
    out = (c[k:] - c[:-k]) / k
    return np.concatenate([np.full(k - 1, np.nan), out])


def fig_mse(by, out):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [3, 1.2]})
    floors = []
    for m, runs in by.items():
        steps = runs[0]["eval_steps"]
        M = np.stack([r["eval_mse"] for r in runs])
        mu, sd = M.mean(0), M.std(0)
        ax.plot(steps, mu, LS[m], color=COL[m], lw=2, label=LABEL[m])
        if m not in ("uniform", "walk"):
            ax.fill_between(steps, mu - sd, mu + sd, color=COL[m], alpha=0.12, lw=0)
        ax.annotate(m, (steps[-1], mu[-1]), xytext=(4, 0), textcoords="offset points", color=INK2, va="center")
        floors += [float(r["oracle"]) for r in runs if np.isfinite(r["oracle"])]
        finals = M[:, -1]
        x0 = ORDER.index(m)
        ax2.scatter(np.full(len(finals), x0) + np.linspace(-0.15, 0.15, len(finals)), finals, s=18,
                    color=COL[m], marker="o" if m not in ("uniform", "walk") else "x", zorder=3)
        ax2.hlines(finals.mean(), x0 - 0.3, x0 + 0.3, color=COL[m], lw=2)
    if floors:
        fl = float(np.mean(floors))
        ax.axhline(fl, color=INK2, lw=1, ls=(0, (2, 3)))
        ax.annotate(f"capacity floor (full-grid fit) {fl:.3f}", (0, fl), xytext=(4, 4), textcoords="offset points",
                    color=INK2, fontsize=8)
        ax2.axhline(fl, color=INK2, lw=1, ls=(0, (2, 3)))
    ax.set_yscale("log"); ax.set_xlabel("samples taken (steps)"); ax.set_ylabel("MSE of g vs f on the [-1, 1] grid")
    ax.set_title("Learner error vs samples (mean over seeds, band = ±1 sd)", loc="left", color=INK)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.legend(loc="lower left", ncol=2)
    ax2.set_yscale("log"); ax2.set_xticks(range(len(ORDER))); ax2.set_xticklabels(ORDER)
    ax2.set_title("Final MSE per seed (bar = mean)", loc="left", color=INK); ax2.grid(axis="y", color=GRID, lw=0.6)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def fig_visitation(by, out, seed_idx=0, tbins=50, xbins=40):
    ms = list(by)
    fig, axs = plt.subplots(len(ms), 2, figsize=(11, 2.0 * len(ms)), gridspec_kw={"width_ratios": [4, 1]},
                            squeeze=False)
    for i, m in enumerate(ms):
        r = by[m][min(seed_idx, len(by[m]) - 1)]
        xs = r["xs"]; n = len(xs)
        H, _, _ = np.histogram2d(np.arange(n), xs, bins=[tbins, xbins], range=[[0, n], [-1, 1]])
        ax = axs[i, 0]
        ax.imshow(H.T, origin="lower", aspect="auto", extent=[0, n, -1, 1], cmap="Blues",
                  vmax=np.percentile(H, 99) or 1)
        ax.set_ylabel("x"); ax.set_title(f"{LABEL[m]}  (seed {int(r['seed'])})", loc="left", color=INK)
        if i == len(ms) - 1:
            ax.set_xlabel("step")
        # pooled over seeds: where the samples went, with the target's unfittable structure for reference
        pooled = np.concatenate([q["xs"] for q in by[m]])
        h, e = np.histogram(pooled, bins=xbins, range=(-1, 1), density=True)
        axm = axs[i, 1]
        axm.barh((e[:-1] + e[1:]) / 2, h, height=(e[1] - e[0]) * 0.9, color=COL[m], alpha=0.85)
        axm.axvline(0.5, color=INK2, lw=0.8, ls=(0, (2, 3)))
        axm.set_ylim(-1, 1); axm.set_yticks([]); axm.set_title("all seeds (density)", loc="left", color=INK2, fontsize=8)
        if i == len(ms) - 1:
            axm.set_xlabel("visits; dotted = uniform")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def fig_fit(by, out, seed_idx=0, xbins=40):
    ms = list(by)
    fig, axs = plt.subplots(2, len(ms), figsize=(3.2 * len(ms), 4.6), gridspec_kw={"height_ratios": [3, 1]},
                            squeeze=False, sharex=True)
    for j, m in enumerate(ms):
        r = by[m][min(seed_idx, len(by[m]) - 1)]
        grid, f, g = r["grid"], r["f_grid"], r["g_snaps"][-1]
        ax = axs[0, j]
        ax.plot(grid, f, color=INK2, lw=1, label="target f")
        ax.plot(grid, g, color=COL[m], lw=2, label="learner g (final)")
        ax.set_title(f"{m}: final mse {float(r['eval_mse'][-1]):.3f}", loc="left", color=INK)
        if j == 0:
            ax.legend(loc="upper left", fontsize=8)
        h, e = np.histogram(r["xs"], bins=xbins, range=(-1, 1))
        axh = axs[1, j]
        axh.bar((e[:-1] + e[1:]) / 2, h, width=(e[1] - e[0]) * 0.9, color=COL[m], alpha=0.85)
        axh.set_xlabel("x"); axh.set_xlim(-1, 1)
        if j == 0:
            axh.set_ylabel("visits")
    fig.suptitle(f"Final fit and where the samples went (seed {int(by[ms[0]][0]['seed'])})", x=0.01, ha="left", color=INK)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def fig_policy(by, out, seed_idx=0, k=50):
    learned = [m for m in by if m in ("mse", "rnd", "lp")]
    if not learned:
        return
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 3.6))
    for m in learned:
        r = by[m][min(seed_idx, len(by[m]) - 1)]
        rr = r["r_raw"]; rr = rr / (rr.std() + 1e-8)
        ax.plot(smooth(rr, k), color=COL[m], lw=1.5, label=m)
        ax2.plot(r["grid"], r["pi_grid"], color=COL[m], lw=2, label=m)
    ax.set_title(f"Raw reward (each / its own sd, {k}-step mean), seed 0", loc="left", color=INK)
    ax.set_xlabel("step"); ax.set_ylabel("reward / sd"); ax.legend(); ax.grid(axis="y", color=GRID, lw=0.6)
    ax2.axhline(0, color=INK2, lw=0.8); ax2.set_ylim(-0.11, 0.11)
    ax2.set_title("Final policy: mean delta at x (sign = direction it pushes)", loc="left", color=INK)
    ax2.set_xlabel("x"); ax2.set_ylabel("delta"); ax2.legend(); ax2.grid(axis="y", color=GRID, lw=0.6)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def summary(by, out, xbins=40):
    lines = ["| method | final MSE | best MSE | mean MSE over run | coverage (of 40 bins) | floor |",
             "|---|---|---|---|---|---|"]
    for m, runs in by.items():
        M = np.stack([r["eval_mse"] for r in runs])
        cov = np.mean([np.mean(np.histogram(r["xs"], bins=xbins, range=(-1, 1))[0] > 0) for r in runs])
        fl = np.nanmean([float(r["oracle"]) for r in runs])
        lines.append(f"| {m} | {M[:, -1].mean():.3f} ± {M[:, -1].std():.3f} | {M.min(1).mean():.3f} | "
                     f"{M.mean(1).mean():.3f} | {cov:.2f} | {fl:.3f} |")
    txt = "\n".join(lines) + f"\n\n{len(next(iter(by.values())))} seeds; MSE on a {len(runs[0]['grid'])}-point grid.\n"
    print(txt)
    with open(out, "w") as fh:
        fh.write(txt)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs"))
    p.add_argument("--seed-idx", type=int, default=0, help="which seed the per-run panels show")
    a = p.parse_args()
    by = load(a.runs)
    if not by:
        raise SystemExit(f"no runs in {a.runs}")
    fig_mse(by, os.path.join(a.runs, "mse_vs_steps.png"))
    fig_visitation(by, os.path.join(a.runs, "visitation.png"), a.seed_idx)
    fig_fit(by, os.path.join(a.runs, "final_fit.png"), a.seed_idx)
    fig_policy(by, os.path.join(a.runs, "policy.png"), a.seed_idx)
    summary(by, os.path.join(a.runs, "summary.md"))
    print("figures ->", a.runs)


if __name__ == "__main__":
    main()
