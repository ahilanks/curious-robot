"""Figures + table for the point-push curiosity sweep (toy/run_sweep.sh).

    python toy/plot_curiosity.py [--runs toy/runs] [--tail 30]

  curves.png   ground-truth behaviour vs env steps, one row per condition (clean / noisy TV),
               mean over seeds with a min-max band
  where.png    where each signal sends the agent and the block: visit densities over the last
               10% of training, summed over seeds (log scale)
  summary.txt  last-`tail`-iteration means (+- sd over seeds), plus where each reward pays:
               its mean on contact / wall / TV steps as a multiple of free space, iters 0-20
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Circle

SIGNALS = ["pred", "lp", "lps", "ln", "none", "count"]
LABEL = {"pred": "prediction error", "lp": "learning progress |e_old - e_now|",
         "lps": "learning progress, signed (gamma-progress)", "ln": "learnable novelty (EpiJEPA score)",
         "none": "random walk (no reward)", "count": "count oracle (true state)"}
SHORT = {"pred": "pred error", "lp": "LP |.|", "lps": "LP signed", "ln": "learnable novelty", "none": "random walk", "count": "count oracle"}
# signals of interest: the first three categorical slots (validated all-pairs); the signed-LP variant
# shares LP's hue with a dash (a 4th hue fails the all-pairs floors); references in neutral ink
COLOR = {"pred": "#2a78d6", "lp": "#eb6834", "lps": "#eb6834", "ln": "#1baf7a", "none": "#898781",
         "count": "#52514e"}
STYLE = {"pred": "-", "lp": "-", "lps": (0, (5, 1.5)), "ln": "-", "none": (0, (4, 2)), "count": (0, (1, 1.5))}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf",
                                                    "#184f95", "#0d366b"])
METRICS = [("agent_cov", "agent coverage / episode", "fraction of 10x10 agent grid"),
           ("block_cov", "block coverage / episode", "fraction of 10x10 block grid"),
           ("contact_rate", "pushing the block", "fraction of steps"),
           ("wall_rate", "pinned against a wall", "fraction of steps"),
           ("tv_frac", "on the noisy TV", "fraction of steps")]


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(1)
    ax.tick_params(colors=MUTED, labelsize=8, length=3, width=1)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def load(runs: Path):
    data = {}
    for d in sorted(runs.iterdir()):
        if not (d / "metrics.jsonl").exists() or not (d / "final.pt").exists():
            continue
        cfg = json.load(open(d / "config.json"))
        key = (cfg["signal"], bool(cfg["tv"]))
        rows = [json.loads(l) for l in open(d / "metrics.jsonl")]
        data.setdefault(key, []).append({"cfg": cfg, "rows": rows, "name": d.name,
                                         "final": torch.load(d / "final.pt", weights_only=True)})
    return data


def series(runs, key):
    n = min(len(r["rows"]) for r in runs)
    x = np.array([r["env_steps"] for r in runs[0]["rows"][:n]]) / 1e6
    y = np.array([[row.get(key, np.nan) for row in r["rows"][:n]] for r in runs], dtype=float)
    return x, y


def curves(data, out):
    fig, axes = plt.subplots(2, len(METRICS), figsize=(17, 6.6), sharex=True)
    fig.patch.set_facecolor(SURFACE)
    for row, tv in enumerate((False, True)):
        for col, (key, title, ylab) in enumerate(METRICS):
            ax = axes[row, col]
            if key == "tv_frac" and not tv:
                ax.axis("off")
                ax.text(0.5, 0.5, "no TV in the\nclean condition", ha="center", va="center", color=MUTED,
                        fontsize=9, transform=ax.transAxes)
                continue
            style_axes(ax)
            for sig in SIGNALS:
                runs = data.get((sig, tv))
                if not runs:
                    continue
                x, y = series(runs, key)
                lo, hi, mu = np.nanmin(y, 0), np.nanmax(y, 0), np.nanmean(y, 0)
                ax.fill_between(x, lo, hi, color=COLOR[sig], alpha=0.10, linewidth=0)
                ax.plot(x, mu, color=COLOR[sig], lw=2 if sig in ("pred", "lp", "lps", "ln") else 1.6,
                        linestyle=STYLE[sig], solid_capstyle="round", label=LABEL[sig])
            ax.set_ylim(bottom=0)
            ax.set_title(title, fontsize=10, color=INK, loc="left")
            if col == 0:
                ax.set_ylabel(("NOISY TV\n" if tv else "CLEAN\n") + ylab, fontsize=9, color=INK2)
            else:
                ax.set_ylabel(ylab, fontsize=8, color=MUTED)
            if row == 1:
                ax.set_xlabel("env steps (millions)", fontsize=8, color=MUTED)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=9, labelcolor=INK2,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Point-push, PPO on each intrinsic signal alone: what the agent physically does "
                 "(mean of 3 seeds, band = min-max)", y=1.075, fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def where(data, out, env_geom):
    cols = [(False, "agent"), (False, "block"), (True, "agent"), (True, "block")]
    fig, axes = plt.subplots(len(SIGNALS), 4, figsize=(10.5, 2.55 * len(SIGNALS)))
    fig.patch.set_facecolor(SURFACE)
    for c, (tv, what) in enumerate(cols):
        axes[0, c].set_title(f"{'noisy TV' if tv else 'clean'}: {what} position", fontsize=10, color=INK)
    for r, sig in enumerate(SIGNALS):
        for c, (tv, what) in enumerate(cols):
            ax = axes[r, c]
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(AXIS)
            runs = data.get((sig, tv))
            if not runs:
                ax.axis("off")
                continue
            h = sum(run["final"][f"hist_{what}"] for run in runs).numpy()
            p = h / max(h.sum(), 1)
            ax.imshow(np.ma.masked_equal(p.T, 0), origin="lower", extent=(0, 1, 0, 1), cmap=BLUES,
                      norm=LogNorm(vmin=1e-5, vmax=0.1), interpolation="nearest")
            ax.set_facecolor(SURFACE)
            ax.add_patch(Circle(env_geom["start_block"], env_geom["block_r"], fill=False, ec=INK2, lw=0.8,
                                ls=(0, (2, 2))))
            ax.plot(*env_geom["start_agent"], marker="o", ms=4, color=INK, mec=SURFACE, mew=1)
            if tv:
                ax.add_patch(Circle(env_geom["tv_center"], env_geom["tv_r"], fill=False, ec="#e34948", lw=1))
            if c == 0:
                ax.set_ylabel(SHORT[sig], fontsize=10, color=INK)
    fig.text(0.5, -0.01, "visit density over the last 10% of training, 3 seeds summed, log scale "
             "(light = rare, dark = frequent).  dot = agent start, dashed circle = block start, "
             "red circle = noisy TV", ha="center", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def summary(data, tail):
    keys = ["agent_cov", "block_cov", "block_disp", "contact_rate", "wall_rate", "tv_frac", "cells_cum"]
    lines = []
    for tv in (False, True):
        lines.append(f"\n== {'NOISY TV' if tv else 'CLEAN'}: last {tail} iterations, mean +- sd over seeds ==")
        lines.append(f"{'signal':18s}" + "".join(f"{k:>16s}" for k in keys) + "   seeds")
        for sig in SIGNALS:
            runs = data.get((sig, tv))
            if not runs:
                continue
            vals = []
            for k in keys:
                per = [np.nanmean([row.get(k, np.nan) for row in r["rows"][-tail:]]) for r in runs]
                vals.append(f"{np.mean(per):9.3f} +-{np.std(per):5.3f}")
            lines.append(f"{SHORT[sig]:18s}" + "".join(f"{v:>16s}" for v in vals) + f"   {len(runs)}")
        lines.append(f"-- where the reward pays, iters 0-20: mean reward on X-steps / free-space steps --")
        for sig in ("pred", "lp", "lps", "ln", "count"):
            runs = data.get((sig, tv))
            if not runs:
                continue
            out = []
            for k in ("r_contact", "r_wall", "r_tv"):
                if k == "r_tv" and not tv:
                    continue
                ratio = [np.nanmean([row[k] / row["r_free"] for row in r["rows"][:21]
                                     if row["r_free"] > 0 and not math.isnan(row[k])]) for r in runs]
                out.append(f"{k[2:]}/free {np.nanmean(ratio):8.2f}x")
            lines.append(f"{SHORT[sig]:18s}  " + "   ".join(out))
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default=str(Path(__file__).resolve().parent / "runs"))
    p.add_argument("--tail", type=int, default=30)
    a = p.parse_args()
    runs = Path(a.runs)
    data = load(runs)
    from point_push import PointPush
    e = PointPush(1, tv=True)
    geom = {"start_agent": e.start_agent.tolist(), "start_block": e.start_block.tolist(),
            "block_r": e.block_r, "tv_center": e.tv_center.tolist(), "tv_r": e.tv_r}
    curves(data, runs / "curves.png")
    where(data, runs / "where.png", geom)
    txt = summary(data, a.tail)
    (runs / "summary.txt").write_text(txt + "\n")
    print(txt)
    print(f"\nwrote {runs / 'curves.png'}, {runs / 'where.png'}, {runs / 'summary.txt'}")


if __name__ == "__main__":
    main()
