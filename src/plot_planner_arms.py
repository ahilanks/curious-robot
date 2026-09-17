"""Summary figure for the RP1 planner experiment (2026-09-17): per-arm curves from runs/<name>/metrics.jsonl.

  python src/plot_planner_arms.py rp1x_l2 rp1x_val rp1x_rp1 rp1x_l2s rp1x_vals rp1x_l2s_h3 rp1x_rp1_h3 --out runs/rp1x_summary.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PANELS = [
    ("goal/curric_d", "distance ladder d"),
    ("goal/dist_to_goal", "||z - z*|| to goal (eps 2.8)"),
    ("goal/dwell_hold_frac", "dwell hold fraction (parked in the ball)"),
    ("smooth/tau_sat_frac", "torque saturation fraction"),
    ("interact/contacts_per_step", "contacts / step"),
    ("interact/object_motion", "object motion"),
]


def load(name):
    f = Path("runs") / name / "metrics.jsonl"
    if not f.exists():
        return []
    rows = []
    for line in f.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def smooth(x, k=10):
    """trailing running mean with a growing window at the start (no edge distortion)."""
    x = np.asarray(x, float)
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.empty_like(x)
    for i in range(len(x)):
        lo = max(0, i - k + 1)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="runs/rp1x_summary.png")
    a = ap.parse_args()
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for name in a.runs:
        rows = load(name)
        if not rows:
            continue
        for ax, (key, title) in zip(axes.flat, PANELS):
            pts = [(r["step"], r[key]) for r in rows if key in r and r[key] is not None and np.isfinite(r[key])]
            if not pts:
                continue
            s, v = zip(*pts)
            ax.plot(s, smooth(v), label=name, lw=1.4)
            ax.set_title(title); ax.set_xlabel("decision"); ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("RP1 planner experiment from wr_sleepret2@200k — l2: CEM+L2 · val: CEM+critic · rp1: RP1 refiner · s: --plan-act-scale · h3: horizon 3")
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=110)
    print("saved", a.out)


if __name__ == "__main__":
    main()
