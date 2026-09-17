"""Read-out for the RP1 three-arm planner experiment (2026-09-17): CEM+latent-L2 vs CEM+critic vs RP1.

Reads runs/<name>/metrics.jsonl (the local twin of the W&B stream) and runs/<name>/train.log
(the [goal-curriculum] advance lines), prints per arm the final-window means of the
pre-registered metrics and the distance-ladder timeline.

  python src/compare_planner_arms.py rp1x_l2 rp1x_val rp1x_rp1 [--window 2000]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

KEYS = [
    ("goal/curric_d", "d", "↑"),
    ("goal/curric_mse_pctl", "pctl", "↑"),
    ("goal/arrival_rate", "arrival", "↑"),
    ("goal/dist_to_goal", "dist", "↓"),
    ("goal/qpos_reach", "qpos_reach", "↑"),
    ("goal/reach_rate", "reach", "↑"),
    ("interact/contacts_per_step", "cont/s", "↑"),
    ("interact/object_motion", "obj_mot", "↑"),
    ("cem/reach_gap", "gap", "↓"),
    ("plan/v0", "v0", "~"),
    ("plan/vK", "vK", "↓"),
    ("vcritic/loss", "crit_loss", "↓"),
    ("perf/steps_per_sec", "sps", "↑"),
]


def load(run_dir: Path) -> list[dict]:
    rows = []
    f = run_dir / "metrics.jsonl"
    if not f.exists():
        return rows
    for line in f.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def wmean(rows, key, lo, hi):
    v = [r[key] for r in rows if lo <= r.get("step", -1) < hi and key in r and r[key] is not None
         and np.isfinite(r[key])]
    return float(np.mean(v)) if v else float("nan")


def ladder(run_dir: Path):
    """[goal-curriculum] step=S ... -> distance d=X lines -> [(S, X)]."""
    f = run_dir / "train.log"
    if not f.exists():
        return []
    out = []
    for line in f.read_text(errors="ignore").splitlines():
        m = re.search(r"\[goal-curriculum\] step=(\d+).*distance d=([0-9.]+)", line)
        if m:
            out.append((int(m.group(1)), float(m.group(2))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--window", type=int, default=2000, help="final-window width (steps)")
    a = ap.parse_args()
    print(f"{'arm':10s} {'steps':>6s} " + " ".join(f"{h:>10s}" for _, h, _ in KEYS))
    print(f"{'':10s} {'':>6s} " + " ".join(f"{d:>10s}" for _, _, d in KEYS))
    for name in a.runs:
        rd = Path("runs") / name
        rows = load(rd)
        if not rows:
            print(f"{name:10s} (no metrics.jsonl)")
            continue
        last = max(r.get("step", 0) for r in rows)
        vals = [wmean(rows, k, last - a.window, last + 1) for k, _, _ in KEYS]
        print(f"{name:10s} {last:6d} " + " ".join(f"{v:10.3f}" for v in vals))
    print("\nladder (step -> d):")
    for name in a.runs:
        lad = ladder(Path("runs") / name)
        print(f"  {name:10s} " + (", ".join(f"{s}->{d:g}" for s, d in lad) if lad else "(no advance yet)"))


if __name__ == "__main__":
    main()
