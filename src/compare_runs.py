"""Compare the 2026-06-12 sim-campaign runs on smoothness / WM-health / interaction.

Reads runs/<name>/metrics.jsonl, averages each metric over the last `--window` logged
points (and over a matched mid-run window for trend), prints a ranked table. The
campaign question: which reward shaping gives the smoothest, simplest, most
hardware-transferable motion without killing curiosity/WM learning?
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

KEYS = [
    ("smooth/action_rate", "rate", "↓"),
    ("smooth/qd_reversal_frac", "rev%", "↓"),
    ("smooth/tau_sat_frac", "sat%", "↓"),
    ("smooth/energy", "energy", "↓"),
    ("smooth/qd_mean", "|qd|", "~"),
    ("reward/r_safe", "r_safe", "↑"),
    ("reward/cur_contrib", "cur", "↑"),
    ("interact/contacts_per_step", "cont/s", "↑"),
    ("wm/pred_persist", "p/p", "↓"),
    ("encoder/eff_rank_probe", "effR", "↑"),
]


def load(run_dir: Path) -> list[dict]:
    rows = []
    f = run_dir / "metrics.jsonl"
    if not f.exists():
        return rows
    for line in f.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "wm/pred_loss" in r and "wm/identity_baseline" in r:
            r["wm/pred_persist"] = r["wm/pred_loss"] / max(r["wm/identity_baseline"], 1e-9)
        rows.append(r)
    return rows


def window_mean(rows: list[dict], key: str, lo: int, hi: int) -> float:
    vals = [r[key] for r in rows if key in r and lo <= r["step"] < hi]
    return float(np.mean(vals)) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", default=[])
    ap.add_argument("--window", type=int, default=1000, help="final-window width (steps)")
    args = ap.parse_args()
    names = args.runs or sorted(
        p.name for p in Path("runs").iterdir()
        if (p / "metrics.jsonl").exists() and not p.name.startswith(("smoke", "sim_scales"))
        and not p.name.endswith(("_oom1", "_rogue")))

    print(f"{'run':10s} {'steps':>6s}", *[f"{h:>8s}" for _, h, _ in KEYS])
    print(f"{'':10s} {'':>6s}", *[f"{d:>8s}" for _, _, d in KEYS])
    for name in names:
        rows = load(Path("runs") / name)
        if not rows:
            continue
        last = rows[-1]["step"]
        lo = max(0, last - args.window + 1)
        cells = [window_mean(rows, k, lo, last + 1) for k, _, _ in KEYS]
        print(f"{name:10s} {last:>6d}", *[f"{c:8.3f}" if np.isfinite(c) else f"{'-':>8s}" for c in cells])


if __name__ == "__main__":
    main()
