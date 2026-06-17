"""Quick collapse/freeze/learning read-out from a run's metrics.jsonl (no W&B needed).

    python src/analyze_run.py <run_name> [<run_name> ...]

Prints the trajectory of the diagnostic signals the logistics doc says to watch
(encoder/eff_rank, interact/object_motion, contacts, curiosity spread, pred vs
persistence) plus a one-line verdict per run. Read-only; safe to run mid-training.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
KEYS = ["step", "encoder/eff_rank", "encoder/z_std", "encoder/feat_corr",
        "wm/pred_loss", "wm/identity_baseline", "reward/r_cur", "reward/cur_contrib",
        "reward/r_safe", "interact/object_motion", "interact/contacts_per_step", "wm/h_fwd"]
HDR = ["step", "effrk", "z_std", "fcorr", "pred", "ident", "r_cur", "cur_c",
       "r_safe", "motion", "cont", "h_fwd"]


def load(name):
    p = ROOT / "runs" / name / "metrics.jsonl"
    if not p.exists():
        print(f"  (no metrics for {name} at {p})")
        return []
    return [json.loads(l) for l in open(p) if l.strip()]


def col(rows, key):
    return np.array([r.get(key, float("nan")) for r in rows], float)


def spread(rows, key, window=20):
    """std of a per-step series over the last `window` logged points -> proxy for the
    per-state spread SAC sees (a curiosity term with ~0 spread can't drive exploration)."""
    v = col(rows, key)
    v = v[~np.isnan(v)]
    return float(v[-window:].std()) if len(v) else float("nan")


def show(name):
    rows = load(name)
    if not rows:
        return
    print(f"\n===== {name}  ({len(rows)} logs, step {rows[-1]['step']}) =====")
    print("  ".join(f"{h:>7}" for h in HDR))
    idx = list(range(0, len(rows), max(1, len(rows) // 14))) + [len(rows) - 1]
    for i in sorted(set(idx)):
        r = rows[i]
        print("  ".join(f"{r.get(k, float('nan')):7.2f}" if isinstance(r.get(k), (int, float))
                         else f"{'-':>7}" for k in KEYS))
    # verdict over the second half (steady-state)
    h = len(rows) // 2
    er = col(rows, "encoder/eff_rank"); er = er[~np.isnan(er)]
    mo = col(rows, "interact/object_motion")
    pred = col(rows, "wm/pred_loss"); ident = col(rows, "wm/identity_baseline")
    er_last = er[-1] if len(er) else float("nan")
    mo_last = float(np.nanmean(mo[-10:])) if len(mo) else float("nan")
    gap = float(np.nanmean((pred - ident)[h:])) if len(pred) else float("nan")
    cur_lvl = float(np.nanmean(col(rows, "reward/cur_contrib")[h:]))   # ~82 symlog (saturated) vs ~0 normalized
    rs = col(rows, "reward/r_safe"); rs_last = float(np.nanmean(rs[-10:])) if len(rs) else float("nan")
    cnt = col(rows, "interact/contacts_per_step"); cnt_last = float(np.nanmean(cnt[-10:])) if len(cnt) else float("nan")
    print(f"  VERDICT: eff_rank_last={er_last:.2f}  r_safe_last={rs_last:.1f}  motion_last={mo_last:.4f}  "
          f"contacts_last={cnt_last:.3f}  (pred-persist)_2ndhalf={gap:+.3f}")
    flags = []
    flags.append("COLLAPSE" if er_last < 4 else "rank-ok")        # eff_rank -> 1-3 = collapsed
    # the freeze attractor pins r_safe ~0 (no motion -> no jerk penalty); active motion keeps it strongly negative
    flags.append("FROZEN" if rs_last > -10 and mo_last < 0.005 else "active")
    flags.append("WM>persist" if gap < -0.02 else "WM<=persist")  # is the WM beating the identity baseline
    print(f"  FLAGS: {' | '.join(flags)}")


if __name__ == "__main__":
    names = sys.argv[1:] or ["beta5_10k"]
    for n in names:
        show(n)
