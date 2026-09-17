"""Pool the block-shift pursuit probe over scene seeds (2026-09-17 multi-seed follow-up).

  python src/aggregate_probe_seeds.py runs/probe_rp1x_eps05 runs/probe_rp1x_eps05_s42 ... [--push-mm 10 --shove-mm 2]

Per head: n scenes, median / mean deliberate closure (shift - ctrl), the count of CONDITION-SPECIFIC
pushes (shift closure >= push_mm AND ctrl closure < shove_mm), of CONDITION-BLIND shoves (ctrl >=
push_mm), and the per-scene lists, so a one-scene signal can be told from a distribution.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--push-mm", type=float, default=10.0)
    ap.add_argument("--shove-mm", type=float, default=2.0)
    a = ap.parse_args()
    pooled: dict[str, list] = {}
    for d in a.dirs:
        f = Path(d) / "results.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        for head, v in r.items():
            for s in v["scenes"]:
                pooled.setdefault(head, []).append((Path(d).name, s["shift"]["closure_mm"], s["ctrl"]["closure_mm"],
                                                    s["shift"]["contacts"], s["shift"]["d0"]))
    print(f"{'head':6s} {'n':>3s} {'delib med':>10s} {'delib mean':>11s} {'shift>=push & ctrl<shove':>26s} {'ctrl>=push':>11s} {'shift>=push':>12s} {'d0 med':>7s}")
    for head, rows in pooled.items():
        sh = np.array([r[1] for r in rows]); ct = np.array([r[2] for r in rows]); d0 = np.array([r[4] for r in rows])
        delib = sh - ct
        push = int(((sh >= a.push_mm) & (ct < a.shove_mm)).sum())
        shove = int((ct >= a.push_mm).sum())
        print(f"{head:6s} {len(rows):3d} {np.median(delib):10.1f} {delib.mean():11.1f} {push:26d} {shove:11d} {int((sh >= a.push_mm).sum()):12d} {np.median(d0):7.2f}")
    print("\nper-scene (dir, shift mm, ctrl mm, contacts, d0) for scenes with shift >= push_mm or ctrl >= push_mm:")
    for head, rows in pooled.items():
        hits = [r for r in rows if r[1] >= a.push_mm or r[2] >= a.push_mm]
        print(f"  {head}: " + "; ".join(f"{r[0].split('_')[-1]}: {r[1]:.0f}/{r[2]:.0f} c{r[3]} d0 {r[4]:.1f}" for r in hits))


if __name__ == "__main__":
    main()
