"""Aggregate the Push-T curiosity sweep: per-arm world-understanding scores + tests vs the none arm.

    python toy/pusht_report.py [--runs toy/runs/pusht] [--start fixed]

Reads <run>/eval.json (pusht_eval.py score) and <run>/metrics.jsonl. Primary score = E1 "abs"
DEMO skill; each arm vs none by Welch t, Holm-corrected over the arms present; the seeds needed for
80% power at the observed effect use the pooled SD and the worst-case Holm alpha (0.05 / #arms).
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ARMS = ("none", "pred", "lp", "count", "ln")


def seeds_for_power(d, alpha, power=0.8, n_max=200):
    """Per-arm n for a two-sided two-sample t-test to reach `power` at standardised effect d."""
    if not np.isfinite(d) or d == 0:
        return float("inf")
    for n in range(2, n_max + 1):
        df, ncp = 2 * n - 2, abs(d) * math.sqrt(n / 2)
        tc = stats.t.ppf(1 - alpha / 2, df)
        if 1 - stats.nct.cdf(tc, df, ncp) + stats.nct.cdf(-tc, df, ncp) >= power:
            return n
    return float("inf")


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, run = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = run
    return adj


def load(runs, start):
    rows = defaultdict(list)
    for d in sorted(Path(runs).glob(f"pt{start[0]}_*_s*")):
        ev = d / "eval.json"
        if not ev.exists():
            continue
        arm = d.name.split("_")[1]
        e = json.load(open(ev))
        m = [json.loads(l) for l in open(d / "metrics.jsonl")]
        tail = m[int(0.9 * len(m)):]
        g = lambda k: float(np.mean([x[k] for x in tail]))
        rows[arm].append({
            "run": d.name, "n_iters": len(m),
            "abs": e["E1"]["abs"]["DEMO"]["skill"], "tframe": e["E1"]["tframe"]["DEMO"]["skill"],
            "abs_gain": e["E1"]["abs"]["DEMO"]["action_gain"],
            "abs_short": e["E1"]["abs"]["DEMO"]["skill_short"], "abs_long": e["E1"]["abs"]["DEMO"]["skill_long"],
            "push": e["E1"]["abs"]["PUSH"]["skill"], "wall": e["E1"]["abs"]["WALL"]["skill"],
            "inv_free": e["E1"]["abs"]["FREE"]["per_h"]["10"]["invented"],
            "inv_near": e["E1"]["abs"]["NEAR"]["per_h"]["10"]["invented"],
            "e2": e["E2"]["DEMO"]["skill"] if "E2" in e else float("nan"),
            "t_moved": e["t_moved_frac"], "contact": g("contact_rate"), "block_cov": g("block_cov"),
        })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default=str(Path(__file__).resolve().parent / "runs" / "pusht"))
    p.add_argument("--start", default="fixed")
    args = p.parse_args()
    rows = load(args.runs, args.start)
    if not rows:
        print("no scored runs")
        return
    cols = [("abs", "E1 abs DEMO"), ("abs_gain", "act.gain"), ("abs_short", "short"), ("abs_long", "long"),
            ("tframe", "E1 tframe"), ("e2", "E2 own"), ("push", "PUSH"), ("wall", "WALL"),
            ("inv_free", "inv FREE"), ("inv_near", "inv NEAR"), ("t_moved", "T moves"), ("contact", "contact")]
    print(f"Push-T, start={args.start}: mean (sd) over seeds  [primary = E1 abs DEMO skill]")
    print(f"{'arm':6s} {'n':>2s} " + " ".join(f"{h:>14s}" for _, h in cols))
    for arm in ARMS:
        if arm not in rows:
            continue
        r = rows[arm]
        cells = []
        for k, _ in cols:
            v = np.array([x[k] for x in r], dtype=float)
            cells.append(f"{np.nanmean(v):+7.3f} ({np.nanstd(v, ddof=1) if len(v) > 1 else 0:.3f})" if np.isfinite(v).any() else f"{'-':>14s}")
        print(f"{arm:6s} {len(r):2d} " + " ".join(f"{c:>14s}" for c in cells))
    print("\nper seed (E1 abs DEMO skill): " + "; ".join(
        f"{arm} " + " ".join(f"{x['abs']:+.3f}" for x in rows[arm]) for arm in ARMS if arm in rows))
    if "none" not in rows or len(rows["none"]) < 2:
        return
    for key, label in (("abs", "PRIMARY E1 abs"), ("tframe", "secondary E1 tframe")):
        base = np.array([x[key] for x in rows["none"]])
        arms = [a for a in ARMS if a != "none" and a in rows and len(rows[a]) >= 2]
        tests = [stats.ttest_ind([x[key] for x in rows[a]], base, equal_var=False) for a in arms]
        adj = holm([t.pvalue for t in tests])
        alpha = 0.05 / max(1, len(arms))
        print(f"\n{label} DEMO skill vs none (Welch t, Holm over {len(arms)} arms):")
        for a, t, pa in zip(arms, tests, adj):
            v = np.array([x[key] for x in rows[a]])
            diff = v.mean() - base.mean()
            sd = math.sqrt((v.var(ddof=1) + base.var(ddof=1)) / 2)
            n_req = seeds_for_power(diff / sd if sd > 0 else float("inf"), alpha)
            print(f"  {a:6s} diff {diff:+.3f}  t {t.statistic:+6.2f}  p {t.pvalue:.4f}  Holm p {pa:.4f}  "
                  f"{'*' if pa < 0.05 else ' '}  seeds/arm for 80% power at this effect: {n_req}")


if __name__ == "__main__":
    main()
