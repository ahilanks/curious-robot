"""Matched-step cross-run health table for the alpha=0 x lambda_cur sweep.

Usage: python compare_runs.py [STEP] [run ...]
  STEP omitted -> uses the common frontier (largest step ALL runs have reached), so
  staggered runs are still compared apples-to-apples. Each run shows its row nearest
  (<=) the target step.

Decides the three questions:
  arm not stalling      -> arm_spd  (-> 0 == STALL)
  prediction improving  -> pred/per (< 1 == beats persistence)
  encoder not collapsing -> effrnk_p up, zstd_p > 0, fcorr_p < 1
"""
import json, sys, glob, os

args = sys.argv[1:]
target = int(args[0]) if args and args[0].isdigit() else None
runs = [a for a in args if not a.isdigit()] or sorted(
    n for p in glob.glob("runs/*/metrics.jsonl")
    for n in [os.path.basename(os.path.dirname(p))] if n.startswith(("a0_", "a02_", "rnd_")))

COLS = [
    ("step", "step", "{:>6.0f}"),
    ("sps", "perf/steps_per_sec", "{:>5.1f}"),
    ("arm_spd", "interact/arm_speed", "{:>7.3f}"),
    ("contact/s", "interact/contacts_per_step", "{:>9.3f}"),
    ("r_cur", "reward/r_cur", "{:>6.3f}"),
    ("rnd_ctr", "reward/rnd_contrib", "{:>7.2f}"),
    ("pred/per", "wm/pred_over_identity", "{:>8.2f}"),
    ("effrnk_p", "encoder/eff_rank_probe", "{:>8.1f}"),
    ("zstd_p", "encoder/z_std_probe", "{:>6.2f}"),
    ("fcorr_p", "encoder/feat_corr_probe", "{:>7.2f}"),
    ("entropy", "policy/entropy", "{:>9.1f}"),
    ("a|.|", "policy/action_absmean", "{:>5.2f}"),
]

def rows(run):
    p = f"runs/{run}/metrics.jsonl"
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []

allrows = {r: rows(r) for r in runs}
allrows = {r: rr for r, rr in allrows.items() if rr}
if not allrows:
    print("no metrics yet"); sys.exit()

if target is None:                       # common frontier = min over runs of their max step
    target = min(rr[-1]["step"] for rr in allrows.values())

def nearest(rr, step):                   # last row with step <= target
    pick = rr[0]
    for row in rr:
        if row["step"] <= step:
            pick = row
        else:
            break
    return pick

width = lambda f: int(f.split(':')[1].rstrip('}').lstrip('>').split('.')[0]) + 1
print(f"# matched at step <= {target}")
print("run".ljust(13) + "".join(h.rjust(width(f)) + " " for h, _, f in COLS))
print("-" * 110)
for r in runs:
    rr = allrows.get(r)
    if not rr:
        print(r.ljust(13) + "  (no metrics)"); continue
    row = nearest(rr, target)
    line = r.ljust(13)
    for _, key, fmt in COLS:
        v = row.get(key)
        line += (fmt.format(v) if isinstance(v, (int, float)) else " " * width(fmt)) + " "
    print(line)
