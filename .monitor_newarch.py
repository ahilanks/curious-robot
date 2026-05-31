"""Summarize a curious-robot run's metrics.jsonl for the freezing-diagnosis signals.
Usage: python .monitor_newarch.py runs/newarch/metrics.jsonl
"""
import json, sys

path = sys.argv[1] if len(sys.argv) > 1 else "runs/newarch/metrics.jsonl"
rows = []
with open(path) as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

if not rows:
    print("(no rows yet)")
    sys.exit(0)

last = rows[-1]
print(f"rows={len(rows)}  final step={last.get('step')}  sps~{last.get('perf/steps_per_sec'):.1f}")

def series(key):
    return [(r["step"], r[key]) for r in rows if key in r]

def show(key, fmt="{:.4f}"):
    s = series(key)
    if not s:
        print(f"  {key:30s} (not logged yet)")
        return None
    first_s, first_v = s[0]
    last_s, last_v = s[-1]
    arrow = "→"
    print(f"  {key:30s} {fmt.format(first_v)} (@{first_s}) {arrow} {fmt.format(last_v)} (@{last_s})")
    return last_v

print("\n-- Does it interact (not freeze)? --")
cps = show("interact/contacts_per_step")
show("interact/frac_touch_block")
show("interact/object_motion")

print("\n-- Curiosity / reward --")
show("reward/r_cur", "{:.2f}")
show("reward/r_safe", "{:.2f}")
show("reward/safe_cur_ratio", "{:.3f}")

print("\n-- Is the WM learning dynamics? --")
pl = show("wm/pred_loss", "{:.4f}")
ib = show("wm/identity_baseline", "{:.4f}")
show("wm/h_fwd", "{:.0f}")
show("wm/mse_block", "{:.3f}")
show("wm/mse_none", "{:.3f}")

print("\n-- Representation health (collapse?) --")
show("encoder/z_std", "{:.4f}")
show("encoder/eff_rank", "{:.2f}")
show("encoder/eff_rank_probe", "{:.2f}")   # encoder health on the fixed diverse probe set
show("encoder/feat_corr", "{:.4f}")

print("\n-- Verdict heuristics --")
if pl is not None and ib is not None:
    ratio = pl / ib if ib else float("nan")
    verdict = "LEARNING (pred<baseline)" if ratio < 1 else "NOT beating identity yet"
    print(f"  pred/identity = {ratio:.3f}  -> {verdict}")
else:
    print("  pred_loss not logged yet (still in warmup or WM not updated)")
if cps is not None:
    print(f"  contacts/step last = {cps:.3f}  ({'frozen' if cps < 0.01 else 'interacting'})")
