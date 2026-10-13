"""Tabulate the Push-T learning-signal study: planning success, probes, open-loop prediction, and what data each
from-scratch arm collected (vs the expert dataset).

    .venv/bin/python report_pusht_fs.py --probe runs/probe_all.json --out runs/pusht_fs_report.md
Reads: runs/eval_*.log (success rates), the probe json(s), $STABLEWM_HOME/checkpoints/pusht_fs_*_s0/{metrics.jsonl,
replay_index.npz}, and the expert h5's state column for the data-statistics reference row.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

import numpy as np

LEWM = Path(__file__).resolve().parent
ARENA = 512.0


def success(log):
    txt = Path(log).read_text(errors="ignore").replace("\r", "\n")
    m = re.findall(r"'success_rate': ([0-9.]+)", txt)
    return float(m[-1]) if m else None


def data_stats(states, ep_rows, ep_starts, contacts=None):
    """states (N, >=7): agent xy, block xy, angle ... ; one row per env step, episodes contiguous."""
    ax, ay, bx, by, th = states[:, 0], states[:, 1], states[:, 2], states[:, 3], states[:, 4]
    g = 16
    cov = lambda x, y: len(np.unique(np.clip((x / ARENA * g).astype(int), 0, g - 1) * g
                                     + np.clip((y / ARENA * g).astype(int), 0, g - 1))) / g ** 2
    ang_cov = len(np.unique(np.clip((np.mod(th, 2 * np.pi) / (2 * np.pi) * 12).astype(int), 0, 11))) / 12  # mod can return 2pi
    near_wall = np.minimum.reduce([bx, by, ARENA - bx, ARENA - by]) < 60
    disp, moved, a_moved, b_path, b_rot, steps = [], [], [], 0.0, 0.0, 0
    for s, n in zip(ep_starts, ep_rows):
        b = states[s:s + n, 2:4]
        if len(b) < 2:
            continue
        db = np.linalg.norm(np.diff(b, axis=0), axis=1)
        disp.append(np.linalg.norm(b[-1] - b[0]))
        moved.append((db > 0.5).mean())
        a_moved.append((np.linalg.norm(np.diff(states[s:s + n, 0:2], axis=0), axis=1) > 0.5).mean())
        b_path += db.sum()
        b_rot += np.abs(np.angle(np.exp(1j * np.diff(states[s:s + n, 4])))).sum()  # wrapped per-step rotation
        steps += n - 1
    out = {"agent_cov": cov(ax, ay), "block_cov": cov(bx, by), "angle_cov": ang_cov,
           "agent_xy_std": float(np.mean([ax.std(), ay.std()])), "agent_moving_frac": float(np.mean(a_moved)),
           "block_disp_per_ep_px": float(np.mean(disp)), "block_moving_frac": float(np.mean(moved)),
           "block_path_per100_px": float(100 * b_path / steps), "block_rot_per100_deg": float(100 * np.degrees(b_rot) / steps),
           "block_near_wall_frac": float(near_wall.mean()), "n_rows": int(len(states))}
    if contacts is not None:
        out["contact_frac"] = float((contacts > 0).mean())
    return out


def hist2(x, y, g=32):
    """Occupancy of the 512 x 512 arena on a g x g grid (fractions; rows = y)."""
    h, _, _ = np.histogram2d(y, x, bins=g, range=[[0, ARENA], [0, ARENA]])
    return (h / max(h.sum(), 1)).round(6).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="append", default=[])
    ap.add_argument("--out", default="runs/pusht_fs_report.md")
    ap.add_argument("--json", default=None, help="also write every number (+ position histograms) as one json")
    a = ap.parse_args()
    summary = {"planning": {}, "probes": {}, "data": {}, "hist": {}, "behaviour": {}}
    home = Path(os.environ.get("STABLEWM_HOME", LEWM / "swm_home"))
    lines = ["# Push-T: offline LeWM vs from-scratch learning-signal arms", ""]

    # planning success: runs/eval_<model>_s<seed>.log (expert pairs) and runs/evalrand_<model>_s<seed>.log (random set)
    sr = {}  # model -> set -> {eval seed: success %}
    for log in glob.glob(str(LEWM / "runs" / "eval*_s[0-9]*.log")):
        m = re.match(r"(eval|evalrand)_(.+)_s(\d+)$", Path(log).stem)
        r = success(log)
        if m is None or r is None or "keyerror" in log or "killed" in log:
            continue
        model = re.sub(r"^(pusht_)?fs_(\w+?)(_s0)?_step", r"fs_\2_step", m.group(2))
        sr.setdefault(model, {}).setdefault(m.group(1), {})[int(m.group(3))] = r
    summary["planning"] = {mo: {s: {str(k): v for k, v in d.items()} for s, d in x.items()} for mo, x in sr.items()}
    cell = lambda d: (f"{np.mean(list(d.values())):.1f} ({', '.join(f'{v:.0f}' for _, v in sorted(d.items()))}; "
                      f"{50 * len(d)} eps)") if d else "-"
    lines += ["## Planning success % (CEM 300x30 / top 30, horizon 5, budget 50, goal +25 steps; 50 episodes per eval seed)", "",
              "Expert set = the authors' eval (start/goal pairs from pusht_expert_train); random set = pusht_random_eval "
              "(random-action trajectories). Cell: mean (per-seed values; episodes).", "",
              "| model | expert set | random set |", "|---|---|---|"]
    for model in sorted(sr):
        lines.append(f"| {model} | {cell(sr[model].get('eval', {}))} | {cell(sr[model].get('evalrand', {}))} |")
    lines.append("")

    # probes
    probes = {}
    for p in a.probe:
        probes.update(json.loads(Path(p).read_text()))
    summary["probes"] = probes
    if probes:
        lines += ["## Frozen-latent probes (R^2; error in px / deg)", "",
                  "| model | set | probe | agent R2 | T pos R2 | T angle R2 | agent px | T px | T deg | eff. rank |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for k, r in probes.items():
            for st in ("expert", "uniform"):
                for pr in ("linear", "mlp"):
                    m = r[f"{st}_{pr}"]
                    lines.append(f"| {k} | {st} | {pr} | {m['agent_r2']:.3f} | {m['block_r2']:.3f} | {m['angle_r2']:.3f} | "
                                 f"{m['agent_err_px']:.1f} | {m['block_err_px']:.1f} | {m['angle_err_deg']:.1f} | "
                                 f"{r['latent']['eff_rank']:.1f} |")
        lines += ["", "## Open-loop prediction on held-out expert windows (linear expert probe; T position error px, k = 1..5 blocks)", "",
                  "| model | predicted | true latent (probe floor) | copy last context |", "|---|---|---|---|"]
        for k, r in probes.items():
            ol = r["openloop_linear_probe"]
            f = lambda tag: " / ".join(f"{e['block_err_px']:.1f}" for e in ol[tag])
            lines.append(f"| {k} | {f('pred')} | {f('true')} | {f('copy')} |")
        lines.append("")

    # data each arm collected
    lines += ["## Data collected (from-scratch arms) vs the expert dataset", "",
              "| data | rows | agent cov | agent xy std px | agent moving | T pos cov | T angle cov | T disp / ep px | "
              "T path / 100 steps px | T rot / 100 steps deg | T moving | T near wall | contact |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    row = lambda name, d: (f"| {name} | {d['n_rows']} | {d['agent_cov']:.2f} | {d['agent_xy_std']:.0f} | "
                           f"{d['agent_moving_frac']:.3f} | {d['block_cov']:.2f} | {d['angle_cov']:.2f} | "
                           f"{d['block_disp_per_ep_px']:.1f} | {d['block_path_per100_px']:.1f} | "
                           f"{d['block_rot_per100_deg']:.1f} | {d['block_moving_frac']:.3f} | "
                           f"{d['block_near_wall_frac']:.3f} | {d.get('contact_frac', float('nan')):.3f} |")
    try:
        import h5py
        import hdf5plugin  # noqa: F401
        with h5py.File(home / "datasets" / "pusht_expert_train.h5", "r") as f:
            st = f["state"][:]
            off, ln = f["ep_offset"][:], f["ep_len"][:]
        summary["data"]["expert"] = data_stats(st, ln, off)
        summary["hist"]["expert"] = {"agent": hist2(st[:, 0], st[:, 1]), "block": hist2(st[:, 2], st[:, 3])}
        lines.append(row("expert (pusht_expert_train)", summary["data"]["expert"]))
    except Exception as e:  # noqa: BLE001
        lines.append(f"| expert | (unavailable: {e}) |")
    for run in sorted(glob.glob(str(home / "checkpoints" / "pusht_fs_*_s*"))):
        if "killed" in run or not os.path.exists(f"{run}/replay_index.npz"):
            continue
        z = np.load(f"{run}/replay_index.npz")
        s10 = z["states"]
        ep_rows, rpe = z["ep_rows"], int(z["rows_per_ep"])
        starts = np.arange(len(ep_rows)) * rpe
        keep = np.concatenate([np.arange(s, s + n) for s, n in zip(starts, ep_rows)])
        d = data_stats(s10[keep][:, [0, 1, 4, 5, 6]],  # agent xy, block xy, angle
                       ep_rows, np.concatenate([[0], np.cumsum(ep_rows)[:-1]]), z["contacts"][keep])
        summary["data"][Path(run).name] = d
        summary["hist"][Path(run).name] = {"agent": hist2(s10[keep, 0], s10[keep, 1]), "block": hist2(s10[keep, 4], s10[keep, 5])}
        lines.append(row(Path(run).name, d))
    lines.append("")

    # behaviour over each arm's run (metrics.jsonl, one row per round; warm-up rounds excluded), by quarter of 500 rounds
    lines += ["## Behaviour over the run (per-round means by quarter of the 500 rounds; warm-up excluded)", "",
              "contact = fraction of steps with agent-T contact; T moved = px per 25-step plan; CEM gain = chosen signal / "
              "mean signal of the first CEM iteration's samples (1 = no better than a random candidate).", "",
              "| arm | rounds | contact | T moved px | T rot deg | CEM gain | WM pred loss |", "|---|---|---|---|---|---|---|"]
    for run in sorted(glob.glob(str(home / "checkpoints" / "pusht_fs_*_s*"))):
        if "killed" in run or not os.path.exists(f"{run}/metrics.jsonl"):
            continue
        ms = [json.loads(x) for x in Path(f"{run}/metrics.jsonl").read_text().splitlines() if x.strip()]
        ms = [m for m in ms if not m.get("warm")]
        win = [ms[i:i + 10] for i in range(0, len(ms), 10)]  # 10-round means for the page's run curves
        summary["behaviour"][Path(run).name] = {
            k: [float(np.nanmean([m.get(src, np.nan) for m in w])) for w in win]
            for k, src in (("round", "round"), ("contact", "beh/contact_frac"), ("blk_px", "beh/blk_disp_px"),
                           ("gain", "sig/gain_vs_it0"), ("pred_loss", "wm/pred_loss"))}
        for q in range(4):
            qs = [m for m in ms if q * 125 <= m["round"] < (q + 1) * 125]
            if not qs:
                continue
            avg = lambda k: np.nanmean([m.get(k, np.nan) for m in qs])
            lines.append(f"| {Path(run).name} | {qs[0]['round']}-{qs[-1]['round']} | {avg('beh/contact_frac'):.3f} | "
                         f"{avg('beh/blk_disp_px'):.1f} | {np.degrees(avg('beh/blk_rot_rad')):.1f} | "
                         f"{avg('sig/gain_vs_it0'):.2f} | {avg('wm/pred_loss'):.4f} |")
    lines.append("")
    Path(a.out).write_text("\n".join(lines))
    if a.json:
        clean = lambda o: ({k: clean(v) for k, v in o.items()} if isinstance(o, dict) else
                           [clean(v) for v in o] if isinstance(o, (list, tuple)) else
                           None if isinstance(o, float) and not np.isfinite(o) else o)  # NaN is not JSON
        Path(a.json).write_text(json.dumps(clean(summary), default=float))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
