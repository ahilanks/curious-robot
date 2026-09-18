"""Overlay compare_sigreg.py metrics from several checkpoints (e.g. epoch 1 vs epoch 8) vs lambda.

    python lewm/plot_sweep_epochs.py runs/sigreg_sweep_ep1/metrics.json runs/sigreg_sweep/metrics.json --out runs/sigreg_sweep/metrics_epochs.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

KEYS = [("decoder_val_mse", "decoder val MSE (pixels, [0,1])"), ("eff_rank", "latent effective rank"),
        ("frac_rand", "1-step jump / random-pair distance"), ("pred_over_persist", "predictor MSE / persistence MSE"),
        ("pred_mse", "predictor one-step MSE (latent)"), ("sigreg_stat", "SIGReg statistic (per-frame latents, B=512)")]


def main(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    series = [(Path(p), json.loads(Path(p).read_text())) for p in a.metrics]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    colors = ["#9ecae1", "#3182bd", "#08519c", "#000000"]
    for ax, (k, title) in zip(axes.ravel(), KEYS):
        for i, (p, ms) in enumerate(series):
            lams = np.array([m["lam"] for m in ms])
            pos = lams[lams > 0]
            x0 = pos.min() / 10 if len(pos) else 1e-3
            xs = np.where(lams > 0, lams, x0)
            ax.plot(xs, [m[k] for m in ms], "o-", color=colors[i % len(colors)], label=f"epoch {ms[0]['epoch']}")
        ax.set_xscale("log"); ax.set_title(title, fontsize=10); ax.grid(alpha=0.3)
        ax.set_xlabel("SIGReg lambda (0 plotted at far left)")
        if k in ("decoder_val_mse", "pred_mse", "sigreg_stat"):
            ax.set_yscale("log")
        ax.legend(fontsize=8)
    fig.suptitle("LeWM Cube, SIGReg-lambda sweep (BatchNorm-recalibrated eval)", fontsize=11)
    fig.tight_layout(); fig.savefig(a.out, dpi=120); plt.close(fig)
    print(f"[plot] -> {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("metrics", nargs="+")
    p.add_argument("--out", required=True)
    main(p.parse_args())
