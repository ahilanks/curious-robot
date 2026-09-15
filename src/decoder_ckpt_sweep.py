"""Decodability over training: fit the post-hoc pixel decoder to SEVERAL checkpoints of one run
on the SAME frames and compare what each encoder's latent keeps (LeWM App. D decoder, diagnostic
only -- no WM gradient anywhere).

Per checkpoint: encode the frames with that ckpt's frozen encoder, fit a fresh decoder, then on the
val frames report pixel MSE, MSE restricted to the block's pixels (purple mask on the REAL frame),
and the fraction of block-visible frames whose DECODE shows the block. One sheet (rows = ckpts,
columns = the val frames with the most block pixels), one plot, one JSON.

    python src/decoder_ckpt_sweep.py --run wr_sleepret2 --steps 1000,10000,30000,100000,200000
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.decoder import LatentDecoder                                           # noqa: E402
from src.train_decoder import (load_wm, load_frames, encode_all, frames_to_target,  # noqa: E402
                               resolve_ckpt_spec, resolve_state)
from src.viz_goal_archive import block_mask                                       # noqa: E402


def fit_decoder(z_all, frames, train_idx, device, steps, batch, hidden, depth, heads, lr=3e-4, wd=0.05,
                seed=0, log_every=500, tag=""):
    """Same recipe as train_decoder.py (AdamW, warmup+cosine, grad-clip 1) on cached latents."""
    torch.manual_seed(seed); np.random.seed(seed)
    dec = LatentDecoder(z_dim=z_all.shape[1], hidden=hidden, depth=depth, heads=heads).to(device)
    opt = torch.optim.AdamW(dec.parameters(), lr=lr, weight_decay=wd)
    warm = min(100, max(1, steps // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    dec.train(); t0 = time.time()
    for step in range(1, steps + 1):
        idx = train_idx[np.random.randint(0, len(train_idx), size=batch)]
        loss = torch.nn.functional.mse_loss(dec(z_all[idx]), frames_to_target(frames[idx], device))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)
        opt.step(); sched.step()
        if step % log_every == 0 or step == steps:
            print(f"[sweep{tag}] step {step}/{steps} mse={loss.item():.5f} ({(time.time() - t0) / step * 1000:.0f} ms/step)",
                  flush=True)
    return dec.eval().requires_grad_(False)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="run name (HF folder), e.g. wr_sleepret2")
    p.add_argument("--steps", required=True, help="comma-separated checkpoint steps")
    p.add_argument("--state", default="", help="state_latest.npz (default runs/<run>/state_latest.npz, HF fallback)")
    p.add_argument("--out-dir", default="", help="default runs/<run>/sweep")
    p.add_argument("--fit-steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--max-frames", type=int, default=20000)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--min-block-px", type=int, default=60)
    p.add_argument("--decode-margin", type=int, default=40, help="looser purple margin for the (smooth) decodes")
    p.add_argument("--sheet-cols", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hf-repo", default=None)
    p.add_argument("--device", default=None)
    a = p.parse_args()
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(a.out_dir or f"runs/{a.run}/sweep"); out_dir.mkdir(parents=True, exist_ok=True)
    steps = [int(s) for s in a.steps.split(",")]

    state_path = resolve_state(a.state or f"runs/{a.run}/state_latest.npz", a.hf_repo)
    frames, props = load_frames(state_path, a.max_frames)
    n = len(frames); n_val = max(8, int(a.val_frac * n))
    val_idx = np.linspace(0, n - 1, n_val).astype(int)
    train_idx = np.setdiff1d(np.arange(n), val_idx)
    real_val = frames[val_idx]
    masks = np.stack([block_mask(f) for f in real_val])                     # (V,H,W) block pixels, real
    area = masks.reshape(len(val_idx), -1).sum(1)
    vis = np.where(area >= a.min_block_px)[0]
    print(f"[sweep] {len(train_idx)} train / {len(val_idx)} val frames from {state_path}; "
          f"block visible in {len(vis)}/{len(val_idx)} val frames", flush=True)
    sheet_pick = np.argsort(-area)[:a.sheet_cols]                            # most block pixels first

    results, sheet_rows = [], [("real", real_val[sheet_pick])]
    Yv = frames_to_target(real_val, device)
    for step in steps:
        ckpt_path = resolve_ckpt_spec(f"runs/{a.run}/ckpt_{step:07d}.pt", a.hf_repo)
        wm = load_wm(ckpt_path, device)
        t0 = time.time()
        z_all = encode_all(wm, frames, props, device, log_every=10 ** 9).to(device)
        print(f"[sweep@{step}] encoded {n} frames in {time.time() - t0:.0f}s", flush=True)
        dec = fit_decoder(z_all, frames, train_idx, device, a.fit_steps, a.batch, a.hidden, a.depth, a.heads,
                          seed=a.seed, tag=f"@{step}")
        with torch.no_grad():
            rec = torch.cat([dec(z_all[val_idx[i:i + a.batch]]) for i in range(0, len(val_idx), a.batch)])
            val_mse = float(torch.nn.functional.mse_loss(rec, Yv))
            m = torch.as_tensor(masks, device=device).unsqueeze(1).expand_as(rec)
            blk_mse = float(((rec - Yv) ** 2)[m].mean()) if int(m.sum()) else float("nan")
            bg_mse = float(((rec - Yv) ** 2)[~m].mean())
        rec_u8 = (rec.permute(0, 2, 3, 1).clamp(0, 1) * 255).byte().cpu().numpy()
        dec_area = np.array([block_mask(r, margin=a.decode_margin).sum() for r in rec_u8])
        hit = float((dec_area[vis] >= a.min_block_px).mean()) if len(vis) else float("nan")
        row = dict(step=step, ckpt=str(ckpt_path), val_mse=val_mse, block_mse=blk_mse, background_mse=bg_mse,
                   block_detect_rate=hit, n_block_frames=int(len(vis)), n_val=int(len(val_idx)),
                   decoder=str(out_dir / f"decoder_{step:07d}.pt"))
        results.append(row)
        torch.save({"decoder": dec.state_dict(), "z_dim": wm.z_dim, "arch": dec.config(), "val_mse": val_mse,
                    "steps": a.fit_steps, "n_frames": int(n), "ckpt": str(ckpt_path), "state": str(state_path)},
                   row["decoder"])
        sheet_rows.append((f"ckpt {step}  val {val_mse:.4f}  blk {blk_mse:.4f}  det {hit:.2f}", rec_u8[sheet_pick]))
        print(f"[sweep@{step}] val mse {val_mse:.5f} | block-pixel mse {blk_mse:.5f} (bg {bg_mse:.5f}) | "
              f"block decoded in {hit * 100:.0f}% of {len(vis)} block frames", flush=True)
        del wm, z_all, dec, rec
        if device == "cuda":
            torch.cuda.empty_cache()

    # ---- sheet (rows: real then one per ckpt), plot, json
    from PIL import Image, ImageDraw, ImageFont
    T, LAB = frames.shape[1], 18
    ncol = len(sheet_pick)
    sheet = Image.new("RGB", (ncol * T, len(sheet_rows) * (T + LAB)), (24, 24, 24))
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default(size=12)
    except TypeError:
        font = ImageFont.load_default()
    for r, (label, imgs) in enumerate(sheet_rows):
        y = r * (T + LAB)
        d.text((4, y + 3), label, fill=(235, 235, 235), font=font)
        for c, img in enumerate(imgs):
            sheet.paste(Image.fromarray(np.ascontiguousarray(img)), (c * T, y + LAB))
    sheet.save(out_dir / "sweep_sheet.png")
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [r["step"] for r in results]
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        ax[0].plot(xs, [r["val_mse"] for r in results], "o-", label="all pixels")
        ax[0].plot(xs, [r["block_mse"] for r in results], "s-", label="block pixels")
        ax[0].plot(xs, [r["background_mse"] for r in results], "^-", label="non-block pixels")
        ax[0].set_xscale("log"); ax[0].set_xlabel("checkpoint step"); ax[0].set_ylabel("val MSE (decode vs real)"); ax[0].legend()
        ax[1].plot(xs, [100 * r["block_detect_rate"] for r in results], "o-")
        ax[1].set_xscale("log"); ax[1].set_xlabel("checkpoint step"); ax[1].set_ylabel("% of block frames: block visible in decode")
        ax[1].set_ylim(0, 105)
        fig.suptitle(f"{a.run}: what the latent keeps, per checkpoint (LeWM decoder, {a.fit_steps} steps each)")
        fig.tight_layout(); fig.savefig(out_dir / "sweep_plot.png", dpi=110)
    except Exception as e:                                                   # plotting is optional
        print(f"[sweep] plot skipped: {e}", flush=True)
    (out_dir / "sweep.json").write_text(json.dumps(dict(run=a.run, state=str(state_path), fit_steps=a.fit_steps,
                                                        results=results), indent=1))
    print("\n[sweep] step      val_mse   block_mse   bg_mse   block_detect", flush=True)
    for r in results:
        print(f"[sweep] {r['step']:>7d}   {r['val_mse']:.5f}   {r['block_mse']:.5f}   {r['background_mse']:.5f}   "
              f"{r['block_detect_rate'] * 100:5.0f}%", flush=True)
    print(f"[sweep] -> {out_dir}/sweep_sheet.png  sweep_plot.png  sweep.json", flush=True)


if __name__ == "__main__":
    main()
