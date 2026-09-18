"""Compare LeWM runs that differ only in the SIGReg weight lambda -- through the decoder's eye.

For each run (lambda read from its saved config.yaml; weights = latest epoch or --epoch), on the
SAME held-out frames:
  compare_recon.png   row 0 = real frames; one row per lambda = decode(encode(frame)) with that run's
                      own post-hoc decoder (lewm/train_decoder.py) -- what each latent keeps
  compare_rollout.png per lambda: 3 real context frames, then T open-loop imagined steps under the
                      dataset's actions, decoded (LeWM Fig. 7); real frames on top
  metrics.png / metrics.json   vs lambda: decoder val MSE, latent effective rank / RankMe / z_std,
                      one-step latent jump vs random-pair distance (locality), predictor one-step
                      MSE and its ratio to the persistence baseline, SIGReg statistic of the latent

    STABLEWM_HOME=lewm/swm_home python lewm/compare_sigreg.py --runs cube_lam0p0 cube_lam0p01 cube_lam0p09 \
        cube_lam0p5 cube_lam2p0 --out lewm/runs/sigreg_sweep
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

LEWM_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(LEWM_DIR))
sys.path.insert(0, str(LEWM_DIR.parent))

from model.decoder import load_decoder                                              # noqa: E402
from module import SIGReg                                                           # noqa: E402
from train_decoder import (ckpt_dir, column_stats, dataset_path, encode_frames, find_weights,    # noqa: E402
                           latent_stats, load_lewm, preprocess, read_col_rows, recalibrate_bn, resize_u8, touch_key)

TILE = 224


def _font(size=13):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


FONT = _font()


def label_row(tiles, text, width_label=190):
    row = np.concatenate([t if t.shape[0] == TILE else np.asarray(Image.fromarray(t).resize((TILE, TILE))) for t in tiles], 1)
    lab = Image.new("RGB", (width_label, TILE), (0, 0, 0))
    d = ImageDraw.Draw(lab)
    for i, line in enumerate(text.split("\n")):
        d.text((4, 4 + 16 * i), line, fill=(255, 255, 255), font=FONT)
    return np.concatenate([np.asarray(lab), row], 1)


def run_params(run: str) -> tuple[float, float]:
    """(SIGReg lambda, touch_scale) of a run from its saved full config (touch_scale 0 = plain LeWM)."""
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(ckpt_dir(run) / "config.yaml")
    return float(cfg.loss.sigreg.weight), float(cfg.model.get("touch_scale", 0.0))


def read_rows(h5, rows):
    return h5["pixels"][rows] if isinstance(rows, slice) else np.stack([h5["pixels"][int(r)] for r in rows])


@torch.no_grad()
def rollout(model, ctx_u8, actions_norm, T, device, H=3, img_size=224, touch=None):
    """ctx_u8 (H,h,w,3) real context frames; actions_norm (H+T, frameskip*a_dim) normalized action blocks.
    -> imagined latents (T, D): JEPA.rollout arithmetic (predict on the last H latents + action embs)."""
    z = encode_frames(model, ctx_u8, device, img_size=img_size, touch=touch).to(device).unsqueeze(0)     # (1, H, D)
    act = torch.as_tensor(actions_norm, device=device).float().unsqueeze(0)                # (1, H+T, A)
    preds = []
    for t in range(T):
        a_emb = model.action_encoder(act[:, t:t + H])
        pred = model.predict(z[:, -H:], a_emb)[:, -1:]
        preds.append(pred[0, 0])
        z = torch.cat([z, pred], 1)
    return torch.stack(preds)


def main(a):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    h5_path = dataset_path(a.dataset)
    sig = SIGReg(knots=17, num_proj=1024).to(device)

    with h5py.File(h5_path, "r") as f:
        ep_len, ep_off = f["ep_len"][:].astype(int), f["ep_offset"][:].astype(int)
        act_all = f["action"][:].astype(np.float32)
        a_dim = act_all.shape[-1]
        n_rows = f["pixels"].shape[0]
        # action z-score exactly as train.py's get_column_normalizer (rows with NaN dropped, unbiased std)
        ok = ~np.isnan(act_all).any(1)
        a_mean, a_std = act_all[ok].mean(0), act_all[ok].std(0, ddof=1)
        # (1) recon frames: the decoder's val blocks of the FIRST run (identical across runs by seed)
        meta0 = json.loads((ckpt_dir(a.runs[0]) / f"decoder_epoch_{find_weights(a.runs[0], a.epoch)[1]}.json").read_text())
        val_starts, block = meta0["val_starts"], meta0["block"]
        pick_blocks = rng.choice(len(val_starts), size=min(a.n_recon, len(val_starts)), replace=False)
        recon_rows = sorted(int(val_starts[b]) + int(rng.integers(0, block)) for b in pick_blocks)
        recon_u8 = resize_u8(read_rows(f, recon_rows))
        # (2) locality / predictor windows: (ep, start) with H+T+1 frames at the training frameskip
        fs, H, T = a.frameskip, a.history, a.horizon
        span = (H + T) * fs + 1
        eps_ok = np.where(ep_len >= span)[0]
        wins = []
        for _ in range(a.n_windows):
            e = int(rng.choice(eps_ok)); s = int(rng.integers(0, ep_len[e] - span + 1))
            wins.append((e, s))
        wins.sort()
        win_frames, win_actions = [], []
        for e, s in wins:
            g = ep_off[e] + s
            win_frames.append(f["pixels"][g:g + H * fs + fs + 1:fs][:H + 1])         # H context + 1 target frame
            raw = act_all[g:g + (H + 1) * fs].reshape(H + 1, fs * a_dim)               # action blocks per step
            win_actions.append(raw)
        win_frames = resize_u8(np.stack(win_frames).reshape(-1, *f["pixels"].shape[1:])).reshape(len(wins), H + 1, TILE, TILE, 3)
        win_actions = (np.stack(win_actions) - np.tile(a_mean, fs)) / np.tile(a_std, fs)
        # (3) one rollout window with T future frames
        e, s = wins[a.rollout_idx % len(wins)]
        g = ep_off[e] + s
        roll_frames = resize_u8(f["pixels"][g:g + (H + T) * fs + 1:fs][:H + T])
        roll_actions = (act_all[g:g + (H + T) * fs].reshape(H + T, fs * a_dim) - np.tile(a_mean, fs)) / np.tile(a_std, fs)
        touch_cols = {}
        for run in a.runs:
            cfg_json = json.loads((ckpt_dir(run) / "config.json").read_text())
            tk = cfg_json.get("touch_key")
            if tk and tk not in touch_cols:
                st = column_stats(f, tk)
                touch_cols[tk] = dict(
                    recon=read_col_rows(f, tk, [[int(r)] for r in recon_rows], st),
                    win=read_col_rows(f, tk, [slice(ep_off[e] + s, ep_off[e] + s + H * fs + 1, fs) for e, s in wins], st),
                    roll=read_col_rows(f, tk, [slice(g, g + H * fs + 1, fs)], st)[:H])
    print(f"[compare] {h5_path}: {n_rows} rows; recon rows {len(recon_rows)}, {len(wins)} windows (H={H}, T={T}, "
          f"frameskip {fs}), rollout ep {e} start {s}", flush=True)

    rows_recon = [label_row(list(recon_u8), "real")]
    rows_roll = [label_row(list(roll_frames), f"real\nctx {H} | future {T}")]
    metrics = []
    for run in a.runs:
        lam, alpha = run_params(run)
        model, wpath, epoch = load_lewm(run, a.epoch, device)
        if not a.no_bn_recal:
            recalibrate_bn(model, h5_path, device, n_batches=a.bn_batches, seed=a.seed)
        dec_path = ckpt_dir(run) / f"decoder_epoch_{epoch}.pt"
        dec, dmeta = load_decoder(dec_path, device)
        djson = json.loads(dec_path.with_suffix(".json").read_text())
        tk = touch_key(model)
        tc = touch_cols.get(tk, {}) if tk else {}
        with torch.no_grad():
            z_recon = encode_frames(model, recon_u8, device, touch=tc.get("recon")).to(device)
            rec = dec.to_uint8_hwc(z_recon)
            # latents of the window frames: (N, H+1, D)
            flat = win_frames.reshape(-1, TILE, TILE, 3)
            zw = encode_frames(model, flat, device, touch=tc.get("win")).to(device).reshape(len(wins), H + 1, -1)
            z_t, z_next = zw[:, H - 1], zw[:, H]
            jump = (z_next - z_t).norm(dim=-1)
            perm = torch.randperm(len(wins), device=device)
            rand_d = (z_t - z_t[perm]).norm(dim=-1)
            act = torch.as_tensor(win_actions, device=device).float()                     # (N, H+1, A)
            a_emb = model.action_encoder(act[:, :H])
            pred = model.predict(zw[:, :H], a_emb)[:, -1]                                  # z_hat_{H}
            pred_mse = float((pred - z_next).pow(2).mean())
            persist_mse = float((z_t - z_next).pow(2).mean())
            sig_stat = float(sig(zw[:, :1].transpose(0, 1)))                              # (1, N, D) per-timestep view
            st = latent_stats(zw[:, 0].cpu())
            z_roll = rollout(model, roll_frames[:H], roll_actions, T, device, H, touch=tc.get("roll"))
            dec_roll = dec.to_uint8_hwc(torch.cat([encode_frames(model, roll_frames[:H], device, touch=tc.get("roll")).to(device), z_roll]))
        m = dict(run=run, lam=lam, touch_scale=alpha, x=(alpha if a.x == "touch_scale" else lam), epoch=epoch,
                 decoder_val_mse=float(dmeta.get("val_mse", float("nan"))),
                 cube_mse=float(djson.get("cube_mse", float("nan"))), bg_mse=float(djson.get("bg_mse", float("nan"))),
                 cube_frames=int(djson.get("cube_frames", 0)),
                 probe_r2_block=float(djson.get("probe_r2", {}).get("block", float("nan"))),
                 probe_r2_effector=float(djson.get("probe_r2", {}).get("effector", float("nan"))),
                 probe_r2_block_far=float(djson.get("probe_r2", {}).get("block_far", float("nan"))),
                 n_far=int(djson.get("probe_r2", {}).get("n_far", 0)),
                 jump=float(jump.mean()), rand_pair=float(rand_d.mean()), frac_rand=float(jump.mean() / rand_d.mean().clamp_min(1e-9)),
                 pred_mse=pred_mse, persist_mse=persist_mse, pred_over_persist=pred_mse / max(persist_mse, 1e-12),
                 sigreg_stat=sig_stat, **st)
        metrics.append(m)
        tag = f"lambda {lam:g}" + (f" touch x{alpha:g}" if alpha else "")
        rows_recon.append(label_row(list(rec), f"{tag}\nval mse {m['decoder_val_mse']:.4f}\ncube mse {m['cube_mse']:.4f}\nprobe R2 cube {m['probe_r2_block']:.2f} (far {m['probe_r2_block_far']:.2f})\neff rank {m['eff_rank']:.0f}/{m['dim']}"))
        rows_roll.append(label_row(list(dec_roll), f"{tag}\ndecode(z ctx | imagined)\npred/persist {m['pred_over_persist']:.2f}"))
        print(f"[compare] {run:16s} lam {lam:<5g} touch {alpha:<4g} ep {epoch}: dec val mse {m['decoder_val_mse']:.4f} cube {m['cube_mse']:.4f} "
              f"probeR2 cube {m['probe_r2_block']:.3f} far {m['probe_r2_block_far']:.3f} (n {m['n_far']}) eff {m['probe_r2_effector']:.3f}  z_std {m['z_std']:.3f}  "
              f"eff_rank {m['eff_rank']:.1f}  rankme {m['rankme']:.1f}  jump {m['jump']:.2f}  rand {m['rand_pair']:.2f}  "
              f"frac_rand {m['frac_rand']:.3f}  pred {pred_mse:.4f}  persist {persist_mse:.4f}  ratio {m['pred_over_persist']:.2f}  "
              f"sigreg {sig_stat:.2f}", flush=True)
        del model, dec; torch.cuda.empty_cache()

    def stack(rows):
        return np.concatenate([np.concatenate([r, np.full((6, r.shape[1], 3), 80, np.uint8)], 0) for r in rows], 0)
    Image.fromarray(stack(rows_recon)).save(out / "compare_recon.png")
    Image.fromarray(stack(rows_roll)).save(out / "compare_rollout.png")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xv = np.array([m["x"] for m in metrics])
    pos = xv[xv > 0]
    x0 = (pos.min() / 10) if len(pos) else 1e-3
    xs = np.where(xv > 0, xv, x0)
    xlabel = ("touch_scale (0 = pixels-only LeWM, plotted at left)" if a.x == "touch_scale"
              else "SIGReg lambda (0 plotted at left)")
    keys = [("decoder_val_mse", "decoder val MSE (pixels, [0,1])"), ("cube_mse", "decoder MSE inside the cube's pixels"),
            ("probe_r2_block", "linear probe R^2: cube position (all frames)"), ("probe_r2_block_far", "linear probe R^2: cube position, cube NOT in hand"),
            ("probe_r2_effector", "linear probe R^2: effector position"), ("eff_rank", "latent effective rank"),
            ("frac_rand", "1-step jump / random-pair distance"), ("pred_over_persist", "predictor MSE / persistence MSE"),
            ("sigreg_stat", "SIGReg statistic (per-frame latents)")]
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    for ax, (k, title) in zip(axes.ravel(), keys):
        ax.plot(xs, [m[k] for m in metrics], "o-", color="#1f77b4")
        for xi, m in zip(xs, metrics):
            ax.annotate(f"{m['x']:g}", (xi, m[k]), textcoords="offset points", xytext=(4, 4), fontsize=8)
        ax.set_xscale("log"); ax.set_title(title, fontsize=10); ax.set_xlabel(xlabel, fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(f"LeWM Cube sweep over {a.x}, epoch {metrics[0]['epoch']} (BatchNorm-recalibrated eval)", fontsize=11)
    fig.tight_layout(); fig.savefig(out / "metrics.png", dpi=120); plt.close(fig)
    print(f"[compare] -> {out}/compare_recon.png, compare_rollout.png, metrics.png, metrics.json", flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--epoch", type=int, default=None)
    p.add_argument("--dataset", default="ogbench/cube_single_expert")
    p.add_argument("--out", default="runs/sigreg_sweep")
    p.add_argument("--n-recon", type=int, default=10, help="held-out frames in the recon sheet")
    p.add_argument("--n-windows", type=int, default=512, help="windows for the latent / predictor statistics")
    p.add_argument("--frameskip", type=int, default=5)
    p.add_argument("--history", type=int, default=3)
    p.add_argument("--horizon", type=int, default=8, help="imagined steps in the rollout sheet")
    p.add_argument("--rollout-idx", type=int, default=0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--no-bn-recal", action="store_true", help="use the checkpoints' BatchNorm running stats as saved")
    p.add_argument("--bn-batches", type=int, default=30)
    p.add_argument("--x", choices=("lambda", "touch_scale"), default="lambda", help="the swept variable (x axis / labels)")
    return p.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
