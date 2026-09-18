"""Post-hoc pixel decoder for a LeWM checkpoint -- LeWM App. D "Decoder (Visualization Only)".

Fits model/decoder.py's LatentDecoder (the App. D architecture: 196 patch queries cross-attending
to the latent) on (z, frame) pairs from an HDF5 dataset under a FROZEN LeWM encoder. z is the
latent the world model predicts and the planner scores: JEPA.encode -> projector output ("emb").
Nothing here touches the world model; the decoder is a diagnostic read-out of what z keeps.

    STABLEWM_HOME=lewm/swm_home python lewm/train_decoder.py --run cube_lam0p09 \
        --dataset ogbench/cube_single_expert [--epoch 3] [--steps 3000]

Frames come from random CONTIGUOUS blocks of the dataset (fast on chunked HDF5), sampled with a
fixed --seed so every run of a sweep is fitted and validated on the SAME frames; fit and val
blocks are disjoint. Frames are preprocessed exactly as the trainer does (uint8 -> [0,1] ->
ImageNet normalize -> Resize 224, antialias) before encoding; the decoder target is the resized
frame in [0,1].

Outputs (in $STABLEWM_HOME/checkpoints/<run>/):
    decoder_epoch_<E>.pt        decoder state + arch + val_mse (model/decoder.load_decoder loads it)
    decoder_epoch_<E>.sheet.png top = real val frames, bottom = decode(encode(frame))
    decoder_epoch_<E>.json      val_mse, latent stats on the val frames, frame indices used
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

LEWM_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LEWM_DIR.parent
sys.path.insert(0, str(LEWM_DIR))          # jepa / module (hydra _target_ paths in config.json)
sys.path.insert(0, str(PROJECT_ROOT))      # model.decoder

from model.decoder import LatentDecoder, load_decoder          # noqa: E402

try:
    import hdf5plugin  # noqa: F401  (registers the dataset's compression filters)
except ImportError:
    pass

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ----------------------------------------------------------------------------- paths / model
def swm_home() -> Path:
    return Path(os.environ.get("STABLEWM_HOME", str(Path.home() / ".stable_worldmodel")))


def ckpt_dir(run: str) -> Path:
    return swm_home() / "checkpoints" / run


def dataset_path(name: str) -> Path:
    p = Path(name)
    if not p.is_absolute():
        p = swm_home() / "datasets" / name
    if p.suffix != ".h5":
        p = p.with_suffix(".h5")
    return p


def find_weights(run: str, epoch: int | None) -> tuple[Path, int]:
    """weights_epoch_<E>.pt for the requested epoch (None = the latest one saved)."""
    d = ckpt_dir(run)
    if epoch is not None:
        p = d / f"weights_epoch_{epoch}.pt"
        if not p.exists():
            raise FileNotFoundError(p)
        return p, epoch
    cands = sorted(glob.glob(str(d / "weights_epoch_*.pt")), key=lambda s: int(Path(s).stem.split("_")[-1]))
    if not cands:
        raise FileNotFoundError(f"no weights_epoch_*.pt in {d}")
    return Path(cands[-1]), int(Path(cands[-1]).stem.split("_")[-1])


def load_lewm(run: str, epoch: int | None, device):
    """Rebuild the JEPA from the saved config.json (hydra instantiate, as train.py) + load weights."""
    import hydra
    from omegaconf import OmegaConf
    d = ckpt_dir(run)
    cfg = OmegaConf.create(json.loads((d / "config.json").read_text()))
    model = hydra.utils.instantiate(cfg)
    wpath, ep = find_weights(run, epoch)
    sd = torch.load(wpath, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[lewm] load_state_dict: missing {len(missing)} unexpected {len(unexpected)}", flush=True)
        if len(missing) > 10:
            raise RuntimeError(f"weights do not match the config: missing {missing[:5]} ...")
    model = model.to(device).eval().requires_grad_(False)
    return model, wpath, ep


@torch.no_grad()
def recalibrate_bn(model, h5_path, device, n_batches: int = 30, batch: int = 64, frameskip: int = 5,
                   history: int = 3, seed: int = 0, img_size: int = 224) -> int:
    """Refresh every BatchNorm's running statistics on dataset windows (cumulative average, in memory only).

    Why: the LeWM projector / pred_proj MLPs carry BatchNorm1d. The trainer computes its losses in train
    mode (batch statistics); a checkpoint's running statistics lag far behind early in training (2026-09-17,
    epoch-1 Cube checkpoints: eval-mode latent mean shifted ~1 per dim, predictor MSE 5-13x worse than with
    batch statistics, SIGReg statistic 30x). Everything downstream (decoder fit, latent stats, rollouts) is
    evaluated in eval mode, so the stats are re-estimated first; nothing is written back to the checkpoint.
    Returns the number of BatchNorm modules touched (0 = nothing to do)."""
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return 0
    for m in bns:
        m.reset_running_stats(); m.momentum = None; m.train()
    rng = np.random.default_rng(seed)
    H, fs = history, frameskip
    with h5py.File(h5_path, "r") as f:
        ep_len, ep_off = f["ep_len"][:].astype(int), f["ep_offset"][:].astype(int)
        act_all = f["action"][:].astype(np.float32)
        a_dim = act_all.shape[-1]
        ok = ~np.isnan(act_all).any(1)
        a_mean, a_std = act_all[ok].mean(0), act_all[ok].std(0, ddof=1)
        key = touch_key(model)
        tstats = column_stats(f, key) if key else None
        eps_ok = np.where(ep_len >= (H + 1) * fs)[0]
        for _ in range(n_batches):
            wins = sorted((int(rng.choice(eps_ok)), 0) for _ in range(batch))
            wins = [(e, int(rng.integers(0, ep_len[e] - (H + 1) * fs + 1))) for e, _ in wins]
            fr = np.stack([f["pixels"][ep_off[e] + s: ep_off[e] + s + H * fs + 1: fs][:H + 1] for e, s in wins])
            ac = np.stack([act_all[ep_off[e] + s: ep_off[e] + s + (H + 1) * fs].reshape(H + 1, fs * a_dim) for e, s in wins])
            ac = np.nan_to_num((ac - np.tile(a_mean, fs)) / np.tile(a_std, fs))
            x = preprocess(resize_u8(fr.reshape(-1, *fr.shape[2:]), img_size), device, img_size).reshape(batch, H + 1, 3, img_size, img_size)
            info = {"pixels": x, "action": torch.as_tensor(ac, device=device).float()}
            if key:
                tch = read_col_rows(f, key, [slice(ep_off[e] + s, ep_off[e] + s + H * fs + 1, fs) for e, s in wins], tstats)
                info[key] = torch.as_tensor(tch.reshape(batch, -1, tch.shape[-1])[:, :H + 1], device=device).float()
            out = model.encode(info)
            model.predict(out["emb"][:, :H], out["act_emb"][:, :H])      # pred_proj's BatchNorm sees predictions
    for m in bns:
        m.eval()
    return len(bns)


def preprocess(px_u8: np.ndarray, device, img_size: int = 224) -> torch.Tensor:
    """uint8 (B,H,W,3) -> normalized float (B,3,img,img): ToImage(ImageNet) + Resize(img_size, antialias)."""
    from torchvision.transforms.v2 import functional as TF
    x = torch.as_tensor(np.ascontiguousarray(px_u8), device=device).permute(0, 3, 1, 2).float() / 255.0
    x = (x - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)
    if x.shape[-1] != img_size or x.shape[-2] != img_size:
        x = TF.resize(x, [img_size, img_size], antialias=True)
    return x


def resize_u8(px_u8: np.ndarray, img_size: int = 224) -> np.ndarray:
    """The decoder's target / sheet frames: the same resize as the encoder input, kept as uint8."""
    if px_u8.shape[1] == img_size and px_u8.shape[2] == img_size:
        return px_u8
    from torchvision.transforms.v2 import functional as TF
    x = torch.as_tensor(np.ascontiguousarray(px_u8)).permute(0, 3, 1, 2).float() / 255.0
    x = TF.resize(x, [img_size, img_size], antialias=True)
    return (x.clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).numpy()


def touch_key(model):
    """The h5 column a JEPATouch model fuses into its encoder (None for plain LeWM)."""
    return getattr(model, "touch_key", None)


def column_stats(h5, key):
    """(mean, std) of an h5 column as the trainer's get_column_normalizer computes them (NaN rows dropped, unbiased std)."""
    col = h5[key][:].astype(np.float32).reshape(len(h5[key]), -1)
    ok = ~np.isnan(col).any(1)
    return col[ok].mean(0), col[ok].std(0, ddof=1)


def read_col_rows(h5, key, rows_or_slices, stats=None):
    """h5[key] for a list of slices / index arrays -> (N, k) float32, optionally z-scored with `stats`."""
    parts = [np.asarray(h5[key][r]).reshape(-1, np.prod(h5[key].shape[1:], dtype=int)) for r in rows_or_slices]
    x = np.concatenate(parts).astype(np.float32)
    if stats is not None:
        x = (x - stats[0]) / stats[1]
    return np.nan_to_num(x)


@torch.no_grad()
def encode_frames(model, px_u8: np.ndarray, device, batch: int = 128, img_size: int = 224,
                  touch: np.ndarray | None = None) -> torch.Tensor:
    """-> (N, D) latents = JEPA.encode(...)['emb'] (projector output) for single frames.
    `touch` (N, k), already z-scored, is required for a JEPATouch model (its encoder input)."""
    key = touch_key(model)
    if key is not None and touch is None:
        raise ValueError(f"model fuses '{key}' into its encoder: pass touch=")
    zs = []
    for i in range(0, len(px_u8), batch):
        x = preprocess(px_u8[i:i + batch], device, img_size).unsqueeze(1)          # (B, 1, 3, H, W)
        info = {"pixels": x}
        if key is not None:
            info[key] = torch.as_tensor(touch[i:i + batch], device=device).float().unsqueeze(1)   # (B, 1, k)
        out = model.encode(info)
        zs.append(out["emb"][:, 0].float().cpu())
    return torch.cat(zs)


# ----------------------------------------------------------------------------- data
def sample_blocks(n_rows: int, n_blocks: int, block: int, rng, exclude=()) -> np.ndarray:
    """Random contiguous row blocks (start indices), non-overlapping with each other and `exclude`."""
    starts, taken = [], list(exclude)
    tries = 0
    while len(starts) < n_blocks and tries < n_blocks * 50:
        tries += 1
        s = int(rng.integers(0, n_rows - block))
        if all(abs(s - t) >= block for t in taken):
            starts.append(s); taken.append(s)
    if len(starts) < n_blocks:
        raise SystemExit("could not place enough non-overlapping blocks")
    return np.array(sorted(starts))


def read_blocks(h5, starts: np.ndarray, block: int) -> np.ndarray:
    px = h5["pixels"]
    return np.concatenate([px[s:s + block] for s in starts])


def cube_mask(u8: np.ndarray, dilate: int = 2) -> np.ndarray:
    """(N,H,W,3) uint8 -> (N,H,W) bool: the red cube's pixels (R high, G/B low; the purple gripper has B high),
    dilated `dilate` px inside each frame so the decoder's blur is scored too."""
    from scipy.ndimage import binary_dilation
    r, g, b = (u8[..., c].astype(np.int16) for c in range(3))
    m = (r > 120) & (g < 90) & (b < 90) & (r - g > 60)
    if dilate:
        st = np.zeros((3, 3, 3), bool); st[1] = True
        m = binary_dilation(m, structure=st, iterations=dilate)
    return m


def masked_mse(dec_u8: np.ndarray, real_u8: np.ndarray, mask: np.ndarray, min_px: int = 30) -> tuple[float, float, int]:
    """-> (mse inside the mask, mse outside, frames used): pixel MSE in [0,1] units over frames with >= min_px masked px."""
    err = ((dec_u8.astype(np.float32) - real_u8.astype(np.float32)) / 255.0) ** 2   # (N,H,W,3)
    keep = mask.reshape(len(mask), -1).sum(1) >= min_px
    if keep.sum() == 0:
        return float("nan"), float(err.mean()), 0
    e, m = err[keep], mask[keep][..., None]
    return float(e[np.broadcast_to(m, e.shape)].mean()), float(e[~np.broadcast_to(m, e.shape)].mean()), int(keep.sum())


def linear_probe_r2(z_fit: torch.Tensor, y_fit: np.ndarray, z_val: torch.Tensor, y_val: np.ndarray, ridge: float = 1e-2,
                    val_mask: np.ndarray | None = None) -> float:
    """Ridge readout z -> y fitted on the fit frames, pooled R^2 on the val frames. LeWM Sec. 6-style
    physical-quantity probing: a linear map, so it measures what the latent holds linearly, not what a decoder can dig out."""
    z_fit, z_val = z_fit.double().cpu(), z_val.double().cpu()
    yf, yv = torch.as_tensor(y_fit, dtype=torch.double), torch.as_tensor(y_val, dtype=torch.double)
    mu, sd = z_fit.mean(0, keepdim=True), z_fit.std(0, keepdim=True) + 1e-6
    X = torch.cat([(z_fit - mu) / sd, torch.ones(len(z_fit), 1, dtype=torch.double)], 1)
    Xv = torch.cat([(z_val - mu) / sd, torch.ones(len(z_val), 1, dtype=torch.double)], 1)
    I = torch.eye(X.shape[1], dtype=torch.double); I[-1, -1] = 0
    W = torch.linalg.solve(X.T @ X + ridge * len(X) * I, X.T @ yf)
    pred = Xv @ W
    if val_mask is not None:                       # score on a subset of the val frames (fit unchanged)
        keep = torch.as_tensor(np.asarray(val_mask, bool))
        if int(keep.sum()) < 20:
            return float("nan")
        yv, pred = yv[keep], pred[keep]
    # POOLED R^2 over the target dims (1 - sum SSres / sum SStot): a per-target mean blows up on a target with
    # ~zero variance in the scored subset (the cube's height is constant while it sits on the table)
    ss_res = ((yv - pred) ** 2).sum(); ss_tot = ((yv - yv.mean(0, keepdim=True)) ** 2).sum()
    return float(1 - ss_res / ss_tot.clamp_min(1e-12))


def latent_stats(z: torch.Tensor) -> dict:
    """z (N, D): per-dim std (mean over dims), effective rank (exp of the eigenvalue entropy of the
    covariance) and RankMe (same on the singular values of the centred matrix)."""
    z = z.double()
    zc = z - z.mean(0, keepdim=True)
    std = float(zc.std(0).mean())
    cov = zc.T @ zc / max(len(z) - 1, 1)
    ev = torch.linalg.eigvalsh(cov).clamp_min(0)
    p = ev / ev.sum().clamp_min(1e-12)
    eff_rank = float(torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum()))
    sv = torch.linalg.svdvals(zc)
    q = sv / sv.sum().clamp_min(1e-12)
    rankme = float(torch.exp(-(q * torch.log(q.clamp_min(1e-12))).sum()))
    return dict(z_std=std, eff_rank=eff_rank, rankme=rankme, z_norm=float(z.norm(dim=-1).mean()),
                dim=int(z.shape[1]))


# ----------------------------------------------------------------------------- main
def main(a: argparse.Namespace) -> dict:
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    rng = np.random.default_rng(a.seed)

    model, wpath, epoch = load_lewm(a.run, a.epoch, device)
    print(f"[decoder] LeWM {a.run} epoch {epoch} <- {wpath}", flush=True)
    out_pt = ckpt_dir(a.run) / (a.out or f"decoder_epoch_{epoch}.pt")

    h5_path = dataset_path(a.dataset)
    if not a.no_bn_recal:
        t0 = time.time()
        n_bn = recalibrate_bn(model, h5_path, device, n_batches=a.bn_batches, seed=a.seed, img_size=a.img_size)
        print(f"[decoder] BatchNorm running stats re-estimated on {a.bn_batches} batches ({n_bn} BN layers, "
              f"{time.time() - t0:.0f}s)", flush=True)
    with h5py.File(h5_path, "r") as f:
        n_rows = f["pixels"].shape[0]
        raw_shape = f["pixels"].shape[1:]
        val_starts = sample_blocks(n_rows, a.n_val // a.block, a.block, rng)
        fit_starts = sample_blocks(n_rows, a.n_fit // a.block, a.block, rng, exclude=val_starts)
        t0 = time.time()
        val_u8 = read_blocks(f, val_starts, a.block)
        fit_u8 = read_blocks(f, fit_starts, a.block)
        sl_val, sl_fit = [slice(s, s + a.block) for s in val_starts], [slice(s, s + a.block) for s in fit_starts]
        key = touch_key(model)
        tstats = column_stats(f, key) if key else None
        touch_val = read_col_rows(f, key, sl_val, tstats) if key else None
        touch_fit = read_col_rows(f, key, sl_fit, tstats) if key else None
        probes = {}
        for name, col in (("block", a.probe_block_col), ("effector", a.probe_effector_col)):
            if col in f:
                probes[name] = (read_col_rows(f, col, sl_fit), read_col_rows(f, col, sl_val))
    print(f"[decoder] {h5_path}: {n_rows} rows of {raw_shape}; fit {len(fit_u8)} / val {len(val_u8)} frames "
          f"in {len(fit_starts)}+{len(val_starts)} blocks of {a.block} (read {time.time() - t0:.0f}s)"
          + (f"; touch input '{key}' (mean {tstats[0][0]:.3f} std {tstats[1][0]:.3f})" if key else ""), flush=True)
    fit_u8, val_u8 = resize_u8(fit_u8, a.img_size), resize_u8(val_u8, a.img_size)

    t0 = time.time()
    z_fit = encode_frames(model, fit_u8, device, a.encode_batch, a.img_size, touch=touch_fit).to(device)
    z_val = encode_frames(model, val_u8, device, a.encode_batch, a.img_size, touch=touch_val).to(device)
    probe_r2 = {n: linear_probe_r2(z_fit, yf, z_val, yv) for n, (yf, yv) in probes.items()}
    if "block" in probes and "effector" in probes:
        # the cube's position when it is NOT in the gripper (>= --far-m from the effector): the frames where the
        # latent has to see the cube itself rather than infer it from the arm
        far = np.linalg.norm(probes["block"][1] - probes["effector"][1], axis=1) >= a.far_m
        probe_r2["block_far"] = linear_probe_r2(z_fit, probes["block"][0], z_val, probes["block"][1], val_mask=far)
        probe_r2["n_far"] = int(far.sum())
    if probe_r2:
        print("[decoder] linear probe R^2 (val): " + "  ".join(f"{n} {v:.3f}" for n, v in probe_r2.items()), flush=True)
    stats = latent_stats(z_val.cpu())
    print(f"[decoder] encoded {len(z_fit) + len(z_val)} frames in {time.time() - t0:.0f}s; val latent: "
          f"z_std {stats['z_std']:.3f} eff_rank {stats['eff_rank']:.1f}/{stats['dim']} rankme {stats['rankme']:.1f} "
          f"|z| {stats['z_norm']:.2f}", flush=True)

    dec = LatentDecoder(z_dim=z_fit.shape[1], hidden=a.hidden, depth=a.depth, heads=a.heads,
                        out_hw=a.img_size).to(device)
    if a.init:
        prev, _ = load_decoder(a.init, device, z_dim=z_fit.shape[1])
        dec.load_state_dict(prev.state_dict()); dec.train().requires_grad_(True)
    n_params = sum(p.numel() for p in dec.parameters())
    print(f"[decoder] arch={dec.config()} params={n_params / 1e6:.2f}M", flush=True)

    def target(u8):
        return torch.as_tensor(np.ascontiguousarray(u8), device=device).permute(0, 3, 1, 2).float() / 255.0

    # the fit recipe of src/train_decoder.py: AdamW, linear warmup + cosine, grad clip 1, pixel MSE
    opt = torch.optim.AdamW(dec.parameters(), lr=a.lr, weight_decay=a.wd)
    warm = min(100, max(1, a.steps // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, a.steps - warm))))

    @torch.no_grad()
    def evaluate() -> float:
        dec.eval()
        tot = 0.0
        for i in range(0, len(z_val), a.batch):
            tot += float(torch.nn.functional.mse_loss(dec(z_val[i:i + a.batch]), target(val_u8[i:i + a.batch]),
                                                      reduction="sum"))
        dec.train()
        return tot / (len(z_val) * 3 * a.img_size * a.img_size)

    t0 = time.time()
    dec.train()
    for step in range(1, a.steps + 1):
        idx = np.random.randint(0, len(z_fit), size=a.batch)
        loss = torch.nn.functional.mse_loss(dec(z_fit[idx]), target(fit_u8[idx]))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)
        opt.step(); sched.step()
        if step % a.log_every == 0 or step == a.steps:
            print(f"[decoder] step {step}/{a.steps} mse={loss.item():.5f} lr={sched.get_last_lr()[0]:.2e} "
                  f"({(time.time() - t0) / step * 1000:.0f} ms/step)", flush=True)
    vmse = evaluate()
    dec.eval()
    with torch.no_grad():
        rec_val = np.concatenate([dec.to_uint8_hwc(z_val[i:i + a.batch]) for i in range(0, len(z_val), a.batch)])
    cube_mse, bg_mse, cube_frames = masked_mse(rec_val, val_u8, cube_mask(val_u8))
    print(f"[decoder] cube-pixel val MSE {cube_mse:.4f} vs background {bg_mse:.4f} over {cube_frames} frames with a visible cube", flush=True)
    torch.save({"decoder": dec.state_dict(), "z_dim": int(z_fit.shape[1]), "arch": dec.config(), "val_mse": vmse,
                "steps": a.steps, "n_frames": int(len(z_fit)), "run": a.run, "epoch": epoch, "weights": str(wpath),
                "dataset": str(h5_path), "seed": a.seed, "bn_recal": not a.no_bn_recal}, out_pt)
    meta = dict(run=a.run, epoch=epoch, weights=str(wpath), dataset=str(h5_path), val_mse=vmse, steps=a.steps,
                n_fit=int(len(z_fit)), n_val=int(len(z_val)), block=a.block, seed=a.seed, img_size=a.img_size,
                bn_recal=not a.no_bn_recal, bn_batches=a.bn_batches, touch_key=key,
                cube_mse=cube_mse, bg_mse=bg_mse, cube_frames=cube_frames,
                probe_r2={n: float(v) for n, v in probe_r2.items()},
                fit_starts=fit_starts.tolist(), val_starts=val_starts.tolist(), latent=stats, decoder=out_pt.name)
    out_pt.with_suffix(".json").write_text(json.dumps(meta, indent=1))

    k = min(a.sheet_n, len(z_val))
    pick = np.linspace(0, len(z_val) - 1, k).astype(int)
    rec = dec.to_uint8_hwc(z_val[pick])
    from PIL import Image
    sheet = np.concatenate([np.concatenate(list(val_u8[pick]), 1), np.concatenate(list(rec), 1)], 0)
    Image.fromarray(sheet).save(out_pt.with_suffix(".sheet.png"))
    print(f"[decoder] val mse={vmse:.5f} -> {out_pt}  (+ .json, .sheet.png)", flush=True)
    return meta


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="checkpoint folder name under $STABLEWM_HOME/checkpoints/")
    p.add_argument("--epoch", type=int, default=None, help="weights_epoch_<E>.pt to decode (default: latest)")
    p.add_argument("--dataset", default="ogbench/cube_single_expert", help="h5 under $STABLEWM_HOME/datasets/ (or a path)")
    p.add_argument("--out", default="", help="decoder file name (default decoder_epoch_<E>.pt, beside the weights)")
    p.add_argument("--n-fit", type=int, default=20000)
    p.add_argument("--n-val", type=int, default=1000)
    p.add_argument("--block", type=int, default=50, help="contiguous rows per sampled block")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--encode-batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--sheet-n", type=int, default=10)
    p.add_argument("--init", default="")
    p.add_argument("--seed", type=int, default=0, help="frame sampling seed: keep it FIXED across a sweep")
    p.add_argument("--no-bn-recal", action="store_true", help="use the checkpoint's BatchNorm running stats as saved")
    p.add_argument("--bn-batches", type=int, default=30, help="windows batches (x64) for the BatchNorm re-estimation")
    p.add_argument("--probe-block-col", default="privileged_block_0_pos", help="h5 column for the object-position linear probe")
    p.add_argument("--probe-effector-col", default="proprio_effector_pos", help="h5 column for the end-effector linear probe")
    p.add_argument("--far-m", type=float, default=0.05, help="cube-to-effector distance (m) defining the 'cube not in hand' probe subset")
    p.add_argument("--device", default=None)
    return p.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
