"""Fit the post-hoc pixel decoder (model/decoder.py) on a run's saved state.

Frames come from a --save-state snapshot (state_latest.npz: the raw replay ring, real
wrist photos on a hardware run); latents come from the checkpoint's FROZEN encoder
(wm.encode under no_grad — the WM is never touched). Objective: plain pixel MSE against
frame/255, the standard post-hoc probe for JEPA-family models. Outputs the decoder ckpt
plus a real-vs-recon contact sheet so quality is eyeballable before trusting the
dashboard's "decoder's eye" row.

    python src/train_decoder.py --ckpt runs/<run>/ckpt_XXXXXXX.pt \
        --state runs/<run>/state_latest.npz --out runs/<run>/decoder.pt

Then pass `--decoder runs/<run>/decoder.pt` to a --live-view train.py run.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.state_encoder import WorldModel, pred_dims_from_args   # noqa: E402
from model.decoder import LatentDecoder                           # noqa: E402
from src.train import to_norm_pixel                               # noqa: E402


def load_frames(state_path: str, max_frames: int) -> np.ndarray:
    st = np.load(state_path)
    px, count = st["pixels"], st["count"]                  # (n_envs, cap, H, W, 3) uint8
    frames = np.concatenate([px[e, :int(count[e])] for e in range(px.shape[0])])
    if len(frames) > max_frames:                            # uniform thin to bound memory/time
        frames = frames[np.linspace(0, len(frames) - 1, max_frames).astype(int)]
    return frames


def main(a: argparse.Namespace) -> None:
    device = a.device or ("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location=device, weights_only=False)
    ck_args = ck.get("args", {})
    get = ck_args.get if isinstance(ck_args, dict) else (lambda k, d=None: getattr(ck_args, k, d))
    wm = WorldModel(n_dof=6, action_block=int(get("action_block", 5) or 5),
                    history_size=int(get("history_size", 3) or 3),
                    dropout=float(get("wm_dropout", 0.1) or 0.1),
                    use_proprio=not bool(get("no_proprio", False)),
                    **pred_dims_from_args(ck_args)).to(device)
    wm.load_state_dict(ck["wm"])
    wm.eval().requires_grad_(False)
    print(f"[decoder] encoder from {a.ckpt} (z_dim={wm.z_dim}, frozen)")

    frames = load_frames(a.state, a.max_frames)
    n_val = max(8, int(0.05 * len(frames)))
    val, train = frames[:n_val], frames[n_val:]
    print(f"[decoder] {len(train)} train / {len(val)} val frames from {a.state}")

    dec = LatentDecoder(z_dim=wm.z_dim).to(device)
    opt = torch.optim.AdamW(dec.parameters(), lr=a.lr)
    prop0 = torch.zeros(a.batch, 18, device=device)         # pixels-only encode ignores proprio

    def encode(batch_u8: np.ndarray) -> torch.Tensor:
        with torch.no_grad():
            return wm.encode(to_norm_pixel(batch_u8, device), prop0[:len(batch_u8)])

    def target(batch_u8: np.ndarray) -> torch.Tensor:
        t = torch.as_tensor(np.ascontiguousarray(batch_u8), device=device)
        return t.permute(0, 3, 1, 2).float() / 255.0

    t0 = time.time()
    for step in range(1, a.steps + 1):
        idx = np.random.randint(0, len(train), size=a.batch)
        loss = torch.nn.functional.mse_loss(dec(encode(train[idx])), target(train[idx]))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == a.steps:
            print(f"[decoder] step {step}/{a.steps} mse={loss.item():.5f} "
                  f"({(time.time()-t0)/step*1000:.0f} ms/step)", flush=True)

    dec.eval()
    with torch.no_grad():
        vmse = float(torch.nn.functional.mse_loss(dec(encode(val[:64])), target(val[:64])))
    torch.save({"decoder": dec.state_dict(), "z_dim": wm.z_dim, "val_mse": vmse,
                "ckpt": str(a.ckpt), "state": str(a.state)}, a.out)
    print(f"[decoder] val mse={vmse:.5f} -> {a.out}")

    # contact sheet: top = real, bottom = decode(encode(real)) — the preserved-info picture
    k = min(8, len(val))
    with torch.no_grad():
        rec = dec.to_uint8_hwc(encode(val[:k]))
    sheet = np.concatenate([np.concatenate(list(val[:k]), axis=1),
                            np.concatenate(list(rec), axis=1)], axis=0)
    sheet_path = str(Path(a.out).with_suffix(".sheet.png"))
    try:
        from PIL import Image
        Image.fromarray(sheet).save(sheet_path)
    except ImportError:
        import cv2
        cv2.imwrite(sheet_path, sheet[..., ::-1])
    print(f"[decoder] contact sheet (top real / bottom recon) -> {sheet_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ckpt", required=True, help="run checkpoint (frozen encoder source)")
    p.add_argument("--state", required=True, help="state_latest.npz with the frame ring")
    p.add_argument("--out", required=True, help="output decoder .pt path")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-frames", type=int, default=20000)
    p.add_argument("--device", default=None)
    main(p.parse_args())
