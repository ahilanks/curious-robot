"""Block decodability of a sim head: pixel error of its post-hoc decoder INSIDE the block's pixels vs outside.

The sim twin of the LeWM-Cube cube-pixel MSE (lewm/train_decoder.py): frames are drawn uniformly from the
run's state ring, the magenta/purple block mask (src/viz_goal_archive.block_mask, 2-px dilation) is taken from
the REAL frame, and decode(encode(frame)) is scored inside vs outside the mask over the frames with at least
--min-px block pixels. Frames are held out of the decoder's fit only in expectation (the decoder was fitted on a
uniform thinning of the same ring), so use it as a relative number between heads fitted the same way.

    python src/decoder_block_mse.py --ckpt runs/<run>/ckpt_X.pt --decoder runs/<run>/decoder_lewm.pt \
        --state runs/<run>/state_latest.npz [--n 3000 --seed 1]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.decoder import load_decoder                          # noqa: E402
from src.train import to_norm_pixel                              # noqa: E402
from src.train_decoder import load_wm                            # noqa: E402
from src.viz_goal_archive import block_mask                      # noqa: E402


def main(a):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(a.seed)
    wm = load_wm(a.ckpt, device)
    dec, meta = load_decoder(a.decoder, device, z_dim=wm.z_dim)
    st = np.load(a.state)
    px, prop, count = st["pixels"], st["proprio"], st["count"]
    pairs = [(e, i) for e in range(px.shape[0]) for i in range(int(count[e]))]
    pick = rng.choice(len(pairs), size=min(a.n, len(pairs)), replace=False)
    idx = sorted(pairs[j] for j in pick)
    frames = np.stack([px[e, i] for e, i in idx]); props = np.stack([prop[e, i] for e, i in idx]).astype(np.float32)
    from scipy.ndimage import binary_dilation
    m = block_mask(frames)
    stc = np.zeros((3, 3, 3), bool); stc[1] = True
    m = binary_dilation(m, structure=stc, iterations=2)
    npx = m.reshape(len(m), -1).sum(1)
    keep = npx >= a.min_px
    recs = []
    with torch.no_grad():
        for i in range(0, len(frames), 64):
            z = wm.encode(to_norm_pixel(frames[i:i + 64], device), torch.as_tensor(props[i:i + 64], device=device))
            recs.append(dec.to_uint8_hwc(z))
    rec = np.concatenate(recs)
    err = ((rec.astype(np.float32) - frames.astype(np.float32)) / 255.0) ** 2
    mk = np.broadcast_to(m[keep][..., None], err[keep].shape)
    inside, outside = float(err[keep][mk].mean()), float(err[keep][~mk].mean())
    # does the decode put block-coloured pixels where the block is? (IoU of the block mask on the decode vs the real one)
    md = block_mask(rec[keep])
    inter = (md & m[keep]).reshape(int(keep.sum()), -1).sum(1); union = (md | m[keep]).reshape(int(keep.sum()), -1).sum(1)
    iou = float(np.mean(inter / np.maximum(union, 1)))
    out = dict(ckpt=a.ckpt, decoder=a.decoder, state=a.state, n_frames=int(len(frames)), frames_with_block=int(keep.sum()),
               block_px_mean=float(npx[keep].mean()) if keep.any() else 0.0, mse_all=float(err.mean()),
               mse_block_px=inside, mse_outside_px=outside, ratio=inside / max(outside, 1e-9), block_iou=iou,
               decoder_val_mse=float(meta.get("val_mse", float("nan"))))
    print(json.dumps(out, indent=1))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True); p.add_argument("--decoder", required=True); p.add_argument("--state", required=True)
    p.add_argument("--n", type=int, default=3000); p.add_argument("--min-px", type=int, default=150)
    p.add_argument("--seed", type=int, default=1); p.add_argument("--out", default="")
    main(p.parse_args())
