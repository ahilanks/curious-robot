"""What did the ladder choose? Decode every archived goal through the post-hoc pixel decoder.

For each goal in a checkpoint's `goal_archive` (src/goal_explore.py GoalArchive: goal photo o_{t+1},
source photo o_t, the action between them, the WM-surprise score at capture) this renders
    [goal photo | decode(z*)]          (z* = frozen encoder on the goal photo; --with-source adds
                                        [source photo | decode(z_src)] so each cell is the transition)
ranked by archive score, plus per-goal numbers: recon L1, latent jump ||z* - z_src||, and whether
the block is visible in the photo vs in the decode (purple-pixel mask, sim colours). Diagnostic only.

    python src/viz_goal_archive.py --ckpt runs/wr_sleepret2/ckpt_0200000.pt \
        --decoder runs/wr_sleepret2/decoder_lewm.pt --out runs/wr_sleepret2/goal_archive.png
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.decoder import load_decoder                                    # noqa: E402
from src.train import to_norm_pixel                                       # noqa: E402
from src.train_decoder import load_wm, resolve_ckpt_spec                  # noqa: E402


def block_mask(img_u8: np.ndarray, min_rb: int = 90, margin: int = 60) -> np.ndarray:
    """Purple/magenta block pixels in the sim wrist camera (R and B both well above G).
    Calibrated on wr_sleepret2 frames: block median RGB (255, 104, 255); table (167, 107, 21),
    wall (58, 97, 136) and gripper (255, 255, 48) never pass. Works on decodes too (they are
    smooth, so a decoded block shows as a smaller blob)."""
    x = img_u8.astype(np.int16)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    return (r > g + margin) & (b > g + margin) & (r > min_rb) & (b > min_rb)


@torch.no_grad()
def encode_batched(wm, px: np.ndarray, prop: np.ndarray, device, batch: int = 64) -> torch.Tensor:
    zs = []
    for i in range(0, len(px), batch):
        zs.append(wm.encode(to_norm_pixel(px[i:i + batch], device),
                            torch.as_tensor(prop[i:i + batch], device=device)))
    return torch.cat(zs)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="run checkpoint (path or HF '<run>/ckpt_X.pt')")
    p.add_argument("--decoder", required=True, help="decoder ckpt fitted on this run's encoder")
    p.add_argument("--out", required=True, help="output PNG")
    p.add_argument("--json", default="", help="per-goal metrics (default: <out>.json)")
    p.add_argument("--cols", type=int, default=8)
    p.add_argument("--with-source", action="store_true", help="4-tile cells: src | goal | dec(src) | dec(goal)")
    p.add_argument("--min-block-px", type=int, default=60, help="mask pixels for 'block visible'")
    p.add_argument("--hf-repo", default=None)
    p.add_argument("--device", default=None)
    a = p.parse_args()
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = resolve_ckpt_spec(a.ckpt, a.hf_repo)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch = ck.get("goal_archive")
    if not arch:
        raise SystemExit(f"{ckpt_path} has no goal_archive")
    arch = arch if isinstance(arch, (list, tuple)) else [arch]
    cat = lambda k: np.concatenate([np.asarray(d[k]) for d in arch])
    gpx, spx, gprop, sprop = cat("gpx"), cat("spx"), cat("gprop"), cat("sprop")
    score, ref, sact = cat("score"), cat("ref"), cat("sact")
    n = len(gpx)
    print(f"[archive] {n} goals from {len(arch)} archive(s) in {ckpt_path} "
          f"(score {score.min():.2f}..{score.max():.2f})", flush=True)

    wm = load_wm(ckpt_path, device)
    dec, dmeta = load_decoder(a.decoder, device, z_dim=wm.z_dim)
    z_goal = encode_batched(wm, gpx, gprop, device)
    z_src = encode_batched(wm, spx, sprop, device)
    dec_goal = dec.to_uint8_hwc(z_goal)
    dec_src = dec.to_uint8_hwc(z_src)
    jump = (z_goal - z_src).norm(dim=1).cpu().numpy()

    order = np.argsort(-score)                                   # archive ranking: highest surprise first
    rows = []
    for rank, i in enumerate(order):
        bp, bd = int(block_mask(gpx[i]).sum()), int(block_mask(dec_goal[i]).sum())
        rows.append(dict(rank=rank, idx=int(i), score=float(score[i]), ref=float(ref[i]),
                         recon_l1=float(np.abs(dec_goal[i].astype(np.int16) - gpx[i].astype(np.int16)).mean()),
                         latent_jump=float(jump[i]), act_norm=float(np.linalg.norm(sact[i])),
                         block_px_photo=bp, block_px_decode=bd,
                         block_in_photo=bp >= a.min_block_px, block_in_decode=bd >= a.min_block_px))
    vis = [r for r in rows if r["block_in_photo"]]
    hit = [r for r in vis if r["block_in_decode"]]
    print(f"[archive] recon L1 mean {np.mean([r['recon_l1'] for r in rows]):.1f} | latent jump median "
          f"{np.median(jump):.2f} | block visible in {len(vis)}/{n} goal photos, decoded in {len(hit)}/{len(vis)} of those "
          f"(decoder val_mse {dmeta.get('val_mse', float('nan')):.4f})", flush=True)

    # ---- sheet: ranked cells, each [goal | dec(goal)] (or 4 tiles with --with-source) + label bar
    from PIL import Image, ImageDraw, ImageFont
    T, LAB = gpx.shape[1], 18
    tiles_per = 4 if a.with_source else 2
    cw, ch = T * tiles_per, T + LAB
    ncol = max(1, min(a.cols, n)); nrow = math.ceil(n / ncol)
    sheet = Image.new("RGB", (ncol * cw, nrow * ch), (24, 24, 24))
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default(size=12)
    except TypeError:
        font = ImageFont.load_default()
    for cell, r in enumerate(rows):
        i, cx, cy = r["idx"], (cell % ncol) * cw, (cell // ncol) * ch
        tiles = ([spx[i], gpx[i], dec_src[i], dec_goal[i]] if a.with_source else [gpx[i], dec_goal[i]])
        for t, img in enumerate(tiles):
            sheet.paste(Image.fromarray(np.ascontiguousarray(img)), (cx + t * T, cy + LAB))
        flag = "" if not r["block_in_photo"] else (" blk:dec" if r["block_in_decode"] else " blk:LOST")
        d.text((cx + 4, cy + 3), f"#{r['rank']:02d} score {r['score']:.2f}  L1 {r['recon_l1']:.0f}  "
                                 f"|dz| {r['latent_jump']:.1f}{flag}", fill=(235, 235, 235), font=font)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(a.out)
    jpath = a.json or (str(Path(a.out).with_suffix("")) + ".json")
    Path(jpath).write_text(json.dumps(dict(ckpt=str(ckpt_path), decoder=a.decoder, n=n,
                                           layout=("src|goal|dec(src)|dec(goal)" if a.with_source else "goal|dec(goal)"),
                                           goals=rows), indent=1))
    print(f"[archive] sheet {sheet.size[0]}x{sheet.size[1]} -> {a.out}\n[archive] metrics -> {jpath}", flush=True)


if __name__ == "__main__":
    main()
