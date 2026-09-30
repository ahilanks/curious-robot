"""Render Push-T episodes from runs' replay.pt as an mp4 grid: what did each agent actually do?

    python toy/pusht_video.py --cells ptf_none_s0:-1:0 ptf_spep_s0:-1:0 ... --cols 3 --out behaviour.mp4

A cell is RUN:ROUND:ENV (ROUND -1 = the last saved round; runs under --runs). Each frame draws the arena, the T with
its centre-of-gravity trail, the agent (red) with a faint trail, and the cell's label. Needs ffmpeg on PATH.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from push_t import T_COG, WALL_HI, WALL_LO

BAR = np.array([(-60, 0), (60, 0), (60, 30), (-60, 30)], dtype=np.float64)
STEM = np.array([(-15, 30), (15, 30), (15, 120), (-15, 120)], dtype=np.float64)


def world(local, pose):
    c, s = np.cos(pose[2]), np.sin(pose[2])
    return pose[:2] + local @ np.array([[c, s], [-s, c]])


def load_cell(runs, spec):
    run, rnd, env = spec.split(":")
    r = torch.load(Path(runs) / run / "replay.pt")
    rnd = int(rnd) % r["state"].shape[0]
    return f"{run.split('_')[1]} s{run.split('_s')[-1]}  r{rnd}", r["state"][rnd, :, int(env)].double().numpy()


def draw(S, t, label, px):
    k = px / 512
    im = Image.new("RGB", (px, px), (248, 247, 243))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, px - 1, px - 1], outline=(120, 120, 120), width=max(1, round(WALL_LO * k)))
    cog = np.array([world(T_COG[None], s[4:7])[0] for s in S[:t + 1]]) * k
    if t:
        d.line([tuple(p) for p in S[:t + 1, :2] * k], fill=(240, 190, 190), width=1)
        d.line([tuple(p) for p in cog], fill=(150, 170, 220), width=2)
    for part in (BAR, STEM):
        d.polygon([tuple(p) for p in world(part, S[t, 4:7]) * k], fill=(80, 110, 190))
    ax, ay, r = S[t, 0] * k, S[t, 1] * k, 15 * k
    d.ellipse([ax - r, ay - r, ax + r, ay + r], fill=(215, 60, 60))
    d.text((4, 3), label, fill=(30, 30, 30))
    return np.asarray(im)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cells", nargs="+", required=True, help="RUN:ROUND:ENV, row-major")
    p.add_argument("--cols", type=int, default=3)
    p.add_argument("--runs", default=str(Path(__file__).resolve().parent / "runs" / "pusht"))
    p.add_argument("--px", type=int, default=240, help="cell size in pixels (the 512 px arena, scaled)")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    cells = [load_cell(args.runs, c) for c in args.cells]
    rows = -(-len(cells) // args.cols)
    W, H = args.cols * args.px, rows * args.px
    T1 = max(S.shape[0] for _, S in cells)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W}x{H}", "-r", str(args.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                           "-crf", "20", args.out], stdin=subprocess.PIPE)
    for t in range(T1):
        frame = np.full((H, W, 3), 255, np.uint8)
        for i, (label, S) in enumerate(cells):
            y, x = divmod(i, args.cols)
            frame[y * args.px:(y + 1) * args.px, x * args.px:(x + 1) * args.px] = draw(S, min(t, len(S) - 1), label,
                                                                                    args.px)
        ff.stdin.write(frame.tobytes())
    ff.stdin.close()
    ff.wait()
    print(f"[pusht_video] {len(cells)} cells x {T1} frames -> {args.out}")


if __name__ == "__main__":
    main()
