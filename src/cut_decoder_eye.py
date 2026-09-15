"""Cut a --live-view-record mp4 down to the stretches where the wrist camera sees blocks.

The recording's first tile (224x224 under a 28 px header) is the real wrist frame; frames whose
magenta-pixel count (src/viz_goal_archive.block_mask, sim colours) clears --min-px are kept,
runs shorter than --min-run are dropped, gaps up to --gap frames are bridged, and the kept runs
are concatenated with a 0.5 s black separator so cuts are visible (the header still shows the step).

    python src/cut_decoder_eye.py runs/sim_eye_blocks/decoder_eye.mp4 --out runs/sim_eye_blocks/decoder_eye_blocks.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.viz_goal_archive import block_mask     # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mp4")
    p.add_argument("--out", required=True)
    p.add_argument("--tile", type=int, default=224, help="tile size in the recording")
    p.add_argument("--header", type=int, default=28, help="header height in the recording")
    p.add_argument("--min-px", type=int, default=150, help="block pixels in the wrist tile to count a frame")
    p.add_argument("--min-run", type=int, default=15, help="drop kept runs shorter than this (frames)")
    p.add_argument("--gap", type=int, default=10, help="bridge gaps up to this many frames inside a run")
    p.add_argument("--fps", type=float, default=0, help="output fps (default: input)")
    p.add_argument("--sep", type=float, default=0.5, help="black separator between runs (s)")
    a = p.parse_args()

    import imageio
    r = imageio.get_reader(a.mp4, format="ffmpeg")
    fps = a.fps or float(r.get_meta_data().get("fps", 15))
    frames = [f for f in r]
    r.close()
    T, H = a.tile, a.header
    px = np.array([int(block_mask(f[H:H + T, :T]).sum()) for f in frames])
    keep = px >= a.min_px
    # bridge short gaps, then drop short runs
    idx = np.where(keep)[0]
    runs: list[list[int]] = []
    for i in idx:
        if runs and i - runs[-1][-1] <= a.gap + 1:
            runs[-1].append(int(i))
        else:
            runs.append([int(i)])
    runs = [(r_[0], r_[-1]) for r_ in runs if r_[-1] - r_[0] + 1 >= a.min_run]
    n_keep = sum(e - s + 1 for s, e in runs)
    print(f"[cut] {len(frames)} frames @ {fps:.0f} fps; {int(keep.sum())} with >= {a.min_px} block px; "
          f"{len(runs)} runs kept ({n_keep} frames = {n_keep / fps:.0f}s)", flush=True)
    if not runs:
        raise SystemExit("[cut] nothing to keep -- lower --min-px or record longer")
    sep = np.zeros_like(frames[0])
    n_sep = int(round(a.sep * fps))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    w = imageio.get_writer(a.out, fps=fps, codec="libx264", quality=8, macro_block_size=None)
    for k, (s, e) in enumerate(runs):
        if k:
            for _ in range(n_sep):
                w.append_data(sep)
        for i in range(s, e + 1):
            w.append_data(frames[i])
        print(f"[cut]   run {k}: frames {s}-{e} ({e - s + 1}), peak {px[s:e + 1].max()} block px", flush=True)
    w.close()
    print(f"[cut] -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
