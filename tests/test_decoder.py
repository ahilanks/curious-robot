"""Offline smoke test for the post-hoc pixel decoder path (no network, CPU, ~1 min).

Builds a tiny random WorldModel checkpoint + a synthetic state_latest.npz ring, then runs
src/train_decoder.py (LeWM decoder, cached latents), reloads through load_decoder, checks the
legacy conv checkpoint path, and renders an imagined rollout with src/viz_decoder_rollout.py.
Run: python3 tests/test_decoder.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from model.decoder import ConvLatentDecoder, LatentDecoder, load_decoder  # noqa: E402
from model.state_encoder import WorldModel  # noqa: E402
from src import train_decoder as td  # noqa: E402
from src import viz_decoder_rollout as vz  # noqa: E402

ok = []


def check(name, cond):
    ok.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name)


def make_fixture(tmp: Path, n_frames: int = 24):
    """Random-init pixels-only WM (small predictor) + a smooth synthetic ring of n_frames."""
    args = dict(no_proprio=True, history_size=3, action_block=5, wm_dropout=0.0,
                wm_pred_depth=1, wm_pred_heads=2, wm_pred_dim_head=16, wm_pred_mlp_dim=64)
    wm = WorldModel(n_dof=6, action_block=5, history_size=3, dropout=0.0, use_proprio=False,
                    depth=1, heads=2, dim_head=16, mlp_dim=64)
    ckpt = tmp / "ckpt_0000010.pt"
    torch.save({"step": 10, "wm": wm.state_dict(), "args": args}, ckpt)
    yy, xx = np.mgrid[0:224, 0:224]
    px = np.zeros((1, n_frames, 224, 224, 3), np.uint8)
    for t in range(n_frames):                       # a bright disc drifting across the frame
        cx, cy = 40 + 6 * t, 112 + 30 * np.sin(t / 3)
        disc = ((xx - cx) ** 2 + (yy - cy) ** 2) < 30 ** 2
        px[0, t, ..., 0] = 40 + 200 * disc
        px[0, t, ..., 1] = 60 + 100 * disc
        px[0, t, ..., 2] = 80
    is_start = np.zeros((1, n_frames), bool)
    is_start[0, 0] = True
    state = tmp / "state_latest.npz"
    np.savez(state, fmt=np.int64(1), step=np.int64(n_frames), head=np.array([n_frames]),
             count=np.array([n_frames]), pixels=px,
             proprio=np.zeros((1, n_frames, 18), np.float32),
             action=np.random.randn(1, n_frames, 30).astype(np.float32) * 0.1,
             r=np.zeros((1, n_frames), np.float32), done=np.zeros((1, n_frames), np.float32),
             is_start=is_start, prio=np.ones((1, n_frames), np.float32))
    return ckpt, state


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    tmp = Path(tempfile.mkdtemp(prefix="dec_smoke_"))
    ckpt, state = make_fixture(tmp)

    # ---- architecture: LeWM App. D shapes + param count sanity
    dec = LatentDecoder(z_dim=192, hidden=64, depth=1, heads=2)
    with torch.no_grad():
        y = dec(torch.randn(2, 192))
    check("lewm decoder output (B,3,224,224) in [0,1]",
          tuple(y.shape) == (2, 3, 224, 224) and float(y.min()) >= 0 and float(y.max()) <= 1)
    check("196 query tokens for 224/16", dec.queries.shape[1] == 196)
    check("uint8 hwc helper", dec.to_uint8_hwc(torch.randn(1, 192)).shape == (1, 224, 224, 3))

    # ---- train_decoder end to end (cached latents, periodic save, sheet)
    out = tmp / "decoder.pt"
    res = td.main(td.parse_args(["--ckpt", str(ckpt), "--state", str(state), "--out", str(out),
                                 "--steps", "12", "--batch", "4", "--encode-batch", "8",
                                 "--hidden", "64", "--depth", "1", "--heads", "2",
                                 "--save-every", "6", "--log-every", "6", "--device", "cpu"]))
    check("train_decoder wrote ckpt + sheet", out.exists() and Path(res["sheet"]).exists())
    check("val mse finite", np.isfinite(res["val_mse"]))
    dec2, meta = load_decoder(str(out), "cpu", z_dim=192)
    check("reloaded decoder is LeWM arch with saved knobs",
          isinstance(dec2, LatentDecoder) and meta["arch"]["hidden"] == 64 and meta["steps"] == 12)
    check("reloaded decoder is frozen/eval",
          not dec2.training and not any(p.requires_grad for p in dec2.parameters()))
    try:
        load_decoder(str(out), "cpu", z_dim=256)
        check("z_dim mismatch raises", False)
    except ValueError:
        check("z_dim mismatch raises", True)

    # ---- --init continues from a saved decoder
    out2 = tmp / "decoder2.pt"
    td.main(td.parse_args(["--ckpt", str(ckpt), "--state", str(state), "--out", str(out2),
                           "--steps", "2", "--batch", "4", "--hidden", "64", "--depth", "1",
                           "--heads", "2", "--init", str(out), "--save-every", "0",
                           "--log-every", "1", "--device", "cpu"]))
    check("--init round trip", out2.exists())

    # ---- legacy conv checkpoint (no arch key) still loads as ConvLatentDecoder
    conv = ConvLatentDecoder(z_dim=192)
    legacy = tmp / "legacy.pt"
    torch.save({"decoder": conv.state_dict(), "z_dim": 192, "val_mse": 0.1}, legacy)
    dec3, _ = load_decoder(str(legacy), "cpu", z_dim=192)
    check("legacy conv ckpt loads", isinstance(dec3, ConvLatentDecoder)
          and dec3(torch.randn(1, 192)).shape == (1, 3, 224, 224))

    # ---- imagined rollout sheet (LeWM Fig. 7 arithmetic) + gif
    png, gif = tmp / "rollout.png", tmp / "rollout.gif"
    r = vz.main(vz.parse_args(["--ckpt", str(ckpt), "--state", str(state), "--decoder", str(out),
                               "--out", str(png), "--horizon", "4", "--n", "2", "--gif", str(gif),
                               "--device", "cpu"]))
    check("rollout png + gif written", png.exists() and gif.exists())
    check("rollout metrics per segment/step",
          len(r["z_mse"]) == 2 and all(len(m) == 4 and np.all(np.isfinite(m)) for m in r["z_mse"]))
    st = np.load(state)
    try:
        vz.sample_segments(st, 1, 1000, np.random.default_rng(0), tries=50)
        check("segment sampler rejects impossible length", False)
    except SystemExit:
        check("segment sampler rejects impossible length", True)

    failed = [n for n, c in ok if not c]
    print(f"\n{len(ok) - len(failed)}/{len(ok)} passed" + (f"; FAILED: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
