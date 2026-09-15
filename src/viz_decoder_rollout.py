"""Open-loop imagined rollouts through the decoder's eye (LeWM Fig. 7 / Fig. 11) — DIAGNOSTIC.

For each sampled segment of a saved state ring: encode the first H real frames (H = the
checkpoint's history_size, LeWM's "three image observations as context"), then let the
predictor roll forward AUTOREGRESSIVELY for T steps under the ACTIONS THE ARM ACTUALLY TOOK
(same arithmetic as lewm.jepa.JEPA.rollout: predict(z[-H:], act_emb[-H:])[:, -1], append,
repeat). Every latent — context and imagined — is decoded by the post-hoc decoder and laid
against the real frames. Whatever drifts between the two rows is what the planner cannot see
or cannot predict. Nothing here touches a gradient.

    python src/viz_decoder_rollout.py --ckpt runs/<run>/ckpt_X.pt --state runs/<run>/state_latest.npz \
        --decoder runs/decoder_wrs2.pt --out runs/<run>/rollout.png [--horizon 8 --n 4 --gif ...]

Output: PNG with, per segment, row 1 = real frames, row 2 = decoded context | decoded
imagined future (white bar marks the context/future boundary); per-step latent MSE
(predicted vs encoded real) and pixel L1 (decoded vs real) printed as a table. Optional GIF
plays real | imagined side by side over time.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.decoder import load_decoder                                    # noqa: E402
from src.train import to_norm_pixel                                       # noqa: E402
from src.train_decoder import load_wm, resolve_ckpt_spec, resolve_state   # noqa: E402
from src.viz_goal_archive import block_mask                                # noqa: E402


def sample_segments(st, n: int, length: int, rng: np.random.Generator, tries: int = 10000):
    """(env, start) pairs whose `length` frames lie inside ONE episode of the ring's filled extent.
    Buffer semantics: pixels[e,i] is the obs at step i, action[e,i] the action taken from it,
    done[e,i] closes transition i, is_start[e,i] opens an episode at i."""
    count, done, is_start = st["count"], st["done"], st["is_start"]
    out = []
    for _ in range(tries):
        if len(out) >= n:
            break
        e = int(rng.integers(0, len(count)))
        c = int(count[e])
        if c < length:
            continue
        i = int(rng.integers(0, c - length + 1))
        if done[e, i:i + length - 1].any() or is_start[e, i + 1:i + length].any():
            continue
        if not any(oe == e and abs(oi - i) < length for oe, oi in out):   # non-overlapping
            out.append((e, i))
    if not out:
        raise SystemExit("no episode-contiguous segment of that length in the state ring")
    return out


@torch.no_grad()
def imagine(wm, frames_u8: np.ndarray, props: np.ndarray, actions: np.ndarray, H: int, T: int, device):
    """-> (z_real (H+T, D), z_imag (H+T, D)): imagined = context latents then T open-loop preds."""
    z_real = wm.encode(to_norm_pixel(frames_u8, device), torch.as_tensor(props, device=device))
    emb = z_real[:H].unsqueeze(0)                                          # (1, H, D)
    act = torch.as_tensor(actions[:H], device=device).unsqueeze(0)         # (1, H, a_dim)
    preds = []
    for t in range(T):
        pred = wm.predict(emb[:, -H:], wm.action_encoder(act[:, -H:]))[:, -1:]   # LeWM rollout step
        preds.append(pred[0, 0])
        emb = torch.cat([emb, pred], dim=1)
        if t < T - 1:
            act = torch.cat([act, torch.as_tensor(actions[H + t:H + t + 1], device=device).unsqueeze(0)], 1)
    return z_real, torch.cat([z_real[:H], torch.stack(preds)])


def main(a: argparse.Namespace) -> dict:
    device = a.device or ("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(a.seed)
    wm = load_wm(resolve_ckpt_spec(a.ckpt, a.hf_repo), device)
    dec, meta = load_decoder(a.decoder, device, z_dim=wm.z_dim)
    H, T = (a.context or wm.history_size), a.horizon
    st = np.load(resolve_state(a.state, a.hf_repo))
    px, prop, act = st["pixels"], st["proprio"], st["action"]
    if a.prefer_blocks:                       # rank many candidate segments by block pixels, keep the top n
        cands = sample_segments(st, a.n * 40, H + T, rng, tries=40000)
        score = [int(block_mask(px[e, i:i + H + T]).sum()) for e, i in cands]
        segs = [cands[j] for j in np.argsort(score)[::-1][:a.n]]
        print(f"[rollout] --prefer-blocks: {len(cands)} candidates, kept {len(segs)} with "
              f"{[score[j] for j in np.argsort(score)[::-1][:a.n]]} block px")
    else:
        segs = sample_segments(st, a.n, H + T, rng)

    rows, table = [], []
    for e, i in segs:
        fr, pr, ac = px[e, i:i + H + T], prop[e, i:i + H + T], act[e, i:i + H + T]
        z_real, z_imag = imagine(wm, fr, pr, ac, H, T, device)
        dec_imag = dec.to_uint8_hwc(z_imag)
        z_mse = ((z_imag[H:] - z_real[H:]) ** 2).mean(-1).cpu().numpy()
        px_l1 = np.abs(dec_imag[H:].astype(np.int16) - fr[H:].astype(np.int16)).mean(axis=(1, 2, 3))
        table.append((e, i, z_mse, px_l1))
        bar = np.full((fr.shape[1], 4, 3), 255, np.uint8)               # context | future marker
        top = np.concatenate([*fr[:H], bar, *fr[H:]], axis=1)
        bot = np.concatenate([*dec_imag[:H], bar, *dec_imag[H:]], axis=1)
        rows += [top, bot, np.full((6, top.shape[1], 3), 0, np.uint8)]
    sheet = np.concatenate(rows[:-1], axis=0)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    Image.fromarray(sheet).save(a.out)

    print(f"[rollout] decoder {a.decoder} (val_mse {meta.get('val_mse', float('nan')):.5f}); "
          f"context H={H}, horizon T={T}, {len(segs)} segments -> {a.out}")
    print("[rollout] per-step latent MSE (imagined vs encoded real) | pixel L1 (decoded vs real)")
    for e, i, zm, pl in table:
        print(f"  env {e} start {i:6d}: z_mse " + " ".join(f"{v:.4f}" for v in zm)
              + " | px_l1 " + " ".join(f"{v:5.1f}" for v in pl))
    if a.gif:
        import imageio
        e, i = segs[0]
        fr = px[e, i:i + H + T]
        z_real, z_imag = imagine(wm, fr, prop[e, i:i + H + T], act[e, i:i + H + T], H, T, device)
        dec_imag = dec.to_uint8_hwc(z_imag)
        imageio.mimsave(a.gif, [np.concatenate([fr[t], dec_imag[t]], axis=1) for t in range(H + T)],
                        duration=a.gif_dt, loop=0)
        print(f"[rollout] gif (real | imagined) -> {a.gif}")
    return {"out": a.out, "z_mse": [t[2] for t in table], "px_l1": [t[3] for t in table]}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--state", required=True)
    p.add_argument("--decoder", required=True)
    p.add_argument("--out", required=True, help="output PNG")
    p.add_argument("--horizon", type=int, default=8, help="imagined steps T after the context")
    p.add_argument("--context", type=int, default=0, help="context frames H (0 = ckpt history_size)")
    p.add_argument("--n", type=int, default=4, help="segments to render")
    p.add_argument("--prefer-blocks", action="store_true",
                   help="pick the n segments (of 40n episode-contiguous candidates) with the most block pixels")
    p.add_argument("--gif", default="", help="optional GIF of the first segment (real | imagined)")
    p.add_argument("--gif-dt", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--hf-repo", default=None)
    return p.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
