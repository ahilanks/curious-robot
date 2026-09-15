"""Fit the post-hoc pixel decoder (model/decoder.py, LeWM App. D) on a run's saved state.

Frames come from a --save-state snapshot (state_latest.npz: the raw replay ring, real wrist
photos on a hardware run); latents come from the checkpoint's FROZEN encoder (wm.encode
under no_grad — the WM is never touched). Objective: plain pixel MSE against frame/255, the
standard post-hoc probe for JEPA-family models (LeWM Fig. 8: "reconstruction is never used
during training, the decoder recovers the scene from the 192-d latent"). Outputs the decoder
ckpt plus a real-vs-recon contact sheet so quality is eyeballable before trusting the
dashboard's "decoder's eye" row.

    python src/train_decoder.py --ckpt runs/<run>/ckpt_XXXXXXX.pt \
        --state runs/<run>/state_latest.npz --out runs/decoder_wrs2.pt

CHANGED 2026-09-14:
  * --ckpt / --state accept HF specs when the local path is missing: "<run>/ckpt_0004000.pt",
    "<run>" (latest ckpt of that run), "<run>/state_latest.npz" or "<run>" for the state.
  * latents are encoded ONCE and cached (the encoder is frozen, so z per frame is constant);
    the decoder then trains on (z, frame) pairs — the ViT forward no longer sits in the
    training loop (CPU/MPS boxes: minutes instead of hours).
  * real proprio from the state is used when the checkpoint's encoder consumes it.
  * --upload pushes the decoder (+ sheet) to $HF_UPLOAD_REPO_ID as <run>/<out basename>.

Then pass `--decoder runs/<out>.pt` (or DECODER=... to run_hw_wr_sleepret2.sh) to a
--live-view train.py run, or render open-loop imagined rollouts with viz_decoder_rollout.py.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.state_encoder import WorldModel, pred_dims_from_args   # noqa: E402
from model.decoder import LatentDecoder, ConvLatentDecoder, load_decoder  # noqa: E402
from src.train import to_norm_pixel, resolve_ckpt                 # noqa: E402

try:                                                              # .env: HF_TOKEN / HF_UPLOAD_REPO_ID
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass


# ------------------------------------------------------------------ inputs (local or HF)
def resolve_state(spec: str, hf_repo: str | None = None) -> str:
    """Local path if it exists, else download <run>/state_latest.npz from the HF repo."""
    if Path(spec).exists():
        return spec
    spec = spec.removeprefix("runs/")                    # runs/<run>/... -> HF <run>/...
    repo = hf_repo or os.environ.get("HF_UPLOAD_REPO_ID")
    if not repo:
        raise SystemExit(f"{spec} not found locally and no HF repo (set HF_UPLOAD_REPO_ID)")
    from huggingface_hub import hf_hub_download
    fn = spec if spec.endswith(".npz") else f"{spec}/state_latest.npz"
    print(f"[hf] downloading {fn} from {repo}", flush=True)
    return hf_hub_download(repo_id=repo, filename=fn, token=os.environ.get("HF_TOKEN"))


def resolve_ckpt_spec(spec: str, hf_repo: str | None = None) -> str:
    """Local path if it exists, else "<run>/ckpt_XXXXXXX.pt" or "<run>" (latest) from HF."""
    if Path(spec).exists():
        return spec
    spec = spec.removeprefix("runs/")                    # runs/<run>/... -> HF <run>/...
    if spec.endswith(".pt"):
        repo = hf_repo or os.environ.get("HF_UPLOAD_REPO_ID")
        if not repo:
            raise SystemExit(f"{spec} not found locally and no HF repo (set HF_UPLOAD_REPO_ID)")
        from huggingface_hub import hf_hub_download
        print(f"[hf] downloading {spec} from {repo}", flush=True)
        return hf_hub_download(repo_id=repo, filename=spec, token=os.environ.get("HF_TOKEN"))
    return resolve_ckpt(None, name=spec, hf_repo=hf_repo)


def load_wm(ckpt_path: str, device) -> WorldModel:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    ck_args = ck.get("args", {})
    get = ck_args.get if isinstance(ck_args, dict) else (lambda k, d=None: getattr(ck_args, k, d))
    wm = WorldModel(n_dof=6, action_block=int(get("action_block", 5) or 5),
                    history_size=int(get("history_size", 3) or 3),
                    dropout=float(get("wm_dropout", 0.1) or 0.1),
                    use_proprio=not bool(get("no_proprio", False)),
                    **pred_dims_from_args(ck_args)).to(device)
    wm.load_state_dict(ck["wm"])
    wm.eval().requires_grad_(False)
    return wm


def load_frames(state_path: str, max_frames: int):
    """-> (frames (N,H,W,3) uint8, proprio (N,P) float32) over the FILLED extent of every env ring."""
    st = np.load(state_path)
    px, count = st["pixels"], st["count"]                  # (n_envs, cap, H, W, 3) uint8
    prop = st["proprio"] if "proprio" in st.files else None
    frames = np.concatenate([px[e, :int(count[e])] for e in range(px.shape[0])])
    props = (np.concatenate([prop[e, :int(count[e])] for e in range(px.shape[0])])
             if prop is not None else np.zeros((len(frames), 18), np.float32))
    if len(frames) > max_frames:                            # uniform thin to bound memory/time
        keep = np.linspace(0, len(frames) - 1, max_frames).astype(int)
        frames, props = frames[keep], props[keep]
    return frames, props.astype(np.float32)


@torch.no_grad()
def encode_all(wm: WorldModel, frames: np.ndarray, props: np.ndarray, device, batch: int = 64,
               log_every: int = 20) -> torch.Tensor:
    """Frozen-encoder latents for every frame, computed ONCE (cached for the whole fit)."""
    zs, t0 = [], time.time()
    for i in range(0, len(frames), batch):
        zs.append(wm.encode(to_norm_pixel(frames[i:i + batch], device),
                            torch.as_tensor(props[i:i + batch], device=device)).cpu())
        if (i // batch) % log_every == 0:
            print(f"[decoder] encode {min(i + batch, len(frames))}/{len(frames)} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return torch.cat(zs)


def frames_to_target(batch_u8: np.ndarray, device) -> torch.Tensor:
    t = torch.as_tensor(np.ascontiguousarray(batch_u8), device=device)
    return t.permute(0, 3, 1, 2).float() / 255.0


def contact_sheet(real: np.ndarray, rec: np.ndarray, path: str) -> None:
    """Top = real, bottom = decode(encode(real)) — the preserved-information picture."""
    sheet = np.concatenate([np.concatenate(list(real), axis=1),
                            np.concatenate(list(rec), axis=1)], axis=0)
    try:
        from PIL import Image
        Image.fromarray(sheet).save(path)
    except ImportError:
        import cv2
        cv2.imwrite(path, sheet[..., ::-1])


def main(a: argparse.Namespace) -> dict:
    device = a.device or ("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    ckpt_path, state_path = resolve_ckpt_spec(a.ckpt, a.hf_repo), resolve_state(a.state, a.hf_repo)
    wm = load_wm(ckpt_path, device)
    print(f"[decoder] encoder from {ckpt_path} (z_dim={wm.z_dim}, frozen)", flush=True)

    frames, props = load_frames(state_path, a.max_frames)
    n_val = max(8, int(a.val_frac * len(frames)))
    val_idx = np.linspace(0, len(frames) - 1, n_val).astype(int)   # val spans the whole session
    train_idx = np.setdiff1d(np.arange(len(frames)), val_idx)
    print(f"[decoder] {len(train_idx)} train / {len(val_idx)} val frames from {state_path}", flush=True)

    z_all = encode_all(wm, frames, props, device, batch=a.encode_batch)   # CACHED latents
    z_all = z_all.to(device)

    if a.arch == "conv":
        dec = ConvLatentDecoder(z_dim=wm.z_dim).to(device)
    else:
        dec = LatentDecoder(z_dim=wm.z_dim, hidden=a.hidden, depth=a.depth, heads=a.heads,
                            n_mem=a.n_mem, self_attn=a.self_attn).to(device)
    if a.init:
        prev, _ = load_decoder(a.init, device, z_dim=wm.z_dim)
        dec.load_state_dict(prev.state_dict())
        dec.train().requires_grad_(True)
        print(f"[decoder] continuing from {a.init}", flush=True)
    n_params = sum(p.numel() for p in dec.parameters())
    print(f"[decoder] arch={dec.config()} params={n_params / 1e6:.2f}M device={device}", flush=True)

    opt = torch.optim.AdamW(dec.parameters(), lr=a.lr, weight_decay=a.wd)
    warm = min(100, max(1, a.steps // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, a.steps - warm))))

    def evaluate() -> float:
        """Pixel MSE over EVERY val frame (they stride the whole session). FIXED 2026-09-15: the old
        `k=64` cap sliced `val_idx[:batch]` = the first 128 frames = the early session only, which
        read ~30% low (hw_wrs2_c: 0.0094 logged vs 0.0138 over all 345 val frames)."""
        dec.eval()
        with torch.no_grad():
            tot, n = 0.0, 0
            for i in range(0, len(val_idx), a.batch):
                idx = val_idx[i:i + a.batch]
                tot += float(torch.nn.functional.mse_loss(
                    dec(z_all[idx]), frames_to_target(frames[idx], device), reduction="sum"))
                n += len(idx) * 3 * dec.out_hw * dec.out_hw
        dec.train()
        return tot / max(n, 1)

    def save(step: int, vmse: float) -> None:
        torch.save({"decoder": dec.state_dict(), "z_dim": wm.z_dim, "arch": dec.config(),
                    "val_mse": vmse, "steps": step, "n_frames": int(len(frames)),
                    "ckpt": str(ckpt_path), "state": str(state_path)}, a.out)

    t0 = time.time()
    dec.train()
    for step in range(1, a.steps + 1):
        idx = train_idx[np.random.randint(0, len(train_idx), size=a.batch)]
        loss = torch.nn.functional.mse_loss(dec(z_all[idx]), frames_to_target(frames[idx], device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % a.log_every == 0 or step == a.steps:
            print(f"[decoder] step {step}/{a.steps} mse={loss.item():.5f} "
                  f"lr={sched.get_last_lr()[0]:.2e} ({(time.time() - t0) / step * 1000:.0f} ms/step)",
                  flush=True)
        if a.save_every and step % a.save_every == 0 and step != a.steps:
            v = evaluate()
            save(step, v)
            print(f"[decoder]   checkpoint @ {step}: val mse={v:.5f} -> {a.out}", flush=True)

    vmse = evaluate()
    dec.eval()
    save(a.steps, vmse)
    print(f"[decoder] val mse={vmse:.5f} -> {a.out}", flush=True)

    k = min(8, len(val_idx))
    rec = dec.to_uint8_hwc(z_all[val_idx[:k]])
    sheet_path = str(Path(a.out).with_suffix(".sheet.png"))
    contact_sheet(frames[val_idx[:k]], rec, sheet_path)
    print(f"[decoder] contact sheet (top real / bottom recon) -> {sheet_path}", flush=True)

    if a.upload:
        repo = a.hf_repo or os.environ.get("HF_UPLOAD_REPO_ID")
        run = a.upload_run or Path(ckpt_path).parent.name
        if not repo or not os.environ.get("HF_TOKEN"):
            print("[hf] upload skipped: need HF_UPLOAD_REPO_ID + HF_TOKEN", flush=True)
        else:
            from huggingface_hub import HfApi
            api = HfApi(token=os.environ["HF_TOKEN"])
            for local in (a.out, sheet_path):
                dst = f"{run}/{Path(local).name}"
                api.upload_file(path_or_fileobj=local, repo_id=repo, path_in_repo=dst)
                print(f"[hf] uploaded {dst} -> {repo}", flush=True)
    return {"val_mse": vmse, "out": a.out, "sheet": sheet_path, "n_params": n_params}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ckpt", required=True, help="run checkpoint path, or HF '<run>/ckpt_X.pt' / '<run>'")
    p.add_argument("--state", required=True, help="state_latest.npz path, or HF '<run>'")
    p.add_argument("--out", required=True, help="output decoder .pt path")
    p.add_argument("--arch", choices=["lewm", "conv"], default="lewm")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--n-mem", type=int, default=1, help="latent memory tokens for cross-attn (LeWM: 1)")
    p.add_argument("--self-attn", action="store_true", help="add query self-attention (not in LeWM)")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--encode-batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--max-frames", type=int, default=20000)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--save-every", type=int, default=500, help="periodic ckpt (0 = only at end)")
    p.add_argument("--init", default="", help="continue from an existing decoder ckpt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--hf-repo", default=None, help="HF repo (default $HF_UPLOAD_REPO_ID)")
    p.add_argument("--upload", action="store_true", help="push decoder + sheet to HF as <run>/<out name>")
    p.add_argument("--upload-run", default="", help="HF folder for --upload (default: ckpt's run dir)")
    return p.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
