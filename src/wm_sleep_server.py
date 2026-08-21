"""GPU sleep server for the wr goal-explore split (2026-08-20): the arm collects on the
Mac; the sleeps run HERE, in parallel, on real data — the distributed mapping of the
wake/sleep doctrine (the arm never pauses for consolidation).

LOOP: poll the HF repo for <collector>/state_latest.npz (the Mac's --save-state-every
uploads) -> on a new commit, download and restore into a fresh ring -> run ONE
consolidation sleep (wm_update passes over the buffer, flatline-stopped like the
campaign's sleeps) -> push <name>/wm_live.pt back to the repo. The Mac's --pull-wm
notices the commit and hot-swaps between decisions (probe-gated).

The ENCODER STAYS FROZEN here by default: the collector's latent geometry (goal dists,
reach eps, curricula state) must stay valid mid-run; a server-side thaw would shift the
space under the arm's feet between swaps. Predictor + action_encoder + pred_proj train.

Pairs with (Mac):
    bash run_hw_wr_sleepret2.sh <name> <steps>   with LEARNER=<this --name> exported

RunPod (same repo, .env with HF_TOKEN + HF_UPLOAD_REPO_ID, bash setup.sh):
    python src/wm_sleep_server.py --collector hw_wrs2_c --name wrs2_sleep \
        --init-ckpt wr_sleepret2/ckpt_0200000.pt
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lewm.module import SIGReg                                     # noqa: E402
from model.state_encoder import WorldModel, pred_dims_from_args    # noqa: E402
from src.train import ReplayBuffer, load_state_snapshot, wm_update  # noqa: E402


def log(msg: str) -> None:
    print(f"[sleep-server] {msg}", flush=True)


def main(a: argparse.Namespace) -> None:
    from huggingface_hub import HfApi, hf_hub_download
    repo = a.repo or os.environ.get("HF_UPLOAD_REPO_ID")
    if not repo:
        raise SystemExit("no HF repo: set HF_UPLOAD_REPO_ID or --repo")
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    device = a.device or ("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")

    # --- weights: boot from --init-ckpt (repo-relative or local), resume own wm_live if present
    init = a.init_ckpt if os.path.exists(a.init_ckpt) else \
        hf_hub_download(repo, a.init_ckpt, token=os.environ.get("HF_TOKEN"))
    ck = torch.load(init, map_location=device, weights_only=False)
    ck_args = ck.get("args", {})
    get = ck_args.get if isinstance(ck_args, dict) else (lambda k, d=None: getattr(ck_args, k, d))
    H, h_fwd = int(get("history_size", 3) or 3), 1
    wm = WorldModel(n_dof=6, action_block=int(get("action_block", 5) or 5), history_size=H,
                    dropout=float(get("wm_dropout", 0.1) or 0.1),
                    use_proprio=not bool(get("no_proprio", False)),
                    **pred_dims_from_args(ck_args)).to(device)
    wm.load_state_dict(ck["wm"])
    try:                                              # resume own lineage over a restart
        own = hf_hub_download(repo, f"{a.name}/wm_live.pt", token=os.environ.get("HF_TOKEN"))
        wm.load_state_dict(torch.load(own, map_location=device, weights_only=False)["wm"])
        log(f"resumed own lineage from {a.name}/wm_live.pt")
    except Exception:
        log(f"fresh lineage from {init}")
    wm.encoder.requires_grad_(False)                  # frozen latent geometry (see module docstring)
    train_params = [p for p in wm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(train_params, lr=a.lr)
    sigreg = SIGReg(knots=17, num_proj=1024).to(device)
    beta = float(get("sigreg_weight", 0.09) or 0.09)
    pertimestep = bool(get("sigreg_pertimestep", True))
    gamma_wm = float(get("gamma_wm", 0.95) or 0.95)
    log(f"device={device} | trainable {sum(p.numel() for p in train_params)/1e6:.2f}M "
        f"(encoder frozen) | watching {repo}/{a.collector}/state_latest.npz")

    state_path_in_repo = f"{a.collector}/state_latest.npz"
    last_oid, sleep_i = None, 0
    while True:
        try:
            info = api.get_paths_info(repo, [state_path_in_repo], expand=True)
            oid = (info[0].last_commit.oid if info and info[0].last_commit else None)
            if oid is None or oid == last_oid:
                time.sleep(a.poll_secs)
                continue
            sp = hf_hub_download(repo, state_path_in_repo, token=os.environ.get("HF_TOKEN"))
            z = np.load(sp)
            n_envs, cap = z["head"].shape[0], z["pixels"].shape[1]
            buf = ReplayBuffer(n_envs, cap, z["pixels"].shape[2], z["action"].shape[-1],
                               z["proprio"].shape[-1], device,
                               goal_explore="goal_px" in z.files)
            if not load_state_snapshot(buf, sp):
                log("state restore failed — waiting for the next snapshot")
                last_oid = oid
                continue
            collector_step = int(z["step"])

            # --- one consolidation sleep: wm_update passes, flatline-stopped ---
            t0, losses = time.time(), []
            wm.train()
            for g in range(a.max_grad_steps):
                batch = buf.sample_wm(a.batch, H + h_fwd, "uniform")
                if batch is None:
                    break
                losses.append(wm_update(wm, sigreg, opt, batch, H, h_fwd, gamma_wm, beta,
                                        device, pertimestep=pertimestep)[0])
                w = a.flatline_window
                if len(losses) >= 2 * w:
                    older, newer = np.mean(losses[-2*w:-w]), np.mean(losses[-w:])
                    if abs(older - newer) / max(abs(older), 1e-9) < a.flatline_tol:
                        break
            wm.eval()
            sleep_i += 1
            log(f"sleep #{sleep_i} on state@{collector_step}: {len(losses)} grad steps, "
                f"pred {losses[0]:.4f} -> {losses[-1]:.4f} in {time.time()-t0:.0f}s")

            # --- push wm_live.pt (single-file commit; the Mac's puller sees the new oid) ---
            out = Path(f"runs/{a.name}"); out.mkdir(parents=True, exist_ok=True)
            local = out / "wm_live.pt"
            torch.save({"wm": wm.state_dict(), "collector_step": collector_step,
                        "sleep": sleep_i, "pred": float(losses[-1])}, local)
            api.upload_file(path_or_fileobj=str(local), repo_id=repo,
                            path_in_repo=f"{a.name}/wm_live.pt")
            if a.keep_every > 0 and sleep_i % a.keep_every == 0:      # local rollback points
                torch.save({"wm": wm.state_dict()}, out / f"wm_sleep_{sleep_i:04d}.pt")
            log(f"pushed {a.name}/wm_live.pt (sleep {sleep_i}, pred {losses[-1]:.4f})")
            last_oid = oid
        except KeyboardInterrupt:
            log("stopped by user")
            return
        except Exception as e:                        # network/HF blips must not kill the server
            log(f"loop error (retrying in {a.poll_secs}s): {type(e).__name__}: {e}")
            time.sleep(a.poll_secs)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--collector", required=True, help="Mac run name whose state_latest.npz to consume")
    p.add_argument("--name", required=True, help="learner id: uploads <name>/wm_live.pt")
    p.add_argument("--init-ckpt", required=True,
                   help="boot weights: local path or repo-relative (e.g. wr_sleepret2/ckpt_0200000.pt)")
    p.add_argument("--repo", default=None, help="HF repo (default $HF_UPLOAD_REPO_ID)")
    p.add_argument("--poll-secs", type=float, default=45.0)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-5, help="wm_lr; predictor-side only (encoder frozen)")
    p.add_argument("--max-grad-steps", type=int, default=600,
                   help="cap per sleep (campaign sleeps converged in 200-550)")
    p.add_argument("--flatline-window", type=int, default=50)
    p.add_argument("--flatline-tol", type=float, default=0.01)
    p.add_argument("--keep-every", type=int, default=5, help="save a local rollback ckpt every K sleeps")
    p.add_argument("--device", default=None)
    main(p.parse_args())
