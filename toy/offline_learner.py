"""Train run_selfplay's latent learner offline on saved runs' data: what makes the world model predict unseen demos?

Same model and loss as the online learner (JEPA on the 8-D observation, MSE to the un-detached target embedding +
lambda SIGReg over groups of 64 embeddings, AdamW, clip 1), trained on minibatches of the runs' replay.pt windows;
--horizon H > 1 trains on open-loop latent rollouts of H steps (weights gamma^k, normalised, as in the main README's
L_wm). Writes <out>/<name>/final.pt (+ a replay.pt link to the first data run, the learner's "own" data) for
score_learner.py.

    python toy/offline_learner.py --data toy/runs/pusht/ptf_spep_s0 --name off_spep_s0 [--steps 20000 --sigreg 0.09 --horizon 1]

--encoder-from RUN copies RUN's learner encoder and freezes it, so only the predictor learns (and SIGReg, which acts
on the encoder, is off): data sets compared through one shared latent.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from push_t import PushT
from run_selfplay import LatentWM, SIGReg

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_windows(runs):
    """Episodes from replay.pt: states (N, T+1, 7), actions (N, T, 2)."""
    S, A = [], []
    for r in runs:
        x = torch.load(Path(r) / "replay.pt")
        S.append(x["state"].permute(0, 2, 1, 3).reshape(-1, *x["state"].shape[1:2], 7))
        A.append(x["act"].permute(0, 2, 1, 3).reshape(-1, *x["act"].shape[1:2], 2))
    return torch.cat(S).float(), torch.cat(A).float()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", nargs="+", required=True, help="run dirs whose replay.pt is the training data")
    p.add_argument("--name", required=True)
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "runs" / "pusht" / "offline"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--batch", type=int, default=4096, help="windows per step (a multiple of 64)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-3)
    p.add_argument("--sigreg", type=float, default=0.09)
    p.add_argument("--horizon", type=int, default=1, help="open-loop latent rollout length of the loss")
    p.add_argument("--gamma", type=float, default=0.95, help="discount on far rollout steps")
    p.add_argument("--z-dim", type=int, default=32)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--encoder-from", default=None, help="run dir whose (frozen) learner encoder to use")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    out = Path(args.out) / args.name
    out.mkdir(parents=True, exist_ok=True)
    S, A = load_windows(args.data)
    S, A = S.to(DEV), A.to(DEV)
    N, T1, H = S.shape[0], S.shape[1], args.horizon
    m = LatentWM(PushT.obs_dim, PushT.act_dim, args.z_dim, args.hidden).to(DEV)
    sig = SIGReg().to(DEV)
    if args.encoder_from:
        src = torch.load(Path(args.encoder_from) / "final.pt", weights_only=False)["learner"]
        m.enc.load_state_dict({k[4:]: v for k, v in src.items() if k.startswith("enc.")})
        m.enc.requires_grad_(False)
    opt = torch.optim.AdamW([q for q in m.parameters() if q.requires_grad], lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps, eta_min=args.lr * 0.01)
    w = torch.tensor([args.gamma ** k for k in range(1, H + 1)], device=DEV)
    w = w / w.sum()
    t0 = time.time()
    for it in range(args.steps):
        ep = torch.randint(0, N, (args.batch,), device=DEV)
        t = torch.randint(0, T1 - H, (args.batch,), device=DEV)
        steps = t[:, None] + torch.arange(H + 1, device=DEV)
        z = m.enc(PushT.obs_from_state(S[ep[:, None], steps]))                  # (B, H+1, Z)
        a = A[ep[:, None], steps[:, :-1]]                                     # (B, H, 2)
        zh, errs = z[:, 0], []
        for k in range(H):
            zh = zh + m.pred(torch.cat([zh, a[:, k]], -1))
            errs.append((zh - z[:, k + 1]).pow(2).mean(-1).mean())
        pred = (torch.stack(errs) * w).sum()
        sr = sig(z.reshape(-1, 64, z.shape[-1])) if not args.encoder_from else torch.zeros((), device=DEV)
        opt.zero_grad(set_to_none=True)
        (pred + args.sigreg * sr).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % 5000 == 0 or it == args.steps - 1:
            print(f"[{args.name}] {it:6d}  pred {pred.item():.4f}  sigreg {sr.item():.3f}  {time.time() - t0:.0f}s", flush=True)
    torch.save({"args": {**vars(args), "wm_hidden": args.hidden}, "learner": m.state_dict()}, out / "final.pt")
    json.dump(vars(args), open(out / "config.json", "w"), indent=2)
    link = out / "replay.pt"
    if not link.exists():
        link.symlink_to(Path(args.data[0]).resolve() / "replay.pt")


if __name__ == "__main__":
    main()
