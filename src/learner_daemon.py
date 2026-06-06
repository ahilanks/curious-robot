"""Always-on learner daemon — the GPU half of the autonomous hardware->cloud loop.

Forever: poll the HF Hub for new collector chunks (buffers/<name>/chunk_*.npz) ->
download + archive locally (hub copies deleted by default to bound storage) -> keep a
pool of the newest --pool-cap transitions -> run offline WM+SAC fine-tune steps on it
(exact offline_train.py update path: wm_update before sac_update, fine-tune LRs) ->
upload <name>/ckpt_<global_step>.pt every --save-every steps. collect_daemon.py polls
those, probations them on the arm, and promotes or rejects.

REPLAY-RATIO GOVERNOR: an A100 outruns a ~2.6 transitions/s collector by orders of
magnitude; uncapped it would re-grind the same pool into an overfit policy between
chunk arrivals. The governor allows at most --replay-ratio gradient steps per collected
transition and sleeps when that budget is spent — keep it LOW while the pool is small.

Warm-resume: on (re)start the daemon loads the newest <name>/ckpt_* from the hub (its
own lineage), else --warmstart-name/--warmstart-step. Optimizer state restarts cold on
crash (transient loss bump, accepted). Checkpoints carry train.py's exact 8 keys.

    python src/learner_daemon.py --name auto1 --warmstart-name safe15 --warmstart-step 100000
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lewm.module import SIGReg                                    # noqa: E402
from model.state_encoder import WorldModel                        # noqa: E402
from src.offline_train import load_buffer                         # noqa: E402
from src.train import (Actor, TwinQ, collapse_metrics, resolve_ckpt,   # noqa: E402
                       sac_update, save_and_upload, wm_update)


def log(out_dir, msg, **kv):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}" + ("  " + json.dumps(kv) if kv else ""),
          flush=True)
    with open(out_dir / "daemon.jsonl", "a") as f:
        f.write(json.dumps({"t": time.time(), "msg": msg, **kv}) + "\n")


def sync_chunks(repo, name, pool_dir, token, delete_hub, out_dir):
    """Download chunks we don't have locally; optionally delete the hub copies after a
    verified download (the pod-local archive becomes the source of truth). Returns
    (n_new_files, n_new_transitions) — the governor budget runs on SESSION-LOCAL
    counters so archive pruning / restarts can't starve it."""
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=token)
    prefix = f"buffers/{name}/chunk_"
    new, new_tr = 0, 0
    for f in sorted(x for x in api.list_repo_files(repo) if x.startswith(prefix)):
        local = pool_dir / Path(f).name
        if local.exists():
            continue
        p = hf_hub_download(repo_id=repo, filename=f, token=token)
        # hf cache entries are RELATIVE symlinks into blobs/ — moving one out of the
        # cache dangles it. Copy through the resolved blob; never mutate the cache.
        import shutil
        shutil.copyfile(os.path.realpath(p), local)
        new += 1
        try:
            new_tr += int(local.stem.split("_")[-1])
        except ValueError:
            new_tr += 1000
        if delete_hub:
            try:
                api.delete_file(f, repo_id=repo)
            except Exception as ex:
                log(out_dir, "hub delete failed (non-fatal)", file=f, err=str(ex)[:100])
    return new, new_tr


def prune_archive(pool_dir, archive_cap, out_dir):
    """Bound the pod-local archive: keep the newest chunks up to archive_cap transitions,
    unlink the rest (oldest first). The training pool is a subset of the archive anyway."""
    files = sorted(pool_dir.glob("chunk_*.npz"), reverse=True)
    total = 0
    for f in files:
        try:
            n = int(f.stem.split("_")[-1])
        except ValueError:
            n = 1000
        total += n
        if total > archive_cap:
            f.unlink(missing_ok=True)
    if total > archive_cap:
        log(out_dir, "archive pruned", kept_cap=archive_cap)


def pool_paths(pool_dir, cap):
    """Newest chunks whose cumulative transition count fits the pool cap (filename
    carries the count: chunk_<ts>_<seq>_<n>.npz)."""
    files = sorted(pool_dir.glob("chunk_*.npz"), reverse=True)     # newest first (ts in name)
    picked, total = [], 0
    for f in files:
        try:
            n = int(f.stem.split("_")[-1])
        except ValueError:
            n = 1000
        if total + n > cap and picked:
            break
        picked.append(f); total += n
    return list(reversed(picked)), total


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass
    repo = args.hf_repo or os.environ.get("HF_UPLOAD_REPO_ID")
    token = os.environ.get("HF_TOKEN")
    out_dir = Path("runs") / f"{args.name}_learner"
    pool_dir = out_dir / "pool"
    pool_dir.mkdir(parents=True, exist_ok=True)
    log(out_dir, "learner boot", device=str(device), name=args.name)

    # --- warm-resume own lineage from the hub, else the warmstart run --------------
    from huggingface_hub import HfApi, hf_hub_download
    own = [f for f in HfApi(token=token).list_repo_files(repo)
           if f.startswith(f"{args.name}/ckpt_") and f.endswith(".pt")]
    if own:
        path = hf_hub_download(repo_id=repo, filename=sorted(own)[-1], token=token)
        global_step = int(sorted(own)[-1].split("ckpt_")[1].split(".pt")[0])
    else:
        path = resolve_ckpt(None, args.warmstart_name, args.warmstart_step, repo)
        global_step = 0
    ck = torch.load(path, map_location=device, weights_only=False)
    src_args = dict(ck.get("args", {}))
    H = int(src_args.get("history_size", 3))
    action_block = int(src_args.get("action_block", 5))
    a_dim = 6 * action_block
    h_fwd = int(ck.get("h_fwd", 1))
    log(out_dir, "warm-start", path=Path(path).name, global_step=global_step,
        H=H, h_fwd=h_fwd)

    wm = WorldModel(n_dof=6, action_block=action_block, history_size=H,
                    dropout=float(src_args.get("wm_dropout", 0.1))).to(device)
    wm.load_state_dict(ck["wm"]); wm.eval()
    sigreg = SIGReg(knots=17, num_proj=1024).to(device)
    actor = Actor(wm.z_dim, a_dim).to(device); actor.load_state_dict(ck["actor"])
    critic = TwinQ(wm.z_dim, a_dim).to(device); critic.load_state_dict(ck["critic"])
    critic_tgt = TwinQ(wm.z_dim, a_dim).to(device); critic_tgt.load_state_dict(ck["critic_tgt"])
    for p_ in critic_tgt.parameters():
        p_.requires_grad_(False)
    log_alpha = ck["log_alpha"].to(device)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    wm_opt = torch.optim.AdamW([p_ for p_ in wm.parameters() if p_.requires_grad],
                               lr=args.wm_lr, weight_decay=1e-3)

    args.start_steps = 0                      # gates read by sac_update
    args.total_steps = 10_000_000             # per_beta anneal horizon (effectively flat)
    saved_args = {**src_args, **{k: v for k, v in vars(args).items()
                                 if not k.startswith("_")},
                  "history_size": H, "action_block": action_block}

    def ckpt_state(step):
        return {"step": step, "wm": wm.state_dict(), "actor": actor.state_dict(),
                "critic": critic.state_dict(), "critic_tgt": critic_tgt.state_dict(),
                "log_alpha": log_alpha.detach().cpu(), "h_fwd": h_fwd, "args": saved_args}

    # =============================================================== forever loop
    buf, pool_n, seen_transitions = None, 0, 0      # governor counters are session-local:
    steps_done_for_budget = 0                       # restart -> budget refills with next chunk
    last_wm = last_sac = last_zb = None
    t0 = time.time()
    while True:
        # --- sync + (re)build the pool when new data lands -------------------------
        try:
            new, new_tr = sync_chunks(repo, args.name, pool_dir, token,
                                      not args.keep_hub_chunks, out_dir)
        except Exception as ex:
            log(out_dir, "chunk sync failed; retrying soon", err=str(ex)[:120])
            time.sleep(60)
            continue
        seen_transitions += new_tr
        if new or buf is None:
            prune_archive(pool_dir, args.archive_cap, out_dir)
            paths, pool_n = pool_paths(pool_dir, args.pool_cap)
            if not paths:
                log(out_dir, "no chunks yet — waiting for the collector")
                time.sleep(args.idle_sleep)
                continue
            buf = load_buffer([str(f) for f in paths], H, args.h_fwd_max, device)
            log(out_dir, "pool rebuilt", chunks=len(paths), transitions=buf.total,
                seen_session=seen_transitions, new_chunks=new)

        # --- replay-ratio governor ---------------------------------------------------
        budget = args.replay_ratio * seen_transitions - steps_done_for_budget
        if budget <= 0:
            log(out_dir, "governor: budget spent — idling", seen=seen_transitions,
                steps=steps_done_for_budget)
            time.sleep(args.idle_sleep)
            continue

        # --- one training cycle (then re-poll) ----------------------------------------
        cycle = int(min(budget, args.cycle_steps))
        for _ in range(cycle):
            if global_step % args.wm_update_every == 0:
                batch = buf.sample_wm(args.wm_batch_size, H + h_fwd)
                if batch is not None:
                    wm.train()
                    last_wm = wm_update(wm, sigreg, wm_opt, batch, H, h_fwd,
                                        args.gamma_wm, args.sigreg_weight, device)
                    wm.eval()
            res = sac_update(buf, wm, actor, critic, critic_tgt, actor_opt, critic_opt,
                             log_alpha, args, global_step, device)
            if res is not None:
                last_sac = (res["critic_loss"], res["actor_loss"]); last_zb = res["zb"]
            global_step += 1
            steps_done_for_budget += 1
            if global_step % args.save_every == 0:
                save_and_upload(ckpt_state(global_step), out_dir, global_step, repo,
                                args.name, not args.no_hf, args.keep_local_ckpts)
                d = {"step": global_step, "pool": buf.total, "seen": seen_transitions}
                if last_wm is not None:
                    d.update(pred_loss=last_wm[0], sigreg=last_wm[1])
                if last_sac is not None:
                    d.update(critic_loss=last_sac[0], actor_loss=last_sac[1])
                if last_zb is not None:
                    zs, er, fc = collapse_metrics(last_zb)
                    d.update(z_std=zs, eff_rank=er, feat_corr=fc)
                d["sps"] = round(global_step / max(time.time() - t0, 1e-9), 1)
                log(out_dir, "ckpt", **{k: (round(v, 4) if isinstance(v, float) else v)
                                        for k, v in d.items()})
        if args.max_steps and global_step >= args.max_steps:   # bounded test runs
            save_and_upload(ckpt_state(global_step), out_dir, global_step, repo,
                            args.name, not args.no_hf, args.keep_local_ckpts)
            log(out_dir, "max-steps reached — exiting", step=global_step)
            return


def parse_args():
    p = argparse.ArgumentParser(description="always-on offline learner over HF chunks")
    p.add_argument("--name", default="auto1")
    p.add_argument("--warmstart-name", default="safe15")
    p.add_argument("--warmstart-step", type=int, default=100000)
    # pool / governor
    p.add_argument("--pool-cap", type=int, default=60_000, help="transitions in RAM (~150KB each)")
    p.add_argument("--archive-cap", type=int, default=300_000,
                   help="transitions kept on pod disk (~45GB); oldest pruned beyond this")
    p.add_argument("--replay-ratio", type=float, default=16.0,
                   help="max gradient steps per collected transition; keep LOW early")
    p.add_argument("--cycle-steps", type=int, default=500, help="steps between hub polls")
    p.add_argument("--idle-sleep", type=float, default=30.0)
    p.add_argument("--keep-hub-chunks", action="store_true",
                   help="don't delete hub chunks after download (default: delete; the "
                        "pod-local pool/ archive is the source of truth)")
    p.add_argument("--max-steps", type=int, default=0, help=">0: exit after N steps (tests)")
    # learning knobs (offline fine-tune defaults; names match train.py/offline_train.py)
    p.add_argument("--wm-update-every", type=int, default=4)
    p.add_argument("--wm-batch-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--updates-per-step", type=int, default=1)
    p.add_argument("--wm-lr", type=float, default=1e-5)
    p.add_argument("--actor-lr", type=float, default=1e-4)
    p.add_argument("--critic-lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument("--gamma-wm", type=float, default=0.95)
    p.add_argument("--sigreg-weight", type=float, default=0.3)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--per-alpha", type=float, default=0.6)
    p.add_argument("--per-beta-start", type=float, default=0.4)
    p.add_argument("--per-priority", choices=["curiosity", "td"], default="td",
                   help="td: priorities self-adapt (chunks carry none)")
    p.add_argument("--h-fwd-max", type=int, default=1, help="pool slack sizing")
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--keep-local-ckpts", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-hf", action="store_true", help="never upload (local tests)")
    p.add_argument("--hf-repo", default=None)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
