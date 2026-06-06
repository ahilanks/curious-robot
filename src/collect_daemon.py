"""24/7 collector daemon — the Mac half of the autonomous hardware->cloud loop.

Runs the frozen acting loop on the physical SO-ARM101 indefinitely:
  - dumps transitions in CHUNKS (save_buffer npz format) and uploads them to the HF Hub
    on a background thread (disk-bounded: local files deleted on confirmed upload,
    oldest dropped if the upload backlog grows — never fill the disk)
  - polls the Hub for new checkpoints from learner_daemon.py and HOT-SWAPS the policy
    between decisions — but only after an on-arm ACCEPTANCE PROBATION: the candidate
    drives ~30 watched decisions first; any watchdog trip (press, saturated actions,
    real fights, NaNs) rejects it and reverts to the last-known-good CHAMPION
    (runs/<name>/champion.pt — the ratchet that makes a bad upload recoverable)
  - temp gate: polls servo temperatures (reg 63) every few decisions; above --temp-gate
    it parks the arm in a gravity-stable fold, drops torque, and waits until
    --temp-resume before continuing (hobby servos are the 24/7 physical ceiling)
  - press watchdog: sustained full-effort + no-motion (the policy pinning itself against
    a stop scores r_safe=0 by spec, so the reward will never fix it) -> retreat toward
    joint midpoints for a few decisions, then resume

The acting path mirrors train.py's frozen mode exactly (encode -> sample -> step_block ->
r = lambda_safe*r_safe + lambda_cur*log1p(r_cur) with r_cur from the CURRENT wm); reward
constants default to the frozen campaign config (round_runbook.md). Chunks are the exact
inverse of offline_train.load_buffer. SIGINT dumps the partial chunk and exits with the
arm holding (same convention as train.py).

    SOARM_PORT=... SOARM_CALIB=so101_calib.json python src/collect_daemon.py \
        --name auto1 --warmstart-name safe15 --warmstart-step 100000
"""
import argparse
import json
import os
import queue
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.state_encoder import WorldModel                       # noqa: E402
from src.train import Actor, curiosity_reward, encode_obs, resolve_ckpt   # noqa: E402
from env.hardware_env import JOINT_HIGH, JOINT_LOW               # noqa: E402

# Gravity-stable fold for torque-off rest (≈ where the arm settles when limp; pan/roll/
# gripper keep their current values). Verify once on the arm before unattended runs.
PARK_SHOULDER, PARK_ELBOW, PARK_WRIST = -1.70, -1.60, -1.00


def log(out_dir, msg, **kv):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line + ("  " + json.dumps(kv) if kv else ""), flush=True)
    with open(out_dir / "daemon.jsonl", "a") as f:
        f.write(json.dumps({"t": time.time(), "msg": msg, **kv}) + "\n")


# ------------------------------------------------------------------ policy bundle
class Policy:
    """wm+actor pair loaded from a checkpoint; champion and candidates both live here."""

    def __init__(self, ckpt_path, step_id, device):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        ca = ck.get("args", {})
        self.H = int(ca.get("history_size", 3))
        self.action_block = int(ca.get("action_block", 5))
        self.a_dim = 6 * self.action_block
        self.wm = WorldModel(n_dof=6, action_block=self.action_block,
                             history_size=self.H).to(device)
        self.wm.load_state_dict(ck["wm"]); self.wm.eval()
        self.actor = Actor(self.wm.z_dim, self.a_dim).to(device)
        self.actor.load_state_dict(ck["actor"]); self.actor.eval()
        self.step_id = step_id
        self.path = str(ckpt_path)


def hub_ckpts(repo, name, token):
    """Sorted [(step, filename)] of <name>/ckpt_*.pt on the hub."""
    from huggingface_hub import HfApi
    files = [f for f in HfApi(token=token).list_repo_files(repo)
             if f.startswith(f"{name}/ckpt_") and f.endswith(".pt")]
    return sorted((int(f.split("ckpt_")[1].split(".pt")[0]), f) for f in files)


def pick_candidate(ckpts, champion_step, rejected):
    """Newest hub ckpt that is newer than the champion and NOT previously rejected.
    Without the rejected-set a high-numbered bad ckpt would be re-probationed every
    poll forever AND shadow every newer good one (hub-latest is a lexical max)."""
    for step, fname in reversed(ckpts):
        if step <= champion_step:
            return None, None
        if step not in rejected:
            return step, fname
    return None, None


def hub_latest(repo, name, token):
    """(step, filename) of the newest <name>/ckpt_*.pt on the hub, or (None, None)."""
    ck = hub_ckpts(repo, name, token)
    return ck[-1] if ck else (None, None)


# ------------------------------------------------------------------- chunk writer
class ChunkWriter:
    """Accumulates transitions; dumps save_buffer-format npz (env_lengths=[n])."""

    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.seq = 0
        self.reset()

    def reset(self):
        self.px, self.prop, self.act = [], [], []
        self.r, self.d, self.isx = [], [], []

    def add(self, px, prop, a, r, d, is_start):
        self.px.append(px); self.prop.append(prop); self.act.append(a)
        self.r.append(r); self.d.append(d); self.isx.append(is_start)

    def __len__(self):
        return len(self.r)

    def dump(self):
        if not self.r:
            return None
        self.seq += 1
        path = self.out_dir / f"chunk_{int(time.time())}_{self.seq:04d}_{len(self.r):05d}.npz"
        np.savez_compressed(path,
                            pixels=np.stack(self.px), proprio=np.stack(self.prop),
                            action=np.stack(self.act), r=np.asarray(self.r, np.float32),
                            d=np.asarray(self.d, np.float32),
                            is_start=np.asarray(self.isx, bool),
                            env_lengths=np.asarray([len(self.r)], np.int64))
        self.reset()
        return path


class Uploader(threading.Thread):
    """Background HF uploads with retry; deletes local files on confirmed upload; if the
    backlog exceeds max_backlog (hub unreachable for hours) drops the OLDEST chunk —
    losing data beats filling the disk and killing collection."""

    def __init__(self, repo, name, out_dir, max_backlog, enabled):
        super().__init__(daemon=True)
        self.q: "queue.Queue[Path]" = queue.Queue()
        self.repo, self.name, self.out_dir = repo, name, out_dir
        self.max_backlog, self.enabled = max_backlog, enabled
        self.uploaded = self.dropped = 0

    def submit(self, path):
        if path is None:
            return
        while self.q.qsize() >= self.max_backlog:
            old = self.q.get_nowait()
            old.unlink(missing_ok=True)
            self.dropped += 1
            log(self.out_dir, "UPLOAD BACKLOG FULL — dropped oldest chunk", file=old.name)
        self.q.put(path)

    def run(self):
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        while True:
            path = self.q.get()
            if not self.enabled:
                continue                                   # --no-hf: keep local, never upload
            for attempt in range(5):
                try:
                    api.upload_file(path_or_fileobj=str(path), repo_id=self.repo,
                                    path_in_repo=f"buffers/{self.name}/{path.name}")
                    path.unlink(missing_ok=True)           # disk bound: local copy gone
                    self.uploaded += 1
                    break
                except Exception as ex:
                    log(self.out_dir, "upload failed; retrying", err=str(ex)[:120],
                        attempt=attempt, file=path.name)
                    time.sleep(min(60, 5 * 2 ** attempt))
            else:
                self.q.put(path)                           # re-queue after repeated failure
                time.sleep(120)


# -------------------------------------------------------------------------- main
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
    out_dir = Path("runs") / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    log(out_dir, "collector boot", device=str(device), name=args.name)

    # --- policy: resume own lineage if it exists, else the warmstart run -----------
    champ_file = out_dir / "champion.pt"
    boot_step, boot_file = (None, None)
    if not args.no_hf:
        try:
            boot_step, boot_file = hub_latest(repo, args.name, token)
        except Exception as ex:
            log(out_dir, "hub poll failed at boot; using warmstart", err=str(ex)[:120])
    if boot_file is not None:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=repo, filename=boot_file, token=token)
    else:
        path = resolve_ckpt(args.init_ckpt, args.warmstart_name, args.warmstart_step, repo)
        boot_step = -1     # warmstart lives in ANOTHER run's numbering; any own-lineage
                           # ckpt (learner counts from 0) must register as newer
    champion = Policy(path, boot_step, device)
    import shutil
    shutil.copyfile(path, champ_file)                       # the ratchet survives hub/cache loss
    rejects = 0
    rejected: set[int] = set()
    state_file = out_dir / "champion.json"
    if state_file.exists():                                 # rejected ckpts stay rejected across restarts
        try:
            rejected = set(json.loads(state_file.read_text()).get("rejected", []))
        except Exception:
            pass
    def save_state():
        state_file.write_text(json.dumps({"step": champion.step_id, "src": champion.path,
                                          "rejects": rejects, "rejected": sorted(rejected)}))
    save_state()
    log(out_dir, "champion loaded", step=champion.step_id, rejected=len(rejected))
    assert champion.action_block == args.action_block, "ckpt action_block != --action-block"

    from env.hardware_env import HardwareSO101Env
    env = HardwareSO101Env(n_envs=1, action_max=args.action_max,
                           safety_delta=args.safety_delta, seed=args.seed)
    uploader = Uploader(repo, args.name, out_dir, args.max_backlog, not args.no_hf)
    uploader.start()
    chunk = ChunkWriter(out_dir)

    stop = {"flag": False}
    def on_sigint(sig, frame):
        stop["flag"] = True
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        log(out_dir, "SIGINT — dumping partial chunk then exiting (arm keeps holding)")
    signal.signal(signal.SIGINT, on_sigint)

    # --- live state ------------------------------------------------------------
    acting = champion                                       # candidate during probation
    probation = None                                        # {"left": int, "trips": [..]}
    def reset_history(reason):
        nonlocal z, hist_z, hist_a, is_start
        obs_ = env.reset()
        z = encode_obs(acting.wm, obs_["image"], obs_["proprio"], device)
        hist_z = z.unsqueeze(0).repeat(acting.H, 1, 1)
        hist_a = torch.zeros(acting.H, 1, acting.a_dim, device=device)
        is_start = True
        return obs_

    obs = env.reset()
    z = encode_obs(acting.wm, obs["image"], obs["proprio"], device)
    hist_z = z.unsqueeze(0).repeat(acting.H, 1, 1)
    hist_a = torch.zeros(acting.H, 1, acting.a_dim, device=device)
    is_start = True
    ep_len = 0
    press_run = 0
    hot_polls = 0
    decisions = chunks_done = 0
    last_poll = last_beat = time.time()
    sat_window, rsafe_window = [], []
    t0 = time.time()

    def park_and_rest(temps):
        """Temp gate: fold to a gravity-stable pose, drop torque, wait until cool."""
        log(out_dir, "TEMP GATE — parking + resting", temps=[float(x) for x in temps])
        q_now = env.bus.read()[0]
        park = q_now.copy()
        park[1], park[2], park[3] = PARK_SHOULDER, PARK_ELBOW, PARK_WRIST
        for k in range(1, 21):                              # ~2 s interpolated fold (paced)
            env.bus.write_goal(q_now + (park - q_now) * k / 20.0)
            time.sleep(0.1)
        env.bus.disable_torque()
        while True:
            time.sleep(30.0)
            t = env.bus.read_temps()
            log(out_dir, "resting", temps=[float(x) for x in t])
            if 0 <= max(t) <= args.temp_resume:
                break
            if stop["flag"]:
                return
        env.bus.enable_torque()                             # holds the parked pose
        log(out_dir, "temps recovered — resuming")

    def retreat():
        """Press watchdog: a few decisions toward joint midpoints, then history break."""
        q_mid = (JOINT_LOW + JOINT_HIGH) / 2.0
        for _ in range(args.retreat_decisions):
            q_now = env.bus.read()[0]
            a_r = np.clip((q_mid - q_now) / (args.action_max * args.action_block), -1, 1)
            ab = np.tile(a_r, (1, args.action_block, 1)).astype(np.float32)
            env.step_block_async(ab)
            env.step_block_wait()
        log(out_dir, "PRESS WATCHDOG — retreated toward midpoints")

    # =========================================================== the forever loop
    while not stop["flag"]:
        cur_px, cur_prop = obs["image"], obs["proprio"]

        with torch.no_grad():
            a, _, _ = acting.actor.sample(z)
        if not torch.isfinite(a).all():                     # NaN policy: instant reject
            log(out_dir, "NON-FINITE ACTION — reverting to champion")
            if probation is not None:
                rejected.add(acting.step_id); rejects += 1; probation = None
                save_state()
            acting = champion
            obs = reset_history("nan")
            continue
        hist_a = torch.cat([hist_a[1:], a.unsqueeze(0)], 0)
        a_env = a.detach().cpu().numpy().reshape(1, args.action_block, 6)

        env.step_block_async(a_env)
        obs_next, sub_infos = env.step_block_wait()

        r_safe = float(np.mean([i["safety_reward"] for i in sub_infos]))
        z_next = encode_obs(acting.wm, obs_next["image"], obs_next["proprio"], device)
        r_cur = float(curiosity_reward(acting.wm, hist_z, hist_a, z_next)[0])
        reward = args.lambda_safe * r_safe + args.lambda_cur * float(np.log1p(r_cur))

        ep_len += 1
        done = float(ep_len >= args.episode_steps)
        chunk.add(cur_px[0], cur_prop[0], a_env.reshape(-1), reward, done, is_start)
        is_start = done > 0
        if done:
            ep_len = 0
        decisions += 1
        z = z_next
        hist_z = torch.cat([hist_z[1:], z_next.unsqueeze(0)], 0)

        # --- watchdog signals (tau_meas is hardware-only; mock provides it too) ----
        taus = np.stack([i["tau_meas"][0] for i in sub_infos])          # (block, 6)
        qvels = np.stack([i["qvel"][0] for i in sub_infos])
        # PER-JOINT press: the pinned joint has pegged tau AND ~zero velocity while the
        # rest of the arm may still wander (the 2026-06-06 organic press had arm-wide
        # |qd|max 0.09 — an arm-quiet predicate would miss every real press).
        pressed = ((np.abs(taus) > args.press_tau) & (np.abs(qvels) < args.press_vel)).any(1)
        press_run = press_run + 1 if pressed.mean() > 0.5 else 0
        sat_window.append(float(np.abs(a_env).mean()))
        rsafe_window.append(r_safe)

        if press_run >= args.press_decisions:
            retreat()
            press_run = 0
            if probation is not None:                       # candidate caused a press: reject
                log(out_dir, "PROBATION REJECT: press", step=acting.step_id)
                rejected.add(acting.step_id); rejects += 1
                acting = champion; probation = None
                save_state()
            obs = reset_history("press")
            continue

        # --- probation accounting ---------------------------------------------------
        if probation is not None:
            probation["left"] -= 1
            if probation["left"] <= 0:
                w = min(len(sat_window), args.probation_steps)
                sat = float(np.mean(sat_window[-w:])); rs = float(np.mean(rsafe_window[-w:]))
                if sat > args.probation_sat or rs < args.probation_rsafe:
                    log(out_dir, "PROBATION REJECT", step=acting.step_id, sat=sat, r_safe=rs)
                    rejected.add(acting.step_id); rejects += 1
                    acting = champion
                    obs = reset_history("reject")
                else:
                    champion = acting
                    shutil.copyfile(acting.path, champ_file)
                    log(out_dir, "PROBATION PASS — new champion", step=champion.step_id,
                        sat=sat, r_safe=rs)
                probation = None
                save_state()

        # --- temp gate (debounced: the sensor sits near the driver FETs and can spike
        #     ~15 degC transiently under heavy drive — drill 2026-06-06 saw 46->33 in 32 s.
        #     Require 2 consecutive hot polls (~16 s apart) before parking.) -------------
        if decisions % args.temp_every == 0:
            temps = env.bus.read_temps()
            hot_polls = hot_polls + 1 if max(temps) > args.temp_gate else 0
            if hot_polls >= 2:
                hot_polls = 0
                if len(chunk) >= args.min_chunk:
                    uploader.submit(chunk.dump()); chunks_done += 1
                park_and_rest(temps)
                obs = reset_history("temp_rest")
                continue

        # --- checkpoint poll (skip while a candidate is on probation) -----------------
        if (not args.no_hf and probation is None
                and time.time() - last_poll > args.poll_every):
            last_poll = time.time()
            try:
                step, fname = pick_candidate(hub_ckpts(repo, args.name, token),
                                             champion.step_id, rejected)
                if step is not None:
                    from huggingface_hub import hf_hub_download
                    p = hf_hub_download(repo_id=repo, filename=fname, token=token)
                    acting = Policy(p, step, device)
                    probation = {"left": args.probation_steps}
                    log(out_dir, "candidate on probation", step=step,
                        probation_steps=args.probation_steps)
                    obs = reset_history("swap")
                    continue
            except Exception as ex:
                log(out_dir, "ckpt poll failed (non-fatal)", err=str(ex)[:120])

        # --- chunk + heartbeat ----------------------------------------------------------
        if len(chunk) >= args.chunk_steps:
            uploader.submit(chunk.dump()); chunks_done += 1
        if time.time() - last_beat > 60:
            last_beat = time.time()
            log(out_dir, "heartbeat", decisions=decisions, chunks=chunks_done,
                champion=champion.step_id, probation=bool(probation), rejects=rejects,
                queue=uploader.q.qsize(), uploaded=uploader.uploaded,
                dropped=uploader.dropped,
                sps=round(decisions / max(time.time() - t0, 1e-9), 2))
        obs = obs_next

    # ----------------------------------------------------------------- shutdown
    if len(chunk):
        uploader.submit(chunk.dump())
    deadline = time.time() + 120                            # let pending uploads finish
    while uploader.q.qsize() and time.time() < deadline:
        time.sleep(1)
    env.close()
    log(out_dir, "collector exit", decisions=decisions, chunks=chunks_done)


def parse_args():
    p = argparse.ArgumentParser(description="24/7 frozen collector with hot-swap + watchdogs")
    p.add_argument("--name", default="auto1", help="loop name: <name>/ckpt_* + buffers/<name>/")
    p.add_argument("--warmstart-name", default="safe15")
    p.add_argument("--warmstart-step", type=int, default=100000)
    p.add_argument("--init-ckpt", default=None, help="local .pt warmstart override")
    # frozen campaign config (round_runbook.md) — change only between campaigns
    p.add_argument("--action-max", type=float, default=0.1)
    p.add_argument("--action-block", type=int, default=5)
    p.add_argument("--lambda-safe", type=float, default=0.1)
    p.add_argument("--lambda-cur", type=float, default=20.0)
    p.add_argument("--safety-delta", type=float, default=15.0)
    p.add_argument("--episode-steps", type=int, default=200, help="bookkeeping truncation")
    # chunking / transport
    p.add_argument("--chunk-steps", type=int, default=1000)
    p.add_argument("--min-chunk", type=int, default=50, help="don't ship slivers on temp rests")
    p.add_argument("--max-backlog", type=int, default=20, help="unuploaded chunks kept on disk")
    p.add_argument("--poll-every", type=float, default=300, help="ckpt poll period [s]")
    # acceptance gate
    p.add_argument("--probation-steps", type=int, default=30)
    p.add_argument("--probation-sat", type=float, default=0.90, help="mean |a| reject threshold")
    p.add_argument("--probation-rsafe", type=float, default=-5.0, help="mean r_safe reject threshold")
    # watchdogs
    p.add_argument("--temp-gate", type=float, default=50.0)
    p.add_argument("--temp-resume", type=float, default=42.0)
    p.add_argument("--temp-every", type=int, default=20, help="decisions between temp polls")
    p.add_argument("--press-tau", type=float, default=2.5,
                   help="N*m; tau_meas clips at 3.35 so 3.0 left little headroom — 2.5 is "
                        "still well above benign interaction loads (~1.9 peak measured), "
                        "and the per-joint qd gate filters moving contacts")
    p.add_argument("--press-vel", type=float, default=0.05)
    p.add_argument("--press-decisions", type=int, default=5, help="~2 s at 2.6 sps")
    p.add_argument("--retreat-decisions", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-hf", action="store_true", help="dry-run: no uploads, no ckpt polls")
    p.add_argument("--hf-repo", default=None)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
