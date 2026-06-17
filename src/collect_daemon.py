"""24/7 collector daemon — the Mac half of the autonomous hardware->cloud loop.

Runs the frozen acting loop on the physical SO-ARM101 indefinitely:
  - dumps transitions in CHUNKS (save_buffer npz format) and uploads them to the HF Hub
    on a background thread (disk-bounded: local files deleted on confirmed upload,
    oldest dropped if the upload backlog grows — never fill the disk)
  - a background fetcher thread polls the Hub (~every --poll-every s) for new
    checkpoints from learner_daemon.py and downloads them off the acting loop; the
    main loop HOT-SWAPS to a ready candidate between decisions — but only after an
    on-arm ACCEPTANCE PROBATION: the candidate drives ~30 watched decisions first; any
    watchdog trip (press, saturated actions, real fights, NaNs) rejects it and reverts
    to the last-known-good CHAMPION (runs/<name>/champion.pt — the ratchet that makes
    a bad upload recoverable)
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
from collections import deque
from pathlib import Path

import numpy as np
import torch

try:
    import wandb
except ImportError:
    wandb = None

try:
    import imageio
except ImportError:
    imageio = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.state_encoder import WorldModel                       # noqa: E402
from src.train import Actor, curiosity_reward, encode_obs, load_actor_state, resolve_ckpt   # noqa: E402
from env.hardware_env import JOINT_HIGH, JOINT_LOW               # noqa: E402

# Gravity-stable fold for torque-off rest (≈ where the arm settles when limp; pan/roll/
# gripper keep their current values). Bench-verified gravity-stable 2026-06-06 (0.0 mrad
# drift after torque-off).
PARK_SHOULDER, PARK_ELBOW, PARK_WRIST = -1.70, -1.60, -1.00
DEAD_BUS_POLLS = 60          # rest polls (30 s apart) of all-failed temp reads before giving up


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
        load_actor_state(self.actor, ck["actor"]); self.actor.eval()
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


class CkptFetcher(threading.Thread):
    """Background hub poll + candidate download. At near-live cadence (learner uploads
    every ~45 s, we poll every ~20 s) a synchronous 59 MB download would stall the
    acting loop for many seconds every minute — so both the list call and the download
    live here, and the main loop only ever adopts a READY LOCAL FILE between decisions.
    Holds one candidate; a fresher hub ckpt replaces an unconsumed one (the arm should
    always probation the newest). Main-loop feedback arrives via note()."""

    def __init__(self, repo, name, token, out_dir, poll_every, champion_step, rejected):
        super().__init__(daemon=True)
        self.repo, self.name, self.token = repo, name, token
        self.out_dir, self.poll_every = out_dir, poll_every
        self.dl_dir = out_dir / "candidates"
        self.dl_dir.mkdir(exist_ok=True)
        for f in self.dl_dir.glob(f"{name}/ckpt_*.pt"):     # leftovers from a crash
            f.unlink(missing_ok=True)
        self.lock = threading.Lock()
        self.champion_step = champion_step
        self.rejected = set(rejected)
        self.slot = None                                    # (step, Path) ready to adopt

    def note(self, champion_step=None, rejected_step=None):
        """Main-loop outcomes, so stale/rejected ckpts are never re-downloaded."""
        with self.lock:
            if champion_step is not None:
                self.champion_step = champion_step
            if rejected_step is not None:
                self.rejected.add(rejected_step)

    def take(self):
        """Pop the ready candidate, or None (called between decisions, never blocks).
        The handed-out step raises the poll floor immediately — otherwise the next poll
        would re-download the very ckpt that is out on probation."""
        with self.lock:
            cand, self.slot = self.slot, None
            if cand is not None:
                self.champion_step = max(self.champion_step, cand[0])
        return cand

    def run(self):
        from huggingface_hub import hf_hub_download
        while True:
            try:
                with self.lock:
                    champ, rej = self.champion_step, set(self.rejected)
                    held = self.slot[0] if self.slot else None
                floor = champ if held is None else max(champ, held)
                step, fname = pick_candidate(hub_ckpts(self.repo, self.name, self.token),
                                             floor, rej)
                if step is not None:
                    # local_dir (not the hf cache): one real file we can unlink after
                    # probation — cache entries would accrete 59 MB per ckpt forever
                    p = Path(hf_hub_download(repo_id=self.repo, filename=fname,
                                             token=self.token, local_dir=self.dl_dir))
                    with self.lock:
                        stale, self.slot = self.slot, (step, p)
                    if stale is not None:
                        stale[1].unlink(missing_ok=True)
                    log(self.out_dir, "candidate downloaded", step=step)
            except Exception as ex:
                log(self.out_dir, "ckpt fetch failed (non-fatal)", err=str(ex)[:120])
            time.sleep(self.poll_every)


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

    # W&B is the dashboard layer ON TOP of daemon.jsonl (which stays the crash-safe local
    # log). Stable id + resume="allow": daemon restarts continue ONE W&B run instead of
    # spawning a new run per restart. Auto-step (no step=): collector decision counts
    # reset on restart and W&B rejects non-monotonic steps — `decisions` is a field.
    run = None
    if not args.no_wandb and wandb is not None and os.environ.get("WANDB_API_KEY"):
        try:                                                # transient W&B outage at boot must not kill the daemon
            run = wandb.init(project=args.wandb_project or os.environ.get("WANDB_PROJECT", "curious-robot"),
                             entity=os.environ.get("WANDB_ENTITY"),
                             name=f"{args.name}-collector", id=f"{args.name}-collector",
                             resume="allow", group=args.name, dir=str(out_dir), config=vars(args))
            print(f"[wandb] {run.url}", flush=True)
        except Exception as e:
            run = None
            log(out_dir, "wandb init failed (non-fatal, dashboard off)", err=str(e))

    def wlog(d):
        if run is not None:
            try:
                run.log(d)
            except Exception:
                pass                                        # dashboard must never kill the arm loop

    # --- policy boot order: newest NON-REJECTED own-lineage ckpt on the hub ->
    #     local champion.pt ratchet -> the warmstart run. The rejected-set loads FIRST
    #     so a crash-restart while a bad ckpt sits as hub-latest cannot adopt it
    #     unprobed (it was rejected for a reason). ---------------------------------
    champ_file = out_dir / "champion.pt"
    state_file = out_dir / "champion.json"
    rejects = 0
    rejected: set[int] = set()
    prev_state = {}
    if state_file.exists():                                 # rejections survive restarts
        try:
            prev_state = json.loads(state_file.read_text())
            rejected = set(prev_state.get("rejected", []))
        except Exception:
            pass
    boot_step, boot_file = (None, None)
    if not args.no_hf:
        try:
            boot_step, boot_file = pick_candidate(hub_ckpts(repo, args.name, token),
                                                  -10**12, rejected)
        except Exception as ex:
            log(out_dir, "hub poll failed at boot; falling back", err=str(ex)[:120])
    if boot_file is not None:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=repo, filename=boot_file, token=token)
    elif champ_file.exists():
        path = str(champ_file)
        boot_step = int(prev_state.get("step", -1))
        log(out_dir, "boot from local champion ratchet", step=boot_step)
    else:
        path = resolve_ckpt(args.init_ckpt, args.warmstart_name, args.warmstart_step, repo)
        boot_step = -1     # warmstart lives in ANOTHER run's numbering; any own-lineage
                           # ckpt (learner counts from 0) must register as newer
    champion = Policy(path, boot_step, device)
    import shutil
    if os.path.abspath(path) != os.path.abspath(champ_file):
        shutil.copyfile(path, champ_file)                   # the ratchet survives hub/cache loss
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
    fetcher = None
    if not args.no_hf:
        fetcher = CkptFetcher(repo, args.name, token, out_dir, args.poll_every,
                              champion.step_id, rejected)
        fetcher.start()

    def discard_candidate(pol):
        """Drop a consumed candidate download (champion.pt holds any promoted copy)."""
        p = Path(pol.path)
        if fetcher is not None and p.is_relative_to(fetcher.dl_dir):
            p.unlink(missing_ok=True)
    # Re-queue chunks a previous run dumped but never uploaded — the upload queue is
    # in-memory, so they'd otherwise be orphaned on disk forever. Validate first: a
    # SIGKILL mid-savez leaves a torn npz, which must never reach the hub (the learner
    # would crash-loop on it). Torn files are quarantined, not deleted.
    for orphan in sorted(out_dir.glob("chunk_*.npz")):
        try:
            with np.load(orphan) as z:
                if "env_lengths" not in z.files:
                    raise ValueError("missing env_lengths")
        except Exception as ex:
            orphan.rename(orphan.with_name(orphan.name + ".corrupt"))
            log(out_dir, "quarantined torn orphan chunk", file=orphan.name, err=str(ex)[:80])
            continue
        uploader.submit(orphan)
        log(out_dir, "re-queued orphan chunk from a previous run", file=orphan.name)
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
    ep_ret = 0.0
    press_run = 0
    hot_polls = 0
    presses = 0
    decisions = chunks_done = 0
    last_beat = time.time()
    sat_window, rsafe_window = [], []
    recent_r = deque(maxlen=200)
    recent_rcur = deque(maxlen=200)
    last_temps = None
    t0 = time.time()

    # --- train videos: mirror train.py's train/wrist + train/overhead panels (same
    # cadence/keys as the pinned runs). Wrist is free (the obs the policy sees);
    # overhead is a second USB camera (--overhead-cam / SOARM_OVERHEAD_CAM, e.g. the
    # Mac's built-in cam pointed at the arm), grabbed only inside the buffer window.
    video_on = imageio is not None and args.video_every > 0
    wrist_buf = deque(maxlen=args.video_steps)
    over_buf = deque(maxlen=args.video_steps)
    over_cam = None
    if video_on and args.overhead_cam >= 0:
        try:
            from env.hardware_env import UsbCamera
            over_cam = UsbCamera(index=args.overhead_cam, hw=224)
        except Exception as ex:
            log(out_dir, "overhead cam unavailable (non-fatal)", err=str(ex)[:80])

    def park_and_rest(temps):
        """Temp gate: fold to a gravity-stable pose, drop torque, wait until cool.
        SIGINT during the rest BREAKS (not returns) so torque is re-enabled and the
        'exits with the arm holding' convention stays true. If every temp read fails
        (-1 x6 = dead bus) for DEAD_BUS_POLLS consecutive polls, raise — a loud dead
        process beats silently impersonating a long cool-down for days (the arm is
        already parked + limp, the safe state for a dead bus)."""
        log(out_dir, "TEMP GATE — parking + resting", temps=[float(x) for x in temps])
        q_now = env.bus.read()[0]
        park = q_now.copy()
        park[1], park[2], park[3] = PARK_SHOULDER, PARK_ELBOW, PARK_WRIST
        for k in range(1, 21):                              # ~2 s interpolated fold (paced)
            env.bus.write_goal(q_now + (park - q_now) * k / 20.0)
            time.sleep(0.1)
            env.bus.read()   # refresh _last_pos: max_step_ticks clamps goals to +/-0.46 rad
                             # of the last READ — without this the >1 rad fold stops short
                             # and torque-off would drop the arm from a non-park pose
        env.bus.disable_torque()
        dead_polls = 0
        while True:
            time.sleep(30.0)
            if stop["flag"]:
                break                                       # fall through to enable_torque
            t = env.bus.read_temps()
            log(out_dir, "resting", temps=[float(x) for x in t])
            if max(t) < 0:                                  # ALL reads failed -> bus likely dead
                dead_polls += 1
                if dead_polls >= DEAD_BUS_POLLS:
                    log(out_dir, "BUS DEAD during temp rest — exiting loudly (arm parked+limp)")
                    raise RuntimeError(
                        f"all servo temp reads failed for {DEAD_BUS_POLLS} consecutive polls "
                        "(~30 min) during temp rest; bus presumed dead. Arm left parked+limp.")
                continue
            dead_polls = 0
            if max(t) <= args.temp_resume:
                break
        env.bus.enable_torque()                             # holds the parked pose
        log(out_dir, "temps recovered — resuming" if not stop["flag"]
            else "SIGINT during rest — re-energized at park, exiting holding")

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

    # --- deferred per-decision bookkeeping (jerk fix #3, 2026-06-11) ---------------
    # Curiosity + reward + chunk.add for decision d need nothing from block d+1, so
    # they run while the arm is already executing it — the only serial work left
    # between blocks is encode + sample (~18 ms vs ~25 ms). This is NOT action
    # pipelining: every action is still sampled from the freshest observation.
    pend = None          # (hist_z, hist_a, z_next, r_safe, px, prop, a_flat, wm)

    def flush_pending():
        """Run the deferred curiosity/reward/buffer write for the stashed decision.
        MUST be called before any history reset, chunk dump on a gate, or shutdown,
        so the transition lands in the chunk and is_start ordering stays exact. The
        generating wm is stored in the stash, so policy swaps can't misattribute it."""
        nonlocal pend, ep_len, ep_ret, is_start
        if pend is None:
            return
        p_hz, p_ha, p_zn, p_rs, p_px, p_prop, p_a, p_wm = pend
        pend = None
        r_cur = float(curiosity_reward(p_wm, p_hz, p_ha, p_zn)[0])
        reward = args.lambda_safe * p_rs + args.lambda_cur * float(np.log1p(r_cur))
        recent_r.append(reward)
        recent_rcur.append(r_cur)
        ep_len += 1
        ep_ret += reward
        done = float(ep_len >= args.episode_steps)
        chunk.add(p_px, p_prop, p_a, reward, done, is_start)
        is_start = done > 0
        if done:
            wlog({"episode/return": ep_ret, "episode/len": ep_len})
            ep_len = 0
            ep_ret = 0.0

    # =========================================================== the forever loop
    while not stop["flag"]:
        cur_px, cur_prop = obs["image"], obs["proprio"]

        with torch.no_grad():
            a = acting.actor(z)        # deterministic mean (deployment); SAC training samples ~pi (see Actor)
        if not torch.isfinite(a).all():                     # NaN policy: instant reject
            log(out_dir, "NON-FINITE ACTION — reverting to champion")
            flush_pending()
            if probation is not None:
                rejected.add(acting.step_id); rejects += 1; probation = None
                if fetcher is not None:
                    fetcher.note(rejected_step=acting.step_id)
                discard_candidate(acting)
                save_state()
            acting = champion
            obs = reset_history("nan")
            continue
        hist_a = torch.cat([hist_a[1:], a.unsqueeze(0)], 0)
        a_env = a.detach().cpu().numpy().reshape(1, args.action_block, 6)

        env.step_block_async(a_env)
        flush_pending()                # decision d-1's bookkeeping overlaps block d's motion
        obs_next, sub_infos = env.step_block_wait()

        r_safe = float(np.mean([i["safety_reward"] for i in sub_infos]))
        z_next = encode_obs(acting.wm, obs_next["image"], obs_next["proprio"], device)
        pend = (hist_z, hist_a, z_next, r_safe, cur_px[0], cur_prop[0],
                a_env.reshape(-1), acting.wm)
        decisions += 1
        z = z_next
        hist_z = torch.cat([hist_z[1:], z_next.unsqueeze(0)], 0)

        # --- train videos: buffer frames in the window before each save, then save the
        #     wrist + overhead clips every video_every (predicate mirrors train.py) ---
        if video_on and 0 < decisions % args.video_every \
                and decisions % args.video_every >= args.video_every - args.video_steps:
            wrist_buf.append(cur_px[0])
            if over_cam is not None:
                try:
                    over_buf.append(over_cam.read())
                except Exception as ex:
                    log(out_dir, "overhead grab failed — disabling (non-fatal)",
                        err=str(ex)[:80])
                    over_cam = None
        if video_on and decisions > 0 and decisions % args.video_every == 0:
            roll_dir = out_dir / "rollouts"
            roll_dir.mkdir(exist_ok=True)
            for tag, buf_ in (("wrist", wrist_buf), ("overhead", over_buf)):
                if not buf_:
                    continue
                vp = roll_dir / f"train_{tag}_{decisions:07d}.mp4"
                try:
                    imageio.mimsave(vp, list(buf_), fps=args.video_fps)
                    if run is not None:
                        wlog({f"train/{tag}": wandb.Video(str(vp), format="mp4")})
                        vp.unlink(missing_ok=True)   # in W&B now; keep local disk clean
                    buf_.clear()
                except Exception as ex:
                    log(out_dir, f"video {tag} failed (non-fatal)", err=str(ex)[:80])

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
            flush_pending()
            retreat()
            press_run = 0
            presses += 1
            wlog({"watchdog/presses": presses, "watchdog/press_decision": decisions})
            if probation is not None:                       # candidate caused a press: reject
                log(out_dir, "PROBATION REJECT: press", step=acting.step_id)
                rejected.add(acting.step_id); rejects += 1
                if fetcher is not None:
                    fetcher.note(rejected_step=acting.step_id)
                discard_candidate(acting)
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
                    if fetcher is not None:
                        fetcher.note(rejected_step=acting.step_id)
                    discard_candidate(acting)
                    wlog({"probation/rejects": rejects, "probation/last_reject_step": acting.step_id,
                          "probation/sat": sat, "probation/r_safe": rs})
                    flush_pending()
                    acting = champion
                    obs = reset_history("reject")
                else:
                    champion = acting
                    shutil.copyfile(acting.path, champ_file)
                    if fetcher is not None:
                        fetcher.note(champion_step=champion.step_id)
                    discard_candidate(champion)
                    log(out_dir, "PROBATION PASS — new champion", step=champion.step_id,
                        sat=sat, r_safe=rs)
                    wlog({"probation/champion_step": champion.step_id,
                          "probation/sat": sat, "probation/r_safe": rs})
                probation = None
                save_state()

        # --- temp gate (debounced: the sensor sits near the driver FETs and can spike
        #     ~15 degC transiently under heavy drive — drill 2026-06-06 saw 46->33 in 32 s.
        #     Require 2 consecutive hot polls (~16 s apart) before parking.) -------------
        if decisions % args.temp_every == 0:
            temps = env.bus.read_temps()
            last_temps = temps
            hot_polls = hot_polls + 1 if max(temps) > args.temp_gate else 0
            if hot_polls >= 2:
                hot_polls = 0
                wlog({"watchdog/temp_trips": 1, "temps/max_at_trip": float(max(temps))})
                flush_pending()
                if len(chunk) >= args.min_chunk:
                    uploader.submit(chunk.dump()); chunks_done += 1
                park_and_rest(temps)
                obs = reset_history("temp_rest")
                continue

        # --- checkpoint adoption (fetcher polls + downloads in the background; the
        #     re-check matters: the slot was filled against possibly-stale state) ------
        if fetcher is not None and probation is None:
            cand = fetcher.take()
            if cand is not None:
                step, cpath = cand
                if step > champion.step_id and step not in rejected:
                    flush_pending()
                    acting = Policy(cpath, step, device)
                    probation = {"left": args.probation_steps}
                    log(out_dir, "candidate on probation", step=step,
                        probation_steps=args.probation_steps)
                    obs = reset_history("swap")
                    continue
                cpath.unlink(missing_ok=True)           # stale by the time it surfaced

        # --- chunk + heartbeat ----------------------------------------------------------
        if len(chunk) >= args.chunk_steps:
            uploader.submit(chunk.dump()); chunks_done += 1
        if time.time() - last_beat > 60:
            last_beat = time.time()
            sps = round(decisions / max(time.time() - t0, 1e-9), 2)
            log(out_dir, "heartbeat", decisions=decisions, chunks=chunks_done,
                champion=champion.step_id, probation=bool(probation), rejects=rejects,
                queue=uploader.q.qsize(), uploaded=uploader.uploaded,
                dropped=uploader.dropped, sps=sps)
            # key names mirror train.py's schema (safe15 et al.) so daemon runs land on
            # the same W&B panels; cur_contrib/safe_cur_ratio use train.py's definitions.
            safe_m = float(np.mean(rsafe_window[-200:])) if rsafe_window else 0.0
            cur_m = (float(args.lambda_cur * np.mean(np.log1p(np.asarray(recent_rcur))))
                     if recent_rcur else 0.0)
            beat = {"buffer/transitions": decisions, "perf/steps_per_sec": sps,
                    "collector/chunks_uploaded": uploader.uploaded,
                    "collector/upload_queue": uploader.q.qsize(),
                    "collector/chunks_dropped": uploader.dropped,
                    "probation/champion_step": champion.step_id,
                    "probation/rejects": rejects, "watchdog/presses": presses,
                    "reward/total": float(np.mean(recent_r)) if recent_r else 0.0,
                    "reward/r_cur": float(np.mean(recent_rcur)) if recent_rcur else 0.0,
                    "reward/r_safe": safe_m,
                    "reward/cur_contrib": cur_m,
                    "reward/safe_cur_ratio": abs(safe_m) / max(abs(cur_m), 1e-6),
                    "action/sat": float(np.mean(sat_window[-200:])) if sat_window else 0.0}
            if last_temps is not None:
                beat["temps/max"] = float(max(last_temps))
                beat.update({f"temps/servo{j+1}": float(t) for j, t in enumerate(last_temps)})
            wlog(beat)
        obs = obs_next

    # ----------------------------------------------------------------- shutdown
    flush_pending()
    if len(chunk):
        uploader.submit(chunk.dump())
    deadline = time.time() + 120                            # let pending uploads finish
    while uploader.q.qsize() and time.time() < deadline:
        time.sleep(1)
    env.close()
    if run is not None:
        run.finish()
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
    p.add_argument("--lambda-safe", type=float, default=2.2)   # 2026-06-12 real-arm calibration
    p.add_argument("--lambda-cur", type=float, default=15.0)   # 2026-06-11: 20 -> 15 (sweep candidate)
    p.add_argument("--safety-delta", type=float, default=9.0)  # benign<=7.4 / bad>=10.7 (true-dt, P8/D16)
    p.add_argument("--episode-steps", type=int, default=200, help="bookkeeping truncation")
    # chunking / transport
    p.add_argument("--chunk-steps", type=int, default=1000)
    p.add_argument("--min-chunk", type=int, default=50, help="don't ship slivers on temp rests")
    p.add_argument("--max-backlog", type=int, default=20, help="unuploaded chunks kept on disk")
    p.add_argument("--poll-every", type=float, default=20,
                   help="ckpt poll period [s] (background thread; near-live tracking "
                        "of the learner's --save-secs uploads)")
    # acceptance gate
    p.add_argument("--probation-steps", type=int, default=30)
    p.add_argument("--probation-sat", type=float, default=0.90, help="mean |a| reject threshold")
    p.add_argument("--probation-rsafe", type=float, default=-0.05,
                   help="mean raw r_safe reject threshold over the probation window. Under delta=9 "
                        "benign decisions score exactly 0, so any sustained negative mean is real: "
                        "-0.05 trips on ~2+ bad decisions per 30 while one external snag passes "
                        "(the old -5.0 was unreachable dead code)")
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
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-project", default=None)
    # train videos (defaults mirror train.py's pinned-run settings)
    p.add_argument("--video-every", type=int, default=1000,
                   help="train-video period (decision steps): save a wrist + overhead clip every N; 0 disables")
    p.add_argument("--video-steps", type=int, default=60,
                   help="frames per train-video clip (window of decision steps before each save)")
    p.add_argument("--video-fps", type=int, default=20)
    p.add_argument("--overhead-cam", type=int,
                   default=int(os.environ.get("SOARM_OVERHEAD_CAM", "1")),
                   help="cv2 index of the overhead/Mac camera; -1 disables overhead clips")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
