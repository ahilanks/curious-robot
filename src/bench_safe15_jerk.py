"""Instrumented frozen-policy rollout on the real arm — attribute the safe15 jerk.

Reproduces collect_daemon's exact acting cadence (actor(z) -> 5x _control_step ->
encode_obs (ViT) -> curiosity -> next decision ...; the daemon's step_block_async +
immediate step_block_wait is the same serial chain, one thread hop) but wraps the bus
and camera in wall-clock probes so every stage of every control substep is timestamped:

    write_goal | dt_safe sleep | bus read (q,qd,tau) | camera read      x action_block
    ...then the inter-block inference gap: encode_obs + curiosity + actor forward

Hypothesis under test: the jerk is the inter-block stall — the arm dead-stops at the
block's last goal while the ViT/WM/actor run on MPS, then the next block yanks it off
again. The per-stage timestamps + per-substep q/qd/tau (raw AND filtered) let us check
that against the rivals (intra-substep read+camera tail, goal staircase, pacing).

Writes everything to <out>/jerk_<name>_<steps>.npz. MPS stages are synchronized before
each timestamp so stage durations are real, not kernel-launch times.

Usage:
    export SOARM_PORT=/dev/cu.usbmodem5AA90245791 SOARM_CALIB=so101_calib.json
    python src/bench_safe15_jerk.py --steps 100 --action-max 0.1
    SOARM_MOCK=1 python src/bench_safe15_jerk.py --steps 10   # dry-run, no hardware
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

from model.state_encoder import WorldModel, pred_dims_from_args                      # noqa: E402
from src.train import Actor, encode_obs, curiosity_reward, load_actor_state, resolve_ckpt   # noqa: E402
from env.hardware_env import (HardwareSO101Env, _default_bus,   # noqa: E402
                              _default_camera)


class BusProbe:
    """Timestamp + record every write_goal/read; everything else delegates."""

    def __init__(self, bus):
        self._b = bus
        self.write_t = []     # (t0, t1) per write_goal
        self.read_t = []      # (t0, t1) per read
        self.goals = []       # commanded goal [rad]
        self.paces = []       # per-servo paced Goal_Speed written with the goal
        self.raw_reads = []   # (q, qd_raw, tau_raw) straight off the wire (pre-EMA)

    def write_goal(self, goal):
        t0 = time.perf_counter()
        self._b.write_goal(goal)
        self.write_t.append((t0, time.perf_counter()))
        self.goals.append(np.asarray(goal, np.float32).copy())
        sp = getattr(self._b, "_last_speeds", None)
        self.paces.append(np.asarray(sp, np.float32).copy() if sp is not None
                          else np.full(6, np.nan, np.float32))

    def read(self):
        t0 = time.perf_counter()
        out = self._b.read()
        self.read_t.append((t0, time.perf_counter()))
        self.raw_reads.append(tuple(np.copy(x) for x in out))
        return out

    def __getattr__(self, k):
        return getattr(self._b, k)


class CamProbe:
    def __init__(self, cam):
        self._c = cam
        self.cam_t = []

    def read(self):
        t0 = time.perf_counter()
        f = self._c.read()
        self.cam_t.append((t0, time.perf_counter()))
        return f

    def __getattr__(self, k):
        return getattr(self._c, k)


def _sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


# Safe above-table rest pose (== bench_pgain_path HOME): pan in the cable-safe band,
# lift/elbow/wrist level, grip half-open. The arm returns here after every run so the
# next run starts from a known pose instead of wherever the policy parked the wrist.
HOME = np.array([0.35, 0.0, 0.0, 0.0, 0.0, 0.5], np.float32)


def go_home(bus, secs: float = 2.0) -> None:
    q0 = bus.read()[0]
    n = max(int(secs / 0.1), 1)
    for k in range(1, n + 1):
        bus.write_goal(q0 + (HOME - q0) * k / n)
        time.sleep(0.1)
        bus.read()    # refresh _last_pos: the max_step_ticks clamp is relative to the
                      # last READ, so an unread multi-step move silently stops ~0.46 rad out


def main():
    p = argparse.ArgumentParser(description="Instrumented jerk bench (frozen policy, real arm)")
    p.add_argument("--name", default="safe15")
    p.add_argument("--step", type=int, default=100000)
    p.add_argument("--ckpt", default=None, help="local .pt override")
    p.add_argument("--steps", type=int, default=100, help="control substeps (30 ms each)")
    p.add_argument("--action-max", type=float, default=0.1, help="campaign-pinned 0.1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/jerk_bench")
    p.add_argument("--no-home", action="store_true",
                   help="skip the end-of-run interpolated return to the HOME pose")
    p.add_argument("--device", default=None, choices=["cpu", "mps", "cuda"],
                   help="inference device override (default: auto-pick cuda/mps/cpu)")
    args = p.parse_args()

    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available()
                     else "mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(args.seed)

    ck = torch.load(resolve_ckpt(args.ckpt, args.name, args.step), map_location=device,
                    weights_only=False)
    ca = ck["args"]
    H, AB = int(ca["history_size"]), int(ca["action_block"])
    a_dim = 6 * AB
    wm = WorldModel(n_dof=6, action_block=AB, history_size=H, **pred_dims_from_args(ca)).to(device)
    wm.load_state_dict(ck["wm"]); wm.eval()
    actor = Actor(wm.z_dim, a_dim).to(device)
    load_actor_state(actor, ck["actor"]); actor.eval()
    print(f"[jerk] {args.name}@{ca.get('total_steps')} H={H} AB={AB} "
          f"action_max={args.action_max} device={device.type}", flush=True)

    bus = BusProbe(_default_bus())
    cam = CamProbe(_default_camera(224))
    env = HardwareSO101Env(bus=bus, camera=cam, action_max=args.action_max,
                           safety_delta=float(ca.get("safety_delta", 15.0)))
    temps0 = bus.read_temps()
    print(f"[jerk] temps before: {temps0.tolist()}", flush=True)

    obs = env.reset()
    # reset() consumed one bus read + one camera read — drop them so per-substep
    # event lists align 1:1 with substeps.
    n_skip_reads, n_skip_cams = len(bus.read_t), len(cam.cam_t)

    z = encode_obs(wm, obs["image"], obs["proprio"], device)
    hist_z = z.unsqueeze(0).repeat(H, 1, 1)
    hist_a = torch.zeros(H, 1, a_dim, device=device)

    sub = {k: [] for k in ("dec", "k", "q", "qd_env", "tau_meas", "tau_recomp", "r_safe")}
    dec = {k: [] for k in ("t_sample", "t_encode", "t_cur", "r_cur", "action")}

    n_dec = (args.steps + AB - 1) // AB
    pend = None              # (hist_z, hist_a, z_next): deferred curiosity, mirrors the
    t_run0 = time.perf_counter()   # daemon's jerk-fix-#3 reorder (runs during the block)
    for d in range(n_dec):
        _sync(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            a = actor(z)                                   # deterministic mean (deploy/bench); training samples ~pi
        block = a.detach().cpu().numpy().reshape(AB, 6)    # .cpu() forces the MPS sync,
        t1 = time.perf_counter()                            # exactly like the daemon
        dec["t_sample"].append((t0, t1))
        dec["action"].append(block.copy())
        hist_a = torch.cat([hist_a[1:], a.unsqueeze(0)], 0)

        env.step_block_async(block[None])
        if pend is not None:                                # overlaps the block's motion
            t0c = time.perf_counter()
            dec["r_cur"].append(float(curiosity_reward(wm, *pend)[0]))
            dec["t_cur"].append((t0c, time.perf_counter()))
        obs_s, sub_infos = env.step_block_wait()
        for k, info in enumerate(sub_infos):
            sub["dec"].append(d); sub["k"].append(k)
            sub["q"].append(info["qpos"][0]); sub["qd_env"].append(info["qvel"][0])
            sub["tau_meas"].append(info["tau_meas"][0])
            sub["tau_recomp"].append(info["applied_torque"][0])
            sub["r_safe"].append(float(info["safety_reward"][0]))

        t0 = time.perf_counter()
        z_next = encode_obs(wm, obs_s["image"], obs_s["proprio"], device)
        _sync(device)
        dec["t_encode"].append((t0, time.perf_counter()))
        pend = (hist_z, hist_a, z_next)
        hist_z = torch.cat([hist_z[1:], z_next.unsqueeze(0)], 0)
        z = z_next
    if pend is not None:                                    # last decision's deferred work
        t0c = time.perf_counter()
        dec["r_cur"].append(float(curiosity_reward(wm, *pend)[0]))
        dec["t_cur"].append((t0c, time.perf_counter()))
    done_steps = n_dec * AB

    wall = time.perf_counter() - t_run0
    temps1 = bus.read_temps()
    print(f"[jerk] {done_steps} substeps / {n_dec} decisions in {wall:.2f}s "
          f"({done_steps / wall:.1f} substeps/s vs 33.3 ideal)", flush=True)
    print(f"[jerk] temps after: {temps1.tolist()}", flush=True)

    out_dir = PROJECT_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"jerk_{args.name}_{done_steps}_det.npz"
    raw = bus.raw_reads[n_skip_reads:]
    np.savez_compressed(
        out,
        # per-substep state (N, ...)
        **{f"sub_{k}": np.asarray(v) for k, v in sub.items()},
        sub_qd_raw=np.asarray([r[1] for r in raw], np.float32),
        sub_tau_raw=np.asarray([r[2] for r in raw], np.float32),
        act_mode=np.asarray("det"),     # deterministic-only since 2026-06-12
        # per-substep bus/camera stage timestamps (N, 2), reset reads dropped
        write_t=np.asarray(bus.write_t, np.float64),
        read_t=np.asarray(bus.read_t[n_skip_reads:], np.float64),
        cam_t=np.asarray(cam.cam_t[n_skip_cams:], np.float64),
        goal=np.asarray(bus.goals, np.float32),
        pace=np.asarray(bus.paces, np.float32),
        # per-decision (D, ...)
        dec_t_sample=np.asarray(dec["t_sample"], np.float64),
        dec_t_encode=np.asarray(dec["t_encode"], np.float64),
        dec_t_cur=np.asarray(dec["t_cur"], np.float64),
        dec_r_cur=np.asarray(dec["r_cur"], np.float32),
        dec_action=np.asarray(dec["action"], np.float32),
        # run metadata
        dt_safe=np.float64(env.dt_safe), action_block=np.int64(AB),
        action_max=np.float64(args.action_max),
        temps_before=temps0, temps_after=temps1,
    )
    if not args.no_home:
        go_home(bus)
        print(f"[jerk] homed to {HOME.tolist()} (holding, torque ON)", flush=True)
    print(f"[jerk] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
