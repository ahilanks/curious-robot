"""Planning through the decoder's eye on a STAGED block-displacement scene -- DIAGNOSTIC ONLY.

Setup (user spec, 2026-09-17): drive the arm to a pose where the wrist camera faces two blocks,
photograph that view as the START, teleport ONE block (A) a few cm -- still in view -- from the
SAME arm pose and photograph the GOAL, restore, then run the frozen CEM act stack toward
z* = encode(goal photo) and watch what the planner imagines at every decision through the
post-hoc pixel decoder (model/decoder.py, tied to this checkpoint's encoder).

Per decision the strip is
  wrist t | decode(z_t) | decode(WM(z_t, plan)) | decode(WM(z_t, executed)) | wrist t+1 | decode(z*) | goal photo
where `plan` is the CEM plan in the units the planner SCORED (raw for an unscaled head; plan x
amax_frac for a --plan-act-scale head) and `executed` = clamp(plan) x amax_frac x dwell scale, the
action the arm actually takes.  For a scaled head the two imagined tiles coincide; for the campaign
head (act_scale 1, amax_frac 0.18) the gap between them IS the 09-17 exaggerated-imagination
finding, made visible.  `wrist t+1` is what really happened, next to what was imagined.

At decision 0 the CEM's iteration-0 candidate set (N(0, init_std) around the zero mean = what the
optimizer starts from, candidate 0 = the mean itself) is rolled through the WM, sorted by the
planner's own terminal cost, and the best / median / worst endpoints are decoded (candidates.png).
After the episode the WM's OPEN-LOOP imagination of the executed trajectory is laid against the
real frames (imagine.png; viz_decoder_rollout.imagine arithmetic).

Nothing here touches a gradient.  The act path mirrors eval_goal_photo.act_stack line for line
(not imported: the raw plan is needed for the decode); the staging mechanics are those of
probe_block_goal_learn.py (save/restore incl. MODEL-level geom size/colour; pixel-diff visibility
guard against the "block vanished" goal photo).

    python src/viz_plan_decoder.py --ckpt runs/wr_sleepret2/ckpt_0200000.pt \
        --decoder runs/wr_sleepret2/decoder_lewm.pt --amax-frac 0.18 --no-dwell --eps 0.5 \
        --budget 40 --shift 0.05 --color-a magenta --size-a 0.02 --out runs/plan_eye/head200k

Outputs (in --out): scene.png, candidates.png, episode.png (every --strip-every decision),
episode.mp4 (every decision), imagine.png, metrics.png, results.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import mujoco                                    # noqa: E402
import numpy as np                               # noqa: E402
import torch                                     # noqa: E402
from PIL import Image, ImageDraw, ImageFont      # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from env.mujoco_env import STANDARD_COLORS, MujocoSO101Env     # noqa: E402
from model.decoder import load_decoder                          # noqa: E402
from src.eval_goal_photo import build_wm                        # noqa: E402
from src.train import cem_plan, encode_obs                      # noqa: E402
from src.viz_decoder_rollout import imagine                     # noqa: E402

COLOR_NAMES = ["red", "green", "blue", "yellow", "magenta", "cyan", "orange", "purple", "white", "black"]
TILE, HDR = 224, 28
FAR = np.array([2.0, 2.0])          # off-table = "block removed" for the visibility diffs


# ----------------------------------------------------------------------------- drawing helpers
def _font(size: int):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


FONT, FONT_B = _font(11), _font(13)


def to_tile(img: np.ndarray) -> np.ndarray:
    if img.shape[0] != TILE or img.shape[1] != TILE:
        img = np.asarray(Image.fromarray(img).resize((TILE, TILE), Image.BILINEAR))
    return img


def labeled(img: np.ndarray, text: str) -> np.ndarray:
    """(224,224,3) -> (HDR+224, 224, 3): black header (up to two 11-px lines) over the tile."""
    canvas = Image.new("RGB", (TILE, TILE + HDR), (0, 0, 0))
    canvas.paste(Image.fromarray(to_tile(img)), (0, HDR))
    ImageDraw.Draw(canvas).text((3, 1), text, fill=(255, 255, 255), font=FONT)
    return np.asarray(canvas)


def strip(tiles, texts, banner: str = "") -> np.ndarray:
    row = np.concatenate([labeled(t, s) for t, s in zip(tiles, texts)], axis=1)
    if banner:
        b = Image.new("RGB", (row.shape[1], HDR), (40, 40, 40))
        ImageDraw.Draw(b).text((4, 5), banner, fill=(255, 255, 0), font=FONT_B)
        row = np.concatenate([np.asarray(b), row], axis=0)
    return row


def stack_rows(rows, gap: int = 6) -> np.ndarray:
    w = max(r.shape[1] for r in rows)
    out = []
    for r in rows:
        if r.shape[1] < w:
            r = np.concatenate([r, np.zeros((r.shape[0], w - r.shape[1], 3), np.uint8)], 1)
        out += [r, np.full((gap, w, 3), 90, np.uint8)]
    return np.concatenate(out[:-1], 0)


def save_png(img: np.ndarray, path: Path):
    Image.fromarray(img).save(path)


# ----------------------------------------------------------------------------- env helpers
def save_state(env):
    # MODEL-level object state too (geom_size / geom_rgba are rerolled in the model by reset), so a
    # restore reproduces the photographed scene bit for bit (probe_block_goal_learn, 2026-08-08).
    return dict(qpos=env.data.qpos.copy(), qvel=env.data.qvel.copy(),
                ctrl=env.data.ctrl.copy(), prev_ctrl=env._prev_ctrl.copy(),
                prev_qvel=env._prev_qvel.copy(), prev_obj=env._prev_obj_xpos.copy(),
                obj_size=env.model.geom_size[env._object_geom_ids].copy(),
                obj_rgba=env.model.geom_rgba[env._object_geom_ids].copy())


def restore_state(env, st):
    env.data.qpos[:] = st["qpos"]; env.data.qvel[:] = st["qvel"]
    env.data.ctrl[:] = st["ctrl"]
    env._prev_ctrl = st["prev_ctrl"].copy()
    env._prev_qvel = st["prev_qvel"].copy()
    env._prev_obj_xpos = st["prev_obj"].copy()
    env.model.geom_size[env._object_geom_ids] = st["obj_size"]
    env.model.geom_rgba[env._object_geom_ids] = st["obj_rgba"]
    mujoco.mj_forward(env.model, env.data)


def teleport(env, idx: int, xy):
    adr, vadr = env._object_qpos_addrs[idx], env._object_qvel_addrs[idx]
    env.data.qpos[adr:adr + 2] = xy
    env.data.qpos[adr + 2] = env._object_resting_z(idx)
    env.data.qvel[vadr:vadr + 6] = 0.0
    mujoco.mj_forward(env.model, env.data)


def block_xy(env, idx: int) -> np.ndarray:
    return env.data.xpos[env._object_body_ids[idx]][:2].copy()


def cam_geom(env):
    cid = env._wrist_cam_id
    half = float(np.tan(np.deg2rad(float(env.model.cam_fovy[cid])) / 2))
    return env.data.cam_xpos[cid].copy(), env.data.cam_xmat[cid].reshape(3, 3).copy(), half


def project(env, xyz):
    """-> (u, v) in half-FOV units (|u|,|v| < 1 = inside the square frustum), None = behind the cam."""
    cam, R, half = cam_geom(env)
    p = R.T @ (np.asarray(xyz, dtype=np.float64) - cam)
    if p[2] > -0.03:
        return None
    return p[0] / -p[2] / half, p[1] / -p[2] / half


def in_view(env, xyz, margin: float = 0.8) -> bool:
    uv = project(env, xyz)
    return uv is not None and max(abs(uv[0]), abs(uv[1])) < margin


def lookat_xy(env, z_plane: float):
    """Where the wrist camera's optical axis (-z of the camera frame) meets the table plane."""
    cam, R, _ = cam_geom(env)
    d = -R[:, 2]
    if d[2] > -0.05:                       # looking level or up: no table intersection
        return None
    t = (z_plane - cam[2]) / d[2]
    if t < 0.12 or t > 0.50:               # too close (block fills the frame) / too far
        return None
    fwd = d[:2] / (np.linalg.norm(d[:2]) + 1e-9)
    return cam[:2] + t * d[:2], fwd


def diff_px(a: np.ndarray, b: np.ndarray) -> int:
    """pixels differing by > 10 LSB in any channel (cube footprint > 30 LSB; AA speckle 1-5 LSB)."""
    return int((np.abs(a.astype(np.int16) - b.astype(np.int16)).max(-1) > 10).sum())


def reach_ok(xy) -> bool:
    r = float(np.linalg.norm(xy))
    return 0.18 <= r <= 0.36 and 0.14 <= xy[0] <= 0.42 and abs(xy[1]) <= 0.22


# ----------------------------------------------------------------------------- staging
def stage_scene(env, a0, args, rng):
    n_dof = env.n_dof
    lo, hi = env.model.jnt_range[:n_dof, 0], env.model.jnt_range[:n_dof, 1]
    ia, ib = args.idx_a, args.idx_b
    gid_a, gid_b = env._object_geom_ids[ia], env._object_geom_ids[ib]

    def settle(n_dec=3):
        for _ in range(n_dec * a0.action_block):
            env.step(np.zeros(n_dof, np.float32), render=False)

    def drive_to(q_target, n_dec=5):
        # servo through physics (target = clip(q + a * action_max)); no qpos teleport of the arm
        for _ in range(n_dec):
            q = env.data.qpos[:n_dof].copy()
            act = np.clip((q_target - q) / a0.action_max, -1, 1)
            for _ in range(a0.action_block):
                env.step(act.astype(np.float32), render=False)

    def photo():
        return env._get_obs()["image"].copy()

    for attempt in range(1, args.stage_attempts + 1):
        env.reset()
        if args.clear_others:
            for i in range(env.n_objects):
                if i not in (ia, ib):
                    teleport(env, i, FAR + [0.1 * i, 0.0])
        for gid, col, size in ((gid_a, args.color_a, args.size_a), (gid_b, args.color_b, args.size_b)):
            if col:
                env.model.geom_rgba[gid, :3] = STANDARD_COLORS[COLOR_NAMES.index(col)]
            if size > 0:
                env.model.geom_size[gid] = [size, size, size]
        drive_to(np.clip(rng.normal(0.0, 0.45, n_dof), lo * 0.85, hi * 0.85))
        la = lookat_xy(env, env._object_resting_z(ia))
        if la is None:
            continue
        xy_a, fwd = la
        if not reach_ok(xy_a):
            continue
        side = np.array([-fwd[1], fwd[0]])
        xy_b = next((c for c in (xy_a + args.sep * side, xy_a - args.sep * side) if reach_ok(c)), None)
        if xy_b is None:
            continue
        teleport(env, ia, xy_a); teleport(env, ib, xy_b)
        settle()
        if np.linalg.norm(block_xy(env, ia) - xy_a) > 0.01 or np.linalg.norm(block_xy(env, ib) - xy_b) > 0.01:
            continue                                            # a block got launched (spawned into the arm)
        # after mj_step the kinematics (xpos) lag the integrated qpos by one timestep; a forward pass
        # first makes the START photo a function of qpos alone, so restore + mj_forward reproduces it exactly
        mujoco.mj_forward(env.model, env.data)
        start_px = photo()
        snap = save_state(env)
        prop = env._get_obs(render=False)["proprio"].copy()
        b0, b1 = block_xy(env, ia), block_xy(env, ib)
        ov_start = env.render_overhead()
        teleport(env, ia, FAR); px_no_a = photo(); restore_state(env, snap)
        teleport(env, ib, FAR); px_no_b = photo(); restore_state(env, snap)
        vis_a, vis_b = diff_px(start_px, px_no_a), diff_px(start_px, px_no_b)
        if vis_a < args.min_px or vis_b < args.min_px:
            print(f"[stage] attempt {attempt}: blocks not both visible (A {vis_a} px, B {vis_b} px)", flush=True)
            continue
        chosen = None
        for dvec in (side, -side, fwd, -fwd, (side + fwd) / math.sqrt(2), (side - fwd) / math.sqrt(2),
                     (-side + fwd) / math.sqrt(2), (-side - fwd) / math.sqrt(2)):
            b_goal = b0 + args.shift * dvec
            if not reach_ok(b_goal) or np.linalg.norm(b_goal - b1) < max(0.045, 0.6 * args.sep):
                continue                                        # out of the reach band / would shove A into B
            if not in_view(env, [*b_goal, env._object_resting_z(ia)]):
                continue
            teleport(env, ia, b_goal)
            goal_px, ov_goal = photo(), env.render_overhead()
            restore_state(env, snap)
            moved, target_vis = diff_px(goal_px, start_px), diff_px(goal_px, px_no_a)
            if moved > args.min_px and target_vis > args.min_px:
                chosen = dict(b_goal=b_goal, goal_px=goal_px, ov_goal=ov_goal, moved_px=moved, target_px=target_vis)
                break
        if chosen is None:
            print(f"[stage] attempt {attempt}: no in-frame shift direction for block A", flush=True)
            continue
        restore_px = diff_px(photo(), start_px)
        if restore_px:
            print(f"[stage] WARNING: restore differs from the start photo by {restore_px} px (> 10 LSB)", flush=True)
        others = [i for i in range(env.n_objects) if i not in (ia, ib)
                  and in_view(env, env.data.xpos[env._object_body_ids[i]], 1.0)]
        sc = dict(snap=snap, prop=prop, start_px=start_px, ov_start=ov_start, b0=b0, b1=b1,
                  vis_a=vis_a, vis_b=vis_b, others_in_view=others, attempt=attempt,
                  size_a=float(env.model.geom_size[gid_a, 0]), size_b=float(env.model.geom_size[gid_b, 0]),
                  rgba_a=env.model.geom_rgba[gid_a].tolist(), rgba_b=env.model.geom_rgba[gid_b].tolist(), **chosen)
        print(f"[stage] attempt {attempt}: A ({b0[0]:+.3f},{b0[1]:+.3f}) -> goal ({sc['b_goal'][0]:+.3f},"
              f"{sc['b_goal'][1]:+.3f})  B ({b1[0]:+.3f},{b1[1]:+.3f})  vis A {vis_a} B {vis_b} px  "
              f"moved {sc['moved_px']} target {sc['target_px']} px  other objects in view: {others}", flush=True)
        return sc
    raise SystemExit(f"[stage] no scene in {args.stage_attempts} attempts")


# ----------------------------------------------------------------------------- planning helpers
def imagine_next(wm, predict, hist_z, hist_a, act):
    """WM one-step prediction from the REAL Hb-latent history under action(s) `act` (M, a_dim), the
    exact context arithmetic cem_plan scores: z_seq[:, -Hb:] and the last Hb actions incl. the candidate."""
    Hb, M = hist_z.shape[0], act.shape[0]
    zc = hist_z[:, 0].unsqueeze(0).expand(M, Hb, -1)
    ac = torch.cat([hist_a[1:, 0].unsqueeze(0).expand(M, Hb - 1, -1), act.unsqueeze(1)], 1)
    return predict(zc, wm.action_encoder(ac))[:, -1]


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    env = MujocoSO101Env(frame_skip=ck["args"]["frame_skip"], action_max=ck["args"]["action_max"],
                         encode_cam=ck["args"]["wm_cam"], safety_delta=ck["args"]["safety_delta"],
                         seed=args.seed, fixed_objects=args.fixed_objects)
    n_dof = env.n_dof
    wm, a = build_wm(ck, n_dof, device)
    if args.horizon:
        a.cem_horizon = args.horizon
    if args.no_dwell:
        a.dwell_hold_mult = 0.0; a.dwell_shrink_start = 0.0
    if args.eps > 0:
        a.goal_reach_eps = args.eps
    dec, dmeta = load_decoder(args.decoder, device, z_dim=wm.z_dim)
    act_scale = args.amax_frac if getattr(a, "plan_act_scale", False) else 1.0
    predict = getattr(wm, "predict_eager", wm.predict)
    a_dim, H, eps = n_dof * a.action_block, a.history_size, a.goal_reach_eps
    print(f"[plan-eye] ckpt {args.ckpt} step {ck.get('step')}  decoder {args.decoder} "
          f"(val_mse {dmeta.get('val_mse', float('nan')):.4f})  CEM {a.cem_samples}x{a.cem_iters} H={a.cem_horizon} "
          f"init_std {a.cem_init_std}  act_scale {act_scale}  amax_frac {args.amax_frac}  eps {eps}  "
          f"dwell hold {a.dwell_hold_mult} shrink {a.dwell_shrink_start}", flush=True)

    sc = stage_scene(env, a, args, rng)
    restore_state(env, sc["snap"])
    obs = env._get_obs()
    b_goal, gap0 = sc["b_goal"], float(np.linalg.norm(sc["b0"] - sc["b_goal"]))

    with torch.no_grad():
        zstar = encode_obs(wm, sc["goal_px"][None], sc["prop"][None], device)
        z = encode_obs(wm, obs["image"][None], obs["proprio"][None], device)
        d0 = float((z - zstar).norm(dim=-1)[0])
        dec0 = dec.to_uint8_hwc(torch.cat([z, zstar]))
    moved_mask = (np.abs(sc["goal_px"].astype(np.int16) - sc["start_px"].astype(np.int16)).max(-1) > 10)
    diff_img = np.repeat((moved_mask * 255).astype(np.uint8)[..., None], 3, -1)
    scene_png = strip(
        [sc["start_px"], sc["goal_px"], diff_img, sc["ov_start"], sc["ov_goal"], dec0[0], dec0[1]],
        [f"START wrist (real)\nA vis {sc['vis_a']}px B vis {sc['vis_b']}px",
         f"GOAL photo: A moved {args.shift * 1000:.0f}mm\nmoved {sc['moved_px']}px target {sc['target_px']}px",
         "|goal - start| > 10 LSB\n(what changed in the photo)",
         f"overhead START\nA ({sc['b0'][0]:+.2f},{sc['b0'][1]:+.2f}) B ({sc['b1'][0]:+.2f},{sc['b1'][1]:+.2f})",
         f"overhead GOAL\nA -> ({b_goal[0]:+.2f},{b_goal[1]:+.2f})",
         f"decode(z_0)\nrecon L1 {np.abs(dec0[0].astype(np.int16) - sc['start_px'].astype(np.int16)).mean():.1f}",
         f"decode(z*)\n||z_0 - z*|| = {d0:.2f} (eps {eps})"],
        banner=f"{args.name}: staged scene (attempt {sc['attempt']}, seed {args.seed})  block A {COLOR_NAMES[int(np.argmin(((STANDARD_COLORS - np.array(sc['rgba_a'][:3])) ** 2).sum(1)))]} "
               f"{sc['size_a'] * 200:.1f}cm  block B {COLOR_NAMES[int(np.argmin(((STANDARD_COLORS - np.array(sc['rgba_b'][:3])) ** 2).sum(1)))]} {sc['size_b'] * 200:.1f}cm  "
               f"other objects in view {sc['others_in_view']}")
    save_png(scene_png, out / "scene.png")
    np.savez_compressed(out / "scene.npz", start_px=sc["start_px"], goal_px=sc["goal_px"],
                        b0=sc["b0"], b1=sc["b1"], b_goal=b_goal, prop=sc["prop"])
    print(f"[plan-eye] d0 = {d0:.3f} (eps {eps}); block A gap {gap0 * 1000:.0f} mm -> {out / 'scene.png'}", flush=True)

    hist_z = z.unsqueeze(0).repeat(H, 1, 1)
    hist_a = torch.zeros(H, 1, a_dim, device=device)
    frames, props, acts, overheads, rows, strips = [obs["image"].copy()], [obs["proprio"].copy()], [], [sc["ov_start"]], [], []
    import imageio
    writer = imageio.get_writer(out / "episode.mp4", fps=args.fps, codec="libx264", quality=8, macro_block_size=None)
    min_gap = gap0
    for t in range(args.budget):
        diag = {}
        with torch.no_grad():
            plan = cem_plan(wm, hist_z, hist_a, zstar, a.cem_samples, a.cem_iters, a.cem_elites,
                            a.cem_init_std, a.cem_horizon, device, diag=diag, gamma=a.cem_gamma,
                            min_std=a.cem_min_std, mppi_temp=a.cem_mppi_temp,
                            early_stop_tol=a.cem_early_stop, early_stop_min_iters=a.cem_early_stop_min_iters,
                            act_scale=act_scale)
            raw = plan[:, 0]                                        # (1, a_dim) planner units
            act = raw.clamp(-1.0, 1.0) * args.amax_frac             # executed units (eval_goal_photo.act_stack)
            gd = float((z - zstar).norm(dim=-1)[0])
            shrink, hold = 1.0, False
            if a.dwell_shrink_start > 0:
                shrink = float(min(max(gd / (a.dwell_shrink_start * eps), a.dwell_shrink_min), 1.0))
                act = act * shrink
            if a.dwell_hold_mult > 0 and gd < a.dwell_hold_mult * eps:
                act, hold = torch.zeros_like(act), True
            z_plan = imagine_next(wm, predict, hist_z, hist_a, raw * act_scale)   # what CEM scored
            z_exec = imagine_next(wm, predict, hist_z, hist_a, act)               # what the WM expects of the real step
            d_plan, d_exec = float((z_plan - zstar).norm()), float((z_exec - zstar).norm())
            # iteration-0 candidate set (zero mean, init_std): the planner's starting distribution
            cand = torch.randn(args.n_cand, a_dim, device=device) * a.cem_init_std
            cand[0] = 0.0
            zc = imagine_next(wm, predict, hist_z, hist_a, cand * act_scale)
            cd = (zc - zstar).norm(dim=-1)
            zero_drift = float((zc[0] - z[0]).norm())          # ||WM(z_t, a=0) - z_t||: the WM's own one-step drift
            order = torch.argsort(cd)
            if t == 0:
                pick = [*order[:4].tolist(), *order[args.n_cand // 2 - 1: args.n_cand // 2 + 1].tolist(), *order[-4:].tolist()]
                imgs = dec.to_uint8_hwc(torch.cat([z, z_plan, zstar, zc[pick], zc[0:1]]))
                texts = [f"decode(z_0)\n||z_0 - z*|| {gd:.2f}",
                         f"CEM plan endpoint\nd* {d_plan:.2f} |plan| {float(raw.norm()):.2f}",
                         f"decode(z*)\ncand d*: min {float(cd.min()):.2f} med {float(cd.median()):.2f} max {float(cd.max()):.2f}"]
                for j, i in enumerate(pick):
                    rank = "best" if j < 4 else ("median" if j < 6 else "worst")
                    texts.append(f"{rank} #{int(i)}  d* {float(cd[i]):.2f}\n|a| {float(cand[i].norm()):.2f}  a_max {float(cand[i].abs().max()):.2f}")
                texts.append(f"zero action (cand 0)\nd* {float(cd[0]):.2f}")
                row1 = strip(list(imgs[:7]), texts[:7],
                             banner=f"{args.name}: decision 0 -- CEM iteration-0 candidates ({args.n_cand} x N(0,{a.cem_init_std}) x act_scale {act_scale}), decoded endpoints sorted by the planner's cost")
                row2 = strip(list(imgs[7:]), texts[7:])
                save_png(stack_rows([row1, row2]), out / "candidates.png")
                print(f"[plan-eye] candidates.png: d* over {args.n_cand} iteration-0 candidates min {float(cd.min()):.3f} "
                      f"median {float(cd.median()):.3f} max {float(cd.max()):.3f}; zero action {float(cd[0]):.3f}; "
                      f"converged plan {d_plan:.3f}; reach gap {gd:.3f}", flush=True)
            dec_now = dec.to_uint8_hwc(torch.cat([z, z_plan, z_exec]))
        # execute the block-action through physics
        hist_a = torch.cat([hist_a[1:], act.unsqueeze(0)], 0)
        subs = act[0].detach().cpu().numpy().reshape(a.action_block, n_dof)
        nc = 0
        for k, sub in enumerate(np.clip(subs, -1, 1)):
            obs, info = env.step(sub.astype(np.float32), render=(k == a.action_block - 1))
            nc += int(info["object_contacts"])
        with torch.no_grad():
            z = encode_obs(wm, obs["image"][None], obs["proprio"][None], device)
            dec_next = dec.to_uint8_hwc(z)[0]
            err_plan, err_exec = float((z_plan - z).norm()), float((z_exec - z).norm())   # imagined vs encoded real next
        hist_z = torch.cat([hist_z[1:], z.unsqueeze(0)], 0)
        d_after = float((z - zstar).norm(dim=-1)[0])
        gap_a = float(np.linalg.norm(block_xy(env, args.idx_a) - b_goal))
        disp_b = float(np.linalg.norm(block_xy(env, args.idx_b) - sc["b1"]))
        min_gap = min(min_gap, gap_a)
        recon_l1 = float(np.abs(dec_now[0].astype(np.int16) - frames[-1].astype(np.int16)).mean())
        row = dict(t=t, d=gd, d_after=d_after, d_plan=d_plan, d_exec=d_exec, cand_min=float(cd.min()),
                   cand_med=float(cd.median()), cand_zero=float(cd[0]), plan_norm=float(raw.norm()),
                   plan_absmax=float(raw.abs().max()), act_norm=float(act.norm()), shrink=shrink, hold=hold,
                   contacts=nc, gap_a_mm=gap_a * 1000, disp_b_mm=disp_b * 1000, recon_l1=recon_l1,
                   zero_drift=zero_drift, err_plan=err_plan, err_exec=err_exec,
                   cem_iters=diag.get("iters_used", float("nan")), z_term_spread=diag.get("z_term_spread", float("nan")))
        rows.append(row)
        s = strip(
            [frames[-1], dec_now[0], dec_now[1], dec_now[2], dec_next, obs["image"], dec0[1], sc["goal_px"]],
            [f"t={t} wrist (real)\n||z-z*|| {gd:.2f}  eps {eps}",
             f"decode(z_t)\nrecon L1 {recon_l1:.1f}  WM(z,0) drift {zero_drift:.2f}",
             f"imagined: plan x{act_scale:g}\n|plan| {float(raw.norm()):.2f}  d* {d_plan:.2f}  err {err_plan:.2f}",
             f"imagined: executed x{args.amax_frac:g}{' x' + format(shrink, '.2f') if shrink < 1 else ''}{'  HOLD' if hold else ''}\n|a| {float(act.norm()):.2f}  d* {d_exec:.2f}  err {err_exec:.2f}",
             f"decode(z_t+1) real next\n||z-z*|| {d_after:.2f}",
             f"t={t + 1} wrist (real)\ncontacts {nc}",
             f"decode(z*)\ncand d* min {float(cd.min()):.2f} med {float(cd.median()):.2f}",
             f"goal photo\nA gap {gap_a * 1000:.0f}mm  B moved {disp_b * 1000:.0f}mm"],
            banner=f"{args.name}  decision {t:3d}   d {gd:.2f} -> {d_after:.2f} (eps {eps})   imagined d* plan {d_plan:.2f} / exec {d_exec:.2f}   "
                   f"block A gap {gap0 * 1000:.0f} -> {gap_a * 1000:.0f} mm (best {min_gap * 1000:.0f})   B moved {disp_b * 1000:.0f} mm   contacts {nc}   "
                   f"WM(z,0) drift {zero_drift:.2f}   CEM iters {row['cem_iters']:.0f}")
        writer.append_data(s)
        if t % args.strip_every == 0:
            strips.append(s)
        frames.append(obs["image"].copy()); props.append(obs["proprio"].copy()); acts.append(act[0].cpu().numpy())
        overheads.append(env.render_overhead())
        print(f"[t {t:3d}] d {gd:6.2f} -> {d_after:6.2f}  imagined d* plan {d_plan:6.2f} exec {d_exec:6.2f}  "
              f"cand min/med {float(cd.min()):5.2f}/{float(cd.median()):5.2f}  drift0 {zero_drift:4.2f}  err plan/exec {err_plan:5.2f}/{err_exec:5.2f}  |plan| {float(raw.norm()):5.2f} |a| {float(act.norm()):5.2f}"
              f"{' HOLD' if hold else ''}  A gap {gap_a * 1000:5.1f}mm  B {disp_b * 1000:4.1f}mm  con {nc:2d}", flush=True)
    writer.close()
    save_png(stack_rows(strips), out / "episode.png")

    # open-loop imagination of the executed trajectory vs the real frames (LeWM Fig. 7 arithmetic)
    T = min(args.imagine_T, len(acts) - H)
    if T > 0:
        fr = np.stack(frames[:H + T]); pr = np.stack(props[:H + T]).astype(np.float32); ac = np.stack(acts[:H + T]).astype(np.float32)
        with torch.no_grad():
            z_real, z_imag = imagine(wm, fr, pr, ac, H, T, device)
            dec_imag = dec.to_uint8_hwc(z_imag)
            dec_real = dec.to_uint8_hwc(z_real)
        z_mse = ((z_imag[H:] - z_real[H:]) ** 2).mean(-1).cpu().numpy()
        bar = np.full((HDR + TILE, 4, 3), 255, np.uint8)
        top = strip(list(fr), [f"real t={i}" + ("  (context)" if i < H else "") for i in range(H + T)])
        mid = strip(list(dec_real), [f"decode(z_real t={i})" for i in range(H + T)])
        bot = strip(list(dec_imag), [f"decode({'z_real' if i < H else 'imagined'} t={i})" + (f"\nz_mse {z_mse[i - H]:.4f}" if i >= H else "") for i in range(H + T)])
        ins = lambda r: np.concatenate([r[:, :H * TILE], bar, r[:, H * TILE:]], 1)
        save_png(stack_rows([ins(top), ins(mid), ins(bot)]),  out / "imagine.png")
        print(f"[plan-eye] imagine.png: {H} context + {T} open-loop steps under the executed actions; "
              f"z_mse per step " + " ".join(f"{v:.4f}" for v in z_mse), flush=True)

    # overhead strip of the episode (every strip_every decision) for the physical story
    ov = [labeled(overheads[i], f"overhead after decision {i - 1}" if i else "overhead start")
          for i in range(0, len(overheads), args.strip_every)]
    save_png(np.concatenate(ov, 1), out / "overhead.png")

    # metrics chart
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ts = [r["t"] for r in rows]
    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax[0].plot(ts, [r["d"] for r in rows], color="#1f77b4", lw=2, label="||z_t - z*|| (real)")
    ax[0].plot(ts, [r["d_plan"] for r in rows], color="#ff7f0e", lw=1.2, ls="--", label=f"imagined d*: plan x{act_scale:g}")
    ax[0].plot(ts, [r["d_exec"] for r in rows], color="#2ca02c", lw=1.2, ls="--", label=f"imagined d*: executed x{args.amax_frac:g}")
    ax[0].plot(ts, [r["cand_min"] for r in rows], color="#7f7f7f", lw=0.8, ls=":", label="best iteration-0 candidate d*")
    ax[0].plot(ts, [r["zero_drift"] for r in rows], color="#8c564b", lw=0.8, ls=":", label="||WM(z_t, a=0) - z_t|| (WM drift)")
    ax[0].axhline(eps, color="k", lw=0.8, ls="-.", label=f"eps {eps}")
    ax[0].set_ylabel("latent distance"); ax[0].legend(fontsize=8, ncol=2); ax[0].set_title(f"{args.name}: planning toward the displaced-block photo", fontsize=10)
    ax[1].plot(ts, [r["gap_a_mm"] for r in rows], color="#d62728", lw=2, label="block A gap to goal (mm)")
    ax[1].plot(ts, [r["disp_b_mm"] for r in rows], color="#9467bd", lw=1.2, label="block B displacement (mm)")
    ax[1].bar(ts, [r["contacts"] for r in rows], color="#bbbbbb", width=0.8, label="object contacts (sub-steps)")
    ax[1].axhline(gap0 * 1000, color="#d62728", lw=0.8, ls=":")
    ax[1].set_xlabel("decision"); ax[1].set_ylabel("mm / contacts"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "metrics.png", dpi=120); plt.close(fig)

    summary = dict(name=args.name, ckpt=args.ckpt, step=ck.get("step"), decoder=args.decoder, seed=args.seed,
                   amax_frac=args.amax_frac, act_scale=act_scale, eps=eps, no_dwell=args.no_dwell,
                   budget=args.budget, shift_m=args.shift, sep_m=args.sep, fixed_objects=args.fixed_objects,
                   scene=dict(b0=sc["b0"].tolist(), b1=sc["b1"].tolist(), b_goal=b_goal.tolist(), vis_a_px=sc["vis_a"],
                              vis_b_px=sc["vis_b"], moved_px=sc["moved_px"], target_px=sc["target_px"],
                              others_in_view=sc["others_in_view"], size_a=sc["size_a"], size_b=sc["size_b"],
                              rgba_a=sc["rgba_a"], rgba_b=sc["rgba_b"], attempt=sc["attempt"]),
                   d0=d0, gap0_mm=gap0 * 1000, min_gap_mm=min_gap * 1000, closure_mm=(gap0 - min_gap) * 1000,
                   min_d=min(min(r["d"], r["d_after"]) for r in rows), final_d=rows[-1]["d_after"],
                   total_contacts=int(sum(r["contacts"] for r in rows)), rows=rows)
    (out / "results.json").write_text(json.dumps(summary, indent=1))
    env.close()
    print(f"[plan-eye] DONE  d0 {d0:.2f} -> min {summary['min_d']:.2f} (final {summary['final_d']:.2f}, eps {eps})   "
          f"block A gap {gap0 * 1000:.0f} -> best {min_gap * 1000:.0f} mm (closure {summary['closure_mm']:+.0f} mm)   "
          f"contacts {summary['total_contacts']}   -> {out}/", flush=True)
    return summary


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="runs/wr_sleepret2/ckpt_0200000.pt")
    p.add_argument("--decoder", default="runs/wr_sleepret2/decoder_lewm.pt")
    p.add_argument("--name", default="plan_eye")
    p.add_argument("--out", default="runs/plan_eye/head200k")
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--budget", type=int, default=40, help="planner decisions to run and render")
    p.add_argument("--amax-frac", type=float, default=1.0,
                   help="the run's amplitude-curriculum fraction at this ckpt (executed = clamp(plan) x frac; "
                        "a --plan-act-scale head also rolls the WM out at that scale). wr_sleepret2@200k ~ 0.18")
    p.add_argument("--eps", type=float, default=0.0, help="override goal_reach_eps (0 = ckpt value)")
    p.add_argument("--no-dwell", action="store_true", help="disable dwell HOLD/SHRINK (block-shift goals sit inside the deployed hold radius)")
    p.add_argument("--horizon", type=int, default=0, help="override cem_horizon (0 = ckpt value)")
    p.add_argument("--shift", type=float, default=0.05, help="block A displacement for the goal photo (m)")
    p.add_argument("--sep", type=float, default=0.07, help="block B offset from A, lateral to the camera axis (m)")
    p.add_argument("--min-px", type=int, default=300, help="pixel-diff visibility threshold (>10 LSB pixels)")
    p.add_argument("--idx-a", type=int, default=0, help="object index used as block A (the displaced one)")
    p.add_argument("--idx-b", type=int, default=1, help="object index used as block B (the anchor)")
    p.add_argument("--color-a", default="", choices=[""] + COLOR_NAMES, help="force block A's colour (default: the layout's)")
    p.add_argument("--color-b", default="", choices=[""] + COLOR_NAMES)
    p.add_argument("--size-a", type=float, default=0.0, help="force block A's half-size in m (layout range 0.012-0.020)")
    p.add_argument("--size-b", type=float, default=0.0)
    p.add_argument("--fixed-objects", action=argparse.BooleanOptionalAction, default=True,
                   help="the ckpt's training layout (fixed_objects) for the other 8 objects; --no-fixed-objects re-rolls them")
    p.add_argument("--clear-others", action="store_true", help="push the 8 other objects off the table (only A and B remain)")
    p.add_argument("--stage-attempts", type=int, default=400)
    p.add_argument("--n-cand", type=int, default=200, help="iteration-0 candidates decoded/scored (ckpt cem_samples = 200)")
    p.add_argument("--strip-every", type=int, default=4, help="episode.png keeps every k-th decision strip (mp4 keeps all)")
    p.add_argument("--fps", type=float, default=3.0)
    p.add_argument("--imagine-T", type=int, default=8, help="open-loop imagined steps in imagine.png")
    return p.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
