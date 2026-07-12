"""Parent-child VLA orchestration: a second (parent) SO-101 driven by a VLA moves blocks
INSIDE the child's wrist-camera view, so the child's from-scratch SSL world model gets to
watch object dynamics it cannot yet produce itself.

Three pieces:
  * SmolVLADriver     -- lerobot SmolVLA wrapper: sim obs -> policy batch -> parent joint
                         targets (radians). Handles unit conversion (dataset degrees <->
                         sim radians) and the policy's internal 50-action chunk queue.
  * blocks_in_child_view -- gaze geometry: project block positions through the CHILD's
                         wrist camera (fovy + extrinsics from mj_data) -> which blocks the
                         child is actually looking at, with image-centre distance.
  * InstructionGen    -- turns the gazed-at block into a language command for the VLA
                         ("push the red cube to the left..."), re-issued on a cadence or
                         when the target leaves the child's view.

The orchestrator (see run_parent_child_demo) keeps the parent acting on whatever the child
currently watches -- the "parent plays where the child looks" contract of the setup.
"""
from __future__ import annotations

import numpy as np
import torch

# ---------------------------------------------------------------- gaze geometry
_COLOR_WORDS = {
    "red": (1.0, 0.1, 0.1), "green": (0.1, 0.8, 0.2), "blue": (0.15, 0.3, 0.9),
    "yellow": (0.95, 0.85, 0.1), "orange": (0.95, 0.55, 0.1), "purple": (0.6, 0.2, 0.8),
    "white": (0.95, 0.95, 0.95), "pink": (0.95, 0.5, 0.7), "gray": (0.5, 0.5, 0.5),
}


def color_word(rgba) -> str:
    rgb = np.asarray(rgba[:3], np.float64)
    dists = {w: float(((rgb - np.asarray(c)) ** 2).sum()) for w, c in _COLOR_WORDS.items()}
    return min(dists, key=dists.get)


def blocks_in_child_view(env, margin: float = 0.92):
    """Blocks inside the CHILD's wrist-cam frustum right now. Returns a list of dicts
    {idx, center (0=image centre..1=edge), color, xpos} sorted by centredness. `margin`
    trims the frustum edge so 'in view' means comfortably in frame, not clipping it."""
    import mujoco
    cid = env._wrist_cam_id
    cam_pos = env.data.cam_xpos[cid]
    R = env.data.cam_xmat[cid].reshape(3, 3)             # world<-cam columns
    half = np.tan(np.deg2rad(float(env.model.cam_fovy[cid])) / 2)
    out = []
    for i, bid in enumerate(env._object_body_ids):
        p = R.T @ (env.data.xpos[bid] - cam_pos)         # cam frame; looks along -z
        if p[2] > -0.03:                                 # behind or on the camera plane
            continue
        u, v = p[0] / -p[2], p[1] / -p[2]
        if abs(u) < half * margin and abs(v) < half * margin:
            rgba = env.model.geom_rgba[env._object_geom_ids[i]]
            out.append({"idx": i, "center": float(max(abs(u), abs(v)) / half),
                        "color": color_word(rgba), "xpos": env.data.xpos[bid].copy()})
    return sorted(out, key=lambda d: d["center"])


# ---------------------------------------------------------------- instructions
class InstructionGen:
    """Language commands for the VLA, aimed at whatever the child watches. Templates are
    phrased like SO-10x community-dataset tasks (short imperative pick/push phrasing)."""
    TEMPLATES = (
        "pick up the {color} cube and place it to the left",
        "pick up the {color} cube and place it to the right",
        "push the {color} cube forward",
        "slide the {color} cube closer to the other blocks",
    )

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self._k = 0

    def instruction_for(self, block) -> str:
        t = self.TEMPLATES[self._k % len(self.TEMPLATES)]
        self._k += 1
        return t.format(color=block["color"])


# ---------------------------------------------------------------- SmolVLA driver
class SmolVLADriver:
    """lerobot SmolVLA -> parent joint targets. select_action keeps an internal chunk
    queue (n_action_steps=50), so calling it once per sim sub-step (~30 Hz, the SO-10x
    dataset rate) costs one real inference per ~1.6 s of sim time.
    units: empirically smolvla_base outputs RADIAN-scale joint targets on SO-101
    (verified 2026-07-10: rad-interpretation gives 30-74 deg coherent chunks that differ
    by instruction; deg-interpretation shrinks them 57x into a fake 'stay-still collapse')."""

    def __init__(self, device="cuda", units: str = "rad", model_id: str = "lerobot/smolvla_base"):
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        self.policy = SmolVLAPolicy.from_pretrained(model_id).to(device).eval()
        self.device = device
        self.units = units
        self.cam_keys = [k for k in self.policy.config.input_features if "image" in k]
        self.n_act = int(np.prod(self.policy.config.output_features["action"].shape))
        # select_action expects PRE-tokenized language (observation.language.tokens/
        # attention_mask) -- tokenize with the VLM's own tokenizer, once per instruction.
        self._tok = self.policy.model.vlm_with_expert.processor.tokenizer
        self._max_len = int(getattr(self.policy.config, "tokenizer_max_length", 48))

    def reset(self, instruction: str):
        self.policy.reset()
        self.instruction = instruction
        enc = self._tok(instruction, max_length=self._max_len, truncation=True,
                        padding="max_length", return_tensors="pt")
        self._lang_tokens = enc["input_ids"].to(self.device)
        self._lang_mask = enc["attention_mask"].to(self.device, dtype=torch.bool)

    @staticmethod
    def _img(t: np.ndarray, device) -> torch.Tensor:
        x = torch.as_tensor(t, device=device).float().permute(2, 0, 1) / 255.0
        return x.unsqueeze(0)                        # (1,3,H,W) in [0,1]

    def _state_out(self, q_rad: np.ndarray) -> np.ndarray:
        if self.units == "deg":
            return np.rad2deg(q_rad).astype(np.float32)
        return q_rad.astype(np.float32)

    def _action_in(self, a: np.ndarray) -> np.ndarray:
        if self.units == "deg":
            return np.deg2rad(a).astype(np.float64)
        return a.astype(np.float64)

    def needs_obs(self) -> bool:
        """True when the next act() will REFILL the 50-action chunk (i.e. render now).
        Renders are the expensive part (~0.35s osmesa for two cams); the chunk is
        open-loop from ONE observation, so pop-only calls can reuse stale frames."""
        from lerobot.utils.constants import ACTION
        return len(self.policy._queues[ACTION]) == 0

    @torch.no_grad()
    def act(self, cams: list[np.ndarray] | None, parent_qpos_rad: np.ndarray) -> np.ndarray:
        """cams: up to 3 HxWx3 uint8 frames (missing slots zero-padded); may be None on
        pop-only calls (see needs_obs) -- the cached frames are reused. Returns absolute
        parent joint targets in RADIANS."""
        if cams is not None:
            self._cams = cams
        batch = {"observation.state": torch.as_tensor(
                     self._state_out(parent_qpos_rad), device=self.device).unsqueeze(0),
                 "task": [self.instruction],
                 "observation.language.tokens": self._lang_tokens,
                 "observation.language.attention_mask": self._lang_mask}
        for k, key in enumerate(self.cam_keys):
            img = self._cams[k] if k < len(self._cams) and self._cams[k] is not None else \
                np.zeros((256, 256, 3), np.uint8)
            batch[key] = self._img(img, self.device)
        a = self.policy.select_action(batch)[0].float().cpu().numpy()[: self.n_act]
        return self._action_in(a)


# ---------------------------------------------------------------- scripted driver
class ScriptedParentDriver:
    """Privileged keyframe pick/push driver with the SAME interface as SmolVLADriver --
    the reliability baseline for the VLA comparison and the fallback parent when
    zero-shot VLAs collapse on sim renders. Needs the env handle (block positions);
    that's fine: it is a sim data collector, not a deployable policy.

    Cycle per instruction: hover over the target block -> descend -> sweep THROUGH it
    (direction from the instruction template) -> lift -> re-hover. Joint targets from a
    2-keyframe reach calibration of the parent pose (base at +x facing -x)."""

    def __init__(self, env, sweep_deg: float = 22.0):
        self.env = env
        self.sweep = np.deg2rad(sweep_deg)
        self._t = 0
        self._target_idx = None
        self._dir = +1.0

    def needs_obs(self) -> bool:
        return False

    def reset(self, instruction: str):
        self.instruction = instruction
        self._t = 0
        self._dir = -1.0 if ("left" in instruction) else +1.0
        self._target_idx = None

    def track(self, block_idx: int | None):
        self._target_idx = block_idx

    def _aim(self):
        env = self.env
        if self._target_idx is None:
            return None
        b = env.data.xpos[env._object_body_ids[self._target_idx]]
        rel = b[:2] - np.array([0.62, 0.0])            # parent base frame (world)
        r = float(np.linalg.norm(rel))
        # parent faces -x (yawed pi): its pan zero points at the child; pan angle from -x axis
        pan = float(np.arctan2(-rel[1], -rel[0]))
        # reach calibration: r~0.16 -> (lift,elbow)=(-0.75,1.15); r~0.30 -> (-1.05,1.45)
        f = np.clip((r - 0.16) / 0.14, 0.0, 1.4)
        lift = -0.75 - 0.30 * f
        elbow = 1.15 + 0.30 * f
        return pan, lift, elbow

    def act(self, cams, parent_qpos_rad: np.ndarray) -> np.ndarray:
        aim = self._aim()
        if aim is None:
            return np.zeros(6)
        pan, lift, elbow = aim
        T = 90                                          # sub-steps per full cycle (~3 s)
        ph = (self._t % T) / T
        self._t += 1
        wrist = 0.5
        if ph < 0.25:                                   # hover above, panned slightly past
            q = [pan - self._dir * self.sweep, lift + 0.35, elbow - 0.2, wrist, 0.0, 0.5]
        elif ph < 0.45:                                 # descend to contact height
            q = [pan - self._dir * self.sweep, lift, elbow, wrist, 0.0, 0.3]
        elif ph < 0.75:                                 # sweep THROUGH the block
            s = (ph - 0.45) / 0.30
            q = [pan + self._dir * self.sweep * (2 * s - 1), lift, elbow, wrist, 0.0, 0.3]
        else:                                           # lift + return
            q = [pan + self._dir * self.sweep, lift + 0.35, elbow - 0.25, wrist, 0.0, 0.5]
        return np.asarray(q, np.float64)


# ---------------------------------------------------------------- training fleet
class ParentFleet:
    """Batched parent driver for the N-env TRAINING loop. Per-env 50-action chunk queues;
    when any env runs low, ONE parent_context() round-trip + ONE batched
    predict_action_chunk refills them all together (per-env gaze-aware instructions
    re-issued at each refill). next_blocks() is called once per child decision and
    returns (n_envs, action_block, 6) absolute parent targets in radians.

    mode='scripted' runs the privileged keyframe sweep instead (no GPU), same queues."""

    def __init__(self, n_envs: int, mode: str = "smolvla", device: str = "cuda",
                 refill_below: int = 5, rate: float = 0.0, log=print,
                 model_id: str = "lerobot/smolvla_base"):
        from collections import deque
        self.n, self.mode, self.device = n_envs, mode, device
        self.refill_below = refill_below
        self.rate = float(rate)                          # rad/substep target slew limit; 0 = off
        self._last_q = [None] * n_envs                   # rate-limit state (absolute joint targets)
        self.gen = InstructionGen()
        self.queues = [deque() for _ in range(n_envs)]
        self.instr = [""] * n_envs
        self._scripted_t = np.zeros(n_envs, np.int64)
        self._scripted_aim = [None] * n_envs             # (pan, lift, elbow, dir) per env
        self.parent_contact_rate = 0.0                   # for the trainer's metrics window
        if mode == "smolvla":
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
            self.policy = SmolVLAPolicy.from_pretrained(model_id).to(device).eval()
            self.cam_keys = [k for k in self.policy.config.input_features if "image" in k]
            self._tok = self.policy.model.vlm_with_expert.processor.tokenizer
            self._max_len = int(getattr(self.policy.config, "tokenizer_max_length", 48))
            log(f"[parent] SmolVLA fleet driver up ({n_envs} envs, radian-native, {model_id})", flush=True)
        else:
            self.policy = None
            log(f"[parent] {mode} fleet driver up ({n_envs} envs)", flush=True)

    def _instructions_from(self, ctxs, ids):
        for i in ids:
            inview = ctxs[i]["inview"]
            # target = most child-centred block the PARENT can actually reach (its keyframe
            # calibration covers r ~ 0.17-0.40 from the parent base at (0.62, 0)); if the
            # child watches none of those, fall back to ANY reachable block -- motion the
            # child may catch beats guaranteed no motion.
            base = ctxs[i].get("parent_base_xy", np.array([0.55, 0.0]))
            def reach_r(b):
                return float(np.linalg.norm(ctxs[i]["block_xy"][b] - base))
            reachable = [b for b, _ in inview if 0.14 <= reach_r(b) <= 0.41]
            if not reachable:
                allr = [(reach_r(b), b) for b in range(len(ctxs[i]["block_xy"]))
                        if 0.16 <= reach_r(b) <= 0.28]
                reachable = [min(allr)[1]] if allr else []
            if reachable:
                bidx = reachable[0]
                block = {"color": color_word(ctxs[i]["block_rgba"][bidx]), "idx": bidx}
                self.instr[i] = self.gen.instruction_for(block)
                self._scripted_aim[i] = (bidx, -1.0 if "left" in self.instr[i] else 1.0)
            else:
                self.instr[i] = "move a cube on the table"
                self._scripted_aim[i] = None

    @torch.no_grad()
    def _refill_smolvla(self, ctxs, ids):
        imgs1 = torch.stack([torch.as_tensor(ctxs[i]["parent_view"]).permute(2, 0, 1)
                             for i in ids]).to(self.device).float() / 255.0
        imgs2 = torch.stack([torch.as_tensor(ctxs[i]["overhead"]).permute(2, 0, 1)
                             for i in ids]).to(self.device).float() / 255.0
        state = torch.stack([torch.as_tensor(ctxs[i]["parent_qpos"]) for i in ids]).to(self.device)
        enc = self._tok([self.instr[i] for i in ids], max_length=self._max_len,
                        truncation=True, padding="max_length", return_tensors="pt")
        batch = {"observation.state": state,
                 "task": [self.instr[i] for i in ids],
                 "observation.language.tokens": enc["input_ids"].to(self.device),
                 "observation.language.attention_mask": enc["attention_mask"].to(self.device, dtype=torch.bool)}
        zeros = torch.zeros_like(imgs1)
        for k, key in enumerate(self.cam_keys):
            batch[key] = (imgs1, imgs2, zeros)[min(k, 2)]
        chunks = self.policy.predict_action_chunk(batch).float().cpu().numpy()  # (B, 50, 6) rad
        for j, i in enumerate(ids):
            self.queues[i].extend(chunks[j])

    # RISE-DROP-SWEEP cycle: the exact phase structure the contact calibration VERIFIED
    # (2026-07-10: reset -> settle at depth -> short sweep = 45-172 contacts at every
    # radius 0.14-0.41). Continuous plowing WEDGES (elbow pinned against the table edge
    # mid-transit -- two-arm workspace is a collision minefield); the periodic RISE phase
    # un-wedges and each drop re-centres on the watched block's bearing.
    _POSE = (-0.7, 1.4, 0.6)                     # lift, elbow, wrist -- verified contact pose

    def _slew(self, i, steps, ctxs):
        """Rate-limit a keyframe chunk (rad/substep, --parent-rate): each emitted target
        moves at most self.rate per joint from the previous one, so the guardian glides
        instead of jumping between keyframes. No-op when rate == 0."""
        if self.rate <= 0:
            return steps
        if self._last_q[i] is None:
            self._last_q[i] = np.asarray(ctxs[i]["parent_qpos"], np.float64).copy()
        out = []
        q = self._last_q[i]
        for kf in steps:
            q = q + np.clip(np.asarray(kf) - q, -self.rate, self.rate)
            out.append(q.copy())
        self._last_q[i] = q
        return out

    def _refill_scripted(self, ctxs, ids):
        T_RISE, T_DROP, T_SWEEP = 25, 30, 65     # 120 sub-steps (4 s) per full cycle
        for i in ids:
            aim = self._scripted_aim[i]
            base = ctxs[i].get("parent_base_xy", np.array([0.55, 0.0]))
            if aim is not None:
                rel = ctxs[i]["block_xy"][aim[0]] - base
                center = float(np.clip(np.arctan2(-rel[1], -rel[0]), -0.5, 0.5))
            else:
                center = 0.0
            lift, elbow, wrist = self._POSE
            steps = []
            for t in range(50):
                ph = (self._scripted_t[i] + t) % (T_RISE + T_DROP + T_SWEEP)
                if ph < T_RISE:                                   # HOME pose = the calibration's reset
                    q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.3]            # state: definitionally un-wedged
                elif ph < T_RISE + T_DROP:                        # drop to depth at the start bearing
                    q = [center - 0.45, lift, elbow, wrist, 0.0, 0.3]
                else:                                             # the verified short sweep
                    s = (ph - T_RISE - T_DROP) / T_SWEEP
                    q = [center - 0.45 + 0.9 * s, lift, elbow, wrist, 0.0, 0.3]
                steps.append(np.asarray(q))
            self._scripted_t[i] += 50
            self.queues[i].extend(self._slew(i, steps, ctxs))

    # HERD mode (2026-07-11, emergence program): instead of re-novelizing blocks in
    # place, the parent SHEPHERDS out-of-reach blocks toward the child's reachable
    # annulus (child fingertip band r~0.21-0.33 at table level, measured in
    # smoke_child_sweep). Environment design only -- the child is never scripted.
    # Mechanism: the VERIFIED fleet sweep cycle, byte-identical keyframes, with only
    # (a) target selection (out-of-child-reach blocks instead of gaze) and (b) sweep
    # DIRECTION chosen so the arc carries the block toward the inter-robot axis --
    # blocks at parent-r 0.29-0.41 land on-axis at child-r ~0.21-0.33, inside the
    # annulus. Radial deep pushes were tried and rejected: extension into a block
    # pins the arm (contact wedge, unrecoverable within a cycle). Near-axis blocks
    # (|parent bearing| < 0.15) are excluded -- a through-sweep would carry them
    # OFF-axis. Envs with no herd candidate fall back to the standard gaze sweep.
    CHILD_R_MAX = 0.33

    def _instructions_herd(self, ctxs, ids):
        self._herd_dir = getattr(self, "_herd_dir", [0.0] * self.n)
        fallback = []
        for i in ids:
            base = ctxs[i].get("parent_base_xy", np.array([0.55, 0.0]))
            bxy = ctxs[i]["block_xy"]
            cand = []
            for b in range(len(bxy)):
                r_child = float(np.linalg.norm(bxy[b]))
                rel = bxy[b] - base
                r_par = float(np.linalg.norm(rel))
                bearing = float(np.arctan2(-rel[1], -rel[0]))   # parent faces -x
                if r_child > self.CHILD_R_MAX and 0.14 <= r_par <= 0.41 and abs(bearing) >= 0.15:
                    lands_in = 0.29 <= r_par <= 0.41            # arc endpoint ~ child annulus
                    cand.append((not lands_in, r_child, b, bearing))
            if cand:
                _, _, bidx, bearing = min(cand)
                self.instr[i] = f"push the {color_word(ctxs[i]['block_rgba'][bidx])} cube toward the other robot"
                self._scripted_aim[i] = (bidx, +1.0)
                self._herd_dir[i] = -np.sign(bearing)           # sweep toward the axis
            else:
                self._herd_dir[i] = 0.0
                fallback.append(i)
        if fallback:
            self._instructions_from(ctxs, fallback)             # nothing to herd: re-novelize

    def _refill_herd(self, ctxs, ids):
        """The DEMO driver's radius-adaptive hover-descend-sweep-lift cycle (verified
        contacts at every r 0.14-0.41), with the sweep direction from _instructions_herd
        so the arc carries the block toward the axis. The fleet's fixed-depth pose was
        tried first and misses targets off its band (hit-or-miss across layouts)."""
        T = 90
        SW = np.deg2rad(22.0)
        sweep_ids = [i for i in ids if getattr(self, "_herd_dir", [0.0] * self.n)[i] == 0.0]
        if sweep_ids:
            self._refill_scripted(ctxs, sweep_ids)
        for i in ids:
            if i in sweep_ids:
                continue
            aim = self._scripted_aim[i]
            base = ctxs[i].get("parent_base_xy", np.array([0.55, 0.0]))
            rel = ctxs[i]["block_xy"][aim[0]] - base
            r = float(np.linalg.norm(rel))
            center = float(np.clip(np.arctan2(-rel[1], -rel[0]), -0.6, 0.6))
            f = np.clip((r - 0.16) / 0.14, 0.0, 1.4)
            lift, elbow, wrist = -0.75 - 0.30 * f, 1.15 + 0.30 * f, 0.5
            d = self._herd_dir[i]
            steps = []
            for t in range(50):
                ph = ((self._scripted_t[i] + t) % T) / T
                if ph < 0.25:                            # hover above, panned to arc start
                    q = [center - d * SW, lift + 0.35, elbow - 0.2, wrist, 0.0, 0.5]
                elif ph < 0.45:                          # descend to contact depth
                    q = [center - d * SW, lift, elbow, wrist, 0.0, 0.3]
                elif ph < 0.75:                          # sweep THROUGH the block toward the axis
                    s = (ph - 0.45) / 0.30
                    q = [center + d * SW * (2 * s - 1), lift, elbow, wrist, 0.0, 0.3]
                else:                                    # lift + return
                    q = [center + d * SW, lift + 0.35, elbow - 0.25, wrist, 0.0, 0.5]
                steps.append(np.asarray(q))
            self._scripted_t[i] += 50
            self.queues[i].extend(self._slew(i, steps, ctxs))

    def next_blocks(self, env, action_block: int) -> np.ndarray:
        low = [i for i in range(self.n) if len(self.queues[i]) < action_block]
        if low:
            ctxs = env.parent_context()                  # one worker round-trip for the fleet
            if self.mode == "herd":
                self._instructions_herd(ctxs, low)
                self._refill_herd(ctxs, low)
            else:
                self._instructions_from(ctxs, low)
                if self.mode == "smolvla":
                    self._refill_smolvla(ctxs, low)
                else:
                    self._refill_scripted(ctxs, low)
        out = np.zeros((self.n, action_block, 6))
        for i in range(self.n):
            for k in range(action_block):
                out[i, k] = self.queues[i].popleft() if self.queues[i] else 0.0
        return out


# ---------------------------------------------------------------- orchestrator
def run_parent_child_demo(env, driver, episodes=4, decisions=120, retarget_every=40,
                          child_policy=None, log=print):
    """Single-env demo loop: the parent VLA manipulates whichever block the child watches.
    child_policy(obs)->action or None (child holds still). Returns per-episode metrics."""
    gen = InstructionGen()
    results = []
    for ep in range(episodes):
        obs = env.reset()
        target, instr = None, None
        ep_stats = {"parent_contacts": 0, "instructed_disp": 0.0, "inview_disp": 0.0,
                    "instructions": []}
        tracked0, tracked_idx = None, None
        for t in range(decisions):
            if t % retarget_every == 0 or target is None:
                inview = blocks_in_child_view(env)
                if tracked_idx is not None:            # bank the previous target's motion
                    ep_stats["instructed_disp"] += float(np.linalg.norm(
                        env.data.xpos[env._object_body_ids[tracked_idx]][:2] - tracked0))
                target = inview[0] if inview else None
                instr = gen.instruction_for(target) if target else "move a cube on the table"
                driver.reset(instr)
                if hasattr(driver, "track"):
                    driver.track(target["idx"] if target else None)
                ep_stats["instructions"].append(instr)
                tracked_idx = target["idx"] if target else None
                tracked0 = env.data.xpos[env._object_body_ids[tracked_idx]][:2].copy() \
                    if target else None
                inview0 = {b["idx"]: b["xpos"][:2].copy() for b in inview}
            # one VLA action per SUB-step (30 Hz, dataset-native cadence); render only
            # when the driver is about to refill its 50-action chunk
            for _s in range(5):
                cams = ([env.render_parent_view(), env.render_overhead(), None]
                        if driver.needs_obs() else None)
                pt = driver.act(cams, env.parent_qpos())
                env.set_parent_target(pt)
                a_child = child_policy(obs) if child_policy else np.zeros(env.n_dof, np.float32)
                obs, info = env.step(a_child, render=(_s == 4))
                ep_stats["parent_contacts"] += int(info.get("parent_object_contacts", 0))
        if tracked_idx is not None and tracked0 is not None:
            ep_stats["instructed_disp"] += float(np.linalg.norm(
                env.data.xpos[env._object_body_ids[tracked_idx]][:2] - tracked0))
        ep_stats["inview_disp"] = float(sum(
            np.linalg.norm(env.data.xpos[env._object_body_ids[i]][:2] - p0)
            for i, p0 in inview0.items())) if inview0 else 0.0
        results.append(ep_stats)
        log(f"[ep {ep}] contacts {ep_stats['parent_contacts']} | instructed disp "
            f"{ep_stats['instructed_disp']*100:.1f} cm | in-view disp {ep_stats['inview_disp']*100:.1f} cm "
            f"| {ep_stats['instructions'][:2]}")
    return results
