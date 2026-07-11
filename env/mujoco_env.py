"""MuJoCo SO-ARM101 environment: wrist-cam pixels + proprioception, an object
soup of cubes/pyramids to poke at, and the README safety reward.

Observation per step:
    image  : (H, W, 3) uint8   -- wrist_cam render (H=W=wrist_resolution, 224 for the ViT)
    proprio: (3*n_dof,) float32 -- [qpos, qvel, applied_torque]  (the u here is u^app_{t-1}
             when consumed by the encoder before the next action)

Action: (n_dof,) float32 in [-1, 1], turned into a delta position target by
`SOArmAdapter`; the XML position actuator then realises the README PD torque law.

step() also returns the README safety reward r_safe for this transition. No episode
termination; reset()/respawn keep an interesting object layout in frame.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import mujoco

from .soarm_adapter import SOArmAdapter
from .safety_reward import safety_reward_np

DEFAULT_SCENE = Path(__file__).resolve().parent / "SO101" / "scene.xml"

# Standard colour palette used for object randomisation.
STANDARD_COLORS = np.array([
    [0.85, 0.10, 0.10], [0.10, 0.70, 0.10], [0.10, 0.20, 0.85],
    [0.95, 0.85, 0.10], [0.85, 0.10, 0.85], [0.10, 0.85, 0.85],
    [0.95, 0.55, 0.10], [0.55, 0.20, 0.80], [0.95, 0.95, 0.95],
    [0.10, 0.10, 0.10],
], dtype=np.float32)

TABLE_TOP_Z = 0.0          # scene.xml: table top at z=0
ARM_BASE_RADIUS = 0.07     # reject spawns inside the base footprint
OBJECT_KEEPOUT = 0.025


def _random_quat(rng: np.random.Generator) -> np.ndarray:
    u1, u2, u3 = rng.random(3)
    return np.array([
        np.sqrt(u1) * np.cos(2 * np.pi * u3),
        np.sqrt(1.0 - u1) * np.sin(2 * np.pi * u2),
        np.sqrt(1.0 - u1) * np.cos(2 * np.pi * u2),
        np.sqrt(u1) * np.sin(2 * np.pi * u3),
    ], dtype=np.float64)


def _yaw_only_quat(rng: np.random.Generator) -> np.ndarray:
    yaw = rng.uniform(-np.pi, np.pi)
    return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)], dtype=np.float64)


def _name_lookup(model: mujoco.MjModel, obj: int, name: str) -> int:
    idx = mujoco.mj_name2id(model, obj, name)
    if idx < 0:
        raise KeyError(f"missing {mujoco.mjtObj(obj).name} '{name}'")
    return idx


class MujocoSO101Env:
    def __init__(
        self,
        scene_path: str | Path = DEFAULT_SCENE,
        wrist_resolution: int = 224,             # 224 for the ViT-tiny encoder
        overhead_resolution: int = 256,
        encode_cam: str = "wrist",               # which camera fills obs["image"] (the WM/encoder input);
                                                 # "overhead" = fixed third-person view (LeWM-style smoother
                                                 # latent for goal-reaching). Default "wrist" = byte-identical.
        frame_skip: int = 6,                     # 30 Hz at timestep 0.005
        action_max: float = 0.3,                 # README dq^max (delta scale per unit action)
        safety_delta: float = 9.0,              # README delta (real-arm calibrated 2026-06-12)
        n_objects: int = 10,
        n_cubes: int = 6,
        cube_size_range: tuple[float, float] = (0.012, 0.020),
        spawn_x_range: tuple[float, float] = (0.14, 0.42),
        spawn_y_range: tuple[float, float] = (-0.22, 0.22),
        side_scatter_frac: float = 0.20,
        side_scatter_x_range: tuple[float, float] = (-0.05, 0.40),
        side_scatter_y_range: tuple[float, float] = (-0.40, 0.40),
        substep_interp: bool = True,
        spawn_max_retries: int = 30,
        respawn_z_padding: float = 0.001,
        table_drop_threshold: float = -0.10,
        seed: int = 0,
        fixed_objects: bool = False,             # place objects deterministically (same layout EVERY env & reset)
        parent_arm: bool = False,                # attach a second (parent) SO-101 across the table -- a VLA-driven
                                                 # "adult" that moves blocks in the child's view. Child action/obs/
                                                 # safety/contact semantics are UNCHANGED (child ids resolved
                                                 # explicitly; parent geoms excluded from the child's contact sets).
        parent_pos: tuple[float, float, float] = (0.55, 0.0, 0.02),   # 2cm float: at z=0 the base mesh
                                                 # interpenetrates the tabletop and the WHOLE arm jams
                                                 # (saturated torque, zero motion — 2026-07-10 post-mortem)
        parent_yaw: float = 3.141592653589793,   # face the child across the block spawn zone
    ):
        self.parent_arm = bool(parent_arm)
        self.parent_base_xy = np.asarray(parent_pos[:2], np.float64)
        if parent_arm:
            self.model = self._build_two_arm_model(scene_path, parent_pos, parent_yaw)
        else:
            self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.dt_safe = frame_skip * float(self.model.opt.timestep)   # accel finite-diff window
        self.safety_delta = float(safety_delta)
        self.wrist_resolution = wrist_resolution
        self.overhead_resolution = overhead_resolution
        self.n_objects = n_objects
        self.n_cubes = n_cubes
        self.cube_size_range = cube_size_range
        self.spawn_x_range = spawn_x_range
        self.spawn_y_range = spawn_y_range
        self.side_scatter_frac = float(side_scatter_frac)
        self.side_scatter_x_range = side_scatter_x_range
        self.side_scatter_y_range = side_scatter_y_range
        self.substep_interp = bool(substep_interp)
        self.spawn_max_retries = int(spawn_max_retries)
        self.respawn_z_padding = float(respawn_z_padding)
        self.table_drop_threshold = table_drop_threshold

        # CHILD actuator/joint ids resolved explicitly: with --parent-arm the model holds 12
        # actuators (child 0-5 unchanged + parent_ 6-11); everything child-facing below slices
        # by these ids so the child's action/proprio/safety surface is byte-identical.
        _act_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                      for i in range(int(self.model.nu))]
        self._parent_act_ids = [i for i, n in enumerate(_act_names) if n and n.startswith("parent_")]
        self._child_act_ids = [i for i, n in enumerate(_act_names) if i not in self._parent_act_ids]
        self.n_dof = len(self._child_act_ids)
        self.tau_max = self.model.actuator_forcerange[self._child_act_ids, 1].copy().astype(np.float32)
        self.ctrl_low = self.model.actuator_ctrlrange[self._child_act_ids, 0].copy().astype(np.float32)
        self.ctrl_high = self.model.actuator_ctrlrange[self._child_act_ids, 1].copy().astype(np.float32)
        self.adapter = SOArmAdapter(self.ctrl_low, self.ctrl_high,
                                    action_max=action_max)
        if self.parent_arm:
            self._parent_ctrl_low = self.model.actuator_ctrlrange[self._parent_act_ids, 0].copy()
            self._parent_ctrl_high = self.model.actuator_ctrlrange[self._parent_act_ids, 1].copy()
            _pj = [f"parent_{n}" for n in
                   ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")]
            self._parent_qpos_adr = np.array(
                [int(self.model.jnt_qposadr[_name_lookup(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)])
                 for j in _pj])
            self._parent_view_cam_id = _name_lookup(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "parent_view")

        self._wrist_cam_id = _name_lookup(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
        self._overhead_cam_id = _name_lookup(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead")
        if encode_cam not in ("wrist", "overhead"):
            raise ValueError(f"encode_cam must be 'wrist' or 'overhead', got {encode_cam!r}")
        # the camera the WM/encoder sees (rendered at wrist_resolution into obs["image"]); "overhead"
        # swaps the egocentric wrist view for the fixed worldbody cam without any other plumbing change.
        self._encode_cam_id = self._overhead_cam_id if encode_cam == "overhead" else self._wrist_cam_id
        self.encode_cam = encode_cam
        # end-effector (gripper) body: world xyz -> the honest "is the arm roaming in space"
        # signal. Distal-wrist jitter pans the wrist cam without translating this; pose_step
        # (joint-space) can't tell those apart, the gripper world position can.
        self._ee_body_id = _name_lookup(self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
        self._object_body_ids, self._object_geom_ids = [], []
        self._object_qpos_addrs, self._object_qvel_addrs = [], []
        for i in range(n_objects):
            body = _name_lookup(self.model, mujoco.mjtObj.mjOBJ_BODY, f"object_{i}")
            geom = _name_lookup(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"object_{i}")
            joint = _name_lookup(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"object_{i}_free")
            self._object_body_ids.append(body)
            self._object_geom_ids.append(geom)
            self._object_qpos_addrs.append(int(self.model.jnt_qposadr[joint]))
            self._object_qvel_addrs.append(int(self.model.jnt_dofadr[joint]))
        self._object_geom_id_set = set(self._object_geom_ids)
        # Classify every geom: objects (named object_*), the table/floor "ground"
        # bucket (named table*/floor), and everything else = arm links. The arm's
        # mesh geoms are UNNAMED, so we must NOT skip name=None here -- doing so
        # leaves the arm set empty and no contact ever registers.
        self._arm_geom_ids: set[int] = set()
        self._table_geom_ids: set[int] = set()
        self._parent_geom_ids: set[int] = set()   # parent-arm links: EXCLUDED from the child's
        for gid in range(self.model.ngeom):       # contact buckets (its touches are not ours)
            if gid in self._object_geom_id_set:
                continue
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid)
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                          int(self.model.geom_bodyid[gid]))
            if self.parent_arm and body_name is not None and body_name.startswith("parent_"):
                self._parent_geom_ids.add(gid)
            elif name is not None and name.startswith(("table", "floor")):
                self._table_geom_ids.add(gid)
            else:
                self._arm_geom_ids.add(gid)

        self._wrist_renderer = mujoco.Renderer(self.model, height=wrist_resolution, width=wrist_resolution)
        self._overhead_renderer = mujoco.Renderer(self.model, height=overhead_resolution, width=overhead_resolution)
        self.rng = np.random.default_rng(seed)
        # Fixed-object mode: draw the object layout from a CONSTANT seed so every env (regardless of
        # its own `seed`) and every reset gets the IDENTICAL scene -- collapses visual variance to just
        # the arm, so the encoder/WM has a far easier (LeWM-cube-like) target. See randomise_all_objects.
        self.fixed_objects = bool(fixed_objects)
        self._fixed_obj_seed = 12345
        self._prev_ctrl = np.zeros(self.n_dof, dtype=np.float64)
        self._prev_qvel = np.zeros(self.n_dof, dtype=np.float32)
        self._prev_obj_xpos = np.zeros((n_objects, 3), dtype=np.float64)

    # --- Object randomisation -------------------------------------------------

    @staticmethod
    def _build_two_arm_model(scene_path, parent_pos, parent_yaw):
        """Compile the scene with a second (parent) SO-101 attached via MjSpec: bodies/
        joints/actuators arrive prefixed `parent_` and APPENDED, so the child keeps qpos
        0-5 / actuators 0-5 / block qpos from 6 -- verified layout-identical. Also adds a
        fixed `parent_view` camera framing the shared block zone for the parent's VLA."""
        import math
        spec = mujoco.MjSpec.from_file(str(scene_path))
        arm = mujoco.MjSpec.from_file(str(Path(scene_path).parent / "so101_new_calib.xml"))
        frame = spec.worldbody.add_frame(
            pos=list(parent_pos),
            quat=[math.cos(parent_yaw / 2), 0.0, 0.0, math.sin(parent_yaw / 2)])
        spec.attach(arm, prefix="parent_", frame=frame)
        cam_pos = np.array([parent_pos[0] - 0.02, -0.44, 0.42])
        target = np.array([0.31, 0.0, 0.02])                  # block spawn zone centre
        z = cam_pos - target; z = z / np.linalg.norm(z)       # camera looks along -z
        x = np.cross(np.array([0.0, 0.0, 1.0]), z); x = x / np.linalg.norm(x)
        y = np.cross(z, x)
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, np.stack([x, y, z], axis=1).flatten())
        spec.worldbody.add_camera(name="parent_view", pos=cam_pos.tolist(), quat=quat.tolist())
        return spec.compile()

    # ---- parent-arm API (the VLA-driven "adult"; no-ops unless parent_arm=True) ----
    def parent_qpos(self) -> np.ndarray:
        return self.data.qpos[self._parent_qpos_adr].copy().astype(np.float32)

    def set_parent_target(self, q_rad: np.ndarray) -> None:
        """Absolute joint-position targets (radians) for the parent's position servos --
        the VLA's native output space (converted from its dataset units by the caller)."""
        self.data.ctrl[self._parent_act_ids] = np.clip(
            q_rad, self._parent_ctrl_low, self._parent_ctrl_high)

    # TUCKED rest pose (2026-07-11, user request): all-zeros is an arm extended
    # horizontally over the table centre -- at reset it physically PRESSES on the
    # child's reset pose (27 contact points measured). The tuck folds the arm over
    # its own base (max extent 0.18 m, zero contacts; empirically scanned, pose U2).
    PARENT_TUCK = np.array([0.0, -1.5, 1.5, 0.0, 0.0, 0.3])

    def parent_home(self) -> None:
        self.data.qpos[self._parent_qpos_adr] = self.PARENT_TUCK
        self.data.ctrl[self._parent_act_ids] = self.PARENT_TUCK

    def render_parent_view(self) -> np.ndarray:
        """256x256 RGB from the fixed parent_view camera (the parent VLA's scene camera)."""
        self._overhead_renderer.update_scene(self.data, camera=self._parent_view_cam_id)
        return self._overhead_renderer.render()

    def parent_context(self) -> dict:
        """Everything the MAIN-process parent driver needs at chunk-refill cadence, in one
        worker round-trip: the parent's cameras + proprio, plus gaze geometry (which blocks
        the CHILD's wrist cam currently sees) and raw block state for scripted drivers."""
        cid = self._wrist_cam_id
        cam_pos = self.data.cam_xpos[cid].copy()
        R = self.data.cam_xmat[cid].reshape(3, 3)
        half = float(np.tan(np.deg2rad(float(self.model.cam_fovy[cid])) / 2))
        inview = []
        for i, bid in enumerate(self._object_body_ids):
            p = R.T @ (self.data.xpos[bid] - cam_pos)
            if p[2] > -0.03:
                continue
            u, v = p[0] / -p[2], p[1] / -p[2]
            if abs(u) < half * 0.92 and abs(v) < half * 0.92:
                inview.append((i, float(max(abs(u), abs(v)) / half)))
        return {
            "parent_view": self.render_parent_view(),
            "overhead": self.render_overhead(),
            "parent_qpos": self.parent_qpos(),
            "inview": sorted(inview, key=lambda t: t[1]),
            "block_xy": np.stack([self.data.xpos[b][:2] for b in self._object_body_ids]),
            "block_rgba": self.model.geom_rgba[self._object_geom_ids].copy(),
            "parent_base_xy": self.parent_base_xy.copy(),
        }

    def _sample_xy(self, in_frustum: bool) -> tuple[float, float]:
        if in_frustum:
            return (float(self.rng.uniform(*self.spawn_x_range)),
                    float(self.rng.uniform(*self.spawn_y_range)))
        return (float(self.rng.uniform(*self.side_scatter_x_range)),
                float(self.rng.uniform(*self.side_scatter_y_range)))

    def _object_resting_z(self, idx: int) -> float:
        if idx < self.n_cubes:
            half = float(self.model.geom_size[self._object_geom_ids[idx], 0])
            return TABLE_TOP_Z + half + self.respawn_z_padding
        return TABLE_TOP_Z + self.respawn_z_padding

    def _object_xy(self, idx: int) -> np.ndarray:
        addr = self._object_qpos_addrs[idx]
        return self.data.qpos[addr:addr + 2].copy()

    def _spawn_collides(self, idx: int, placed_indices: list[int]) -> bool:
        mujoco.mj_forward(self.model, self.data)
        own_geom = self._object_geom_ids[idx]
        for c in range(self.data.ncon):
            con = self.data.contact[c]
            g1, g2 = int(con.geom1), int(con.geom2)
            if (g1 == own_geom and g2 in self._arm_geom_ids) or (
                g2 == own_geom and g1 in self._arm_geom_ids):
                return True
        own_xy = self._object_xy(idx)
        own_keep = OBJECT_KEEPOUT
        if idx < self.n_cubes:
            own_keep = max(own_keep, float(self.model.geom_size[own_geom, 0]) * 1.6)
        for j in placed_indices:
            if j == idx:
                continue
            other_xy = self._object_xy(j)
            other_keep = OBJECT_KEEPOUT
            if j < self.n_cubes:
                other_keep = max(other_keep, float(self.model.geom_size[self._object_geom_ids[j], 0]) * 1.6)
            if np.linalg.norm(own_xy - other_xy) < (own_keep + other_keep) * 0.6:
                return True
        if np.linalg.norm(own_xy) < ARM_BASE_RADIUS:
            return True
        return False

    def _place_object_safely(self, idx: int, placed_indices: list[int]) -> None:
        addr = self._object_qpos_addrs[idx]
        vel_addr = self._object_qvel_addrs[idx]
        is_pyramid = idx >= self.n_cubes
        if not is_pyramid:
            s = self.rng.uniform(*self.cube_size_range)
            self.model.geom_size[self._object_geom_ids[idx]] = [s, s, s]
        self.model.geom_rgba[self._object_geom_ids[idx], :3] = STANDARD_COLORS[
            self.rng.integers(0, len(STANDARD_COLORS))]
        self.model.geom_rgba[self._object_geom_ids[idx], 3] = 1.0
        z = self._object_resting_z(idx)
        for _ in range(self.spawn_max_retries):
            in_frustum = self.rng.random() >= self.side_scatter_frac
            x, y = self._sample_xy(in_frustum)
            if x * x + y * y < ARM_BASE_RADIUS * ARM_BASE_RADIUS:
                continue
            self.data.qpos[addr:addr + 3] = [x, y, z]
            quat = _yaw_only_quat(self.rng) if is_pyramid else _random_quat(self.rng)
            self.data.qpos[addr + 3:addr + 7] = quat
            self.data.qvel[vel_addr:vel_addr + 6] = 0.0
            if not self._spawn_collides(idx, placed_indices):
                return
        self.data.qpos[addr:addr + 3] = [0.30, 0.0, z]
        self.data.qpos[addr + 3:addr + 7] = _yaw_only_quat(self.rng)
        self.data.qvel[vel_addr:vel_addr + 6] = 0.0

    def randomise_all_objects(self) -> None:
        # fixed_objects: place from a CONSTANT-seed generator so the layout (positions, sizes, colors,
        # orientations, and the deterministic collision-retry path) is byte-identical across every env
        # and every reset. Restore the per-env rng afterwards so arm noise etc. stays per-env.
        saved_rng = None
        if self.fixed_objects:
            saved_rng, self.rng = self.rng, np.random.default_rng(self._fixed_obj_seed)
        try:
            for i in range(self.n_objects):
                v = self._object_qvel_addrs[i]
                self.data.qvel[v:v + 6] = 0.0
            placed: list[int] = []
            for i in range(self.n_objects):
                self._place_object_safely(i, placed)
                placed.append(i)
        finally:
            if saved_rng is not None:
                self.rng = saved_rng

    # --- Lifecycle ------------------------------------------------------------

    def reset(self) -> dict[str, np.ndarray]:
        mujoco.mj_resetData(self.model, self.data)
        if self.fixed_objects:
            # Place objects against a FIXED (zero) arm pose so the layout can't vary with the per-env
            # arm jitter; re-add the small arm-start noise AFTER placement (blocks stay identical, arm
            # keeps per-env diversity).
            self.data.qpos[:self.n_dof] = 0.0
            self.randomise_all_objects()
            self.data.qpos[:self.n_dof] = self.rng.normal(0.0, 0.02, size=self.n_dof)
        else:
            self.data.qpos[:self.n_dof] = self.rng.normal(0.0, 0.02, size=self.n_dof)
            self.randomise_all_objects()
        self._prev_ctrl = self.data.qpos[:self.n_dof].copy().astype(np.float64)
        self.data.ctrl[:self.n_dof] = self._prev_ctrl
        if self.parent_arm:                       # parent wakes at home pose, holding position
            self.data.qpos[self._parent_qpos_adr] = 0.0
            self.data.ctrl[self._parent_act_ids] = 0.0
            self._parent_obj_contacts = 0
        mujoco.mj_forward(self.model, self.data)
        self._prev_qvel = self.data.qvel[:self.n_dof].copy().astype(np.float32)
        self._prev_obj_xpos = self._object_xpos()
        return self._get_obs()

    def _object_xpos(self) -> np.ndarray:
        return np.stack([self.data.xpos[b].copy() for b in self._object_body_ids])

    def _interaction_stats(self) -> tuple[int, int, float]:
        """What the arm is touching/moving this step -- the core "interacts with
        blocks" signal. Returns (#arm-object contacts, #arm-table/floor contacts,
        total object xy-displacement since last step). Each contact is classified
        once, object before table, so the buckets never double-count."""
        n_obj = n_table = 0
        self._parent_obj_contacts = 0            # parent-arm touches (diagnostic; NOT the child's)
        for c in range(self.data.ncon):
            con = self.data.contact[c]
            g1, g2 = int(con.geom1), int(con.geom2)
            if self.parent_arm:
                p1, p2 = g1 in self._parent_geom_ids, g2 in self._parent_geom_ids
                if (p1 and g2 in self._object_geom_id_set) or (p2 and g1 in self._object_geom_id_set):
                    self._parent_obj_contacts += 1
            a1, a2 = g1 in self._arm_geom_ids, g2 in self._arm_geom_ids
            if not (a1 or a2):
                continue
            if (a1 and g2 in self._object_geom_id_set) or (a2 and g1 in self._object_geom_id_set):
                n_obj += 1
            elif (a1 and g2 in self._table_geom_ids) or (a2 and g1 in self._table_geom_ids):
                n_table += 1
        xpos = self._object_xpos()
        motion = float(np.linalg.norm(xpos[:, :2] - self._prev_obj_xpos[:, :2], axis=-1).sum())
        self._prev_obj_xpos = xpos
        return n_obj, n_table, motion

    def step(self, action: np.ndarray, render: bool = True) -> tuple[dict[str, np.ndarray], dict]:
        """One env step: delta-target PD actuation, frame_skip substeps, safety reward.
        render=False skips the (OSMesa, CPU-bound) wrist render and returns image=None --
        used for the non-final substeps of an action_block, whose image is discarded
        anyway (only the block's final obs is kept). Cuts renders ~action_block-fold."""
        qpos_now = self.data.qpos[:self.n_dof].copy()
        qvel_before = self.data.qvel[:self.n_dof].copy().astype(np.float32)
        target = self.adapter.ctrl_target(action, qpos_now)
        prev = self._prev_ctrl.copy()
        if self.substep_interp:
            for k in range(self.frame_skip):
                alpha = (k + 1) / self.frame_skip
                self.data.ctrl[:self.n_dof] = (1.0 - alpha) * prev + alpha * target
                mujoco.mj_step(self.model, self.data)
        else:
            self.data.ctrl[:self.n_dof] = target
            for _ in range(self.frame_skip):
                mujoco.mj_step(self.model, self.data)
        self._prev_ctrl = target
        n_contact, n_table, obj_motion = self._interaction_stats()   # before respawn (teleports excluded)
        self._maybe_respawn_fallen()

        applied_torque = self.data.actuator_force[self._child_act_ids].copy().astype(np.float32)
        qvel = self.data.qvel[:self.n_dof].copy().astype(np.float32)
        r_safe = float(safety_reward_np(applied_torque, qvel, qvel_before, self.tau_max,
                                        dt_safe=self.dt_safe, delta=self.safety_delta))
        obs = self._get_obs(render=render)
        info = {
            "applied_torque": applied_torque,
            "qvel": qvel,
            "qvel_prev": qvel_before,
            "qpos": self.data.qpos[:self.n_dof].copy().astype(np.float32),
            "ee_pos": self.data.xpos[self._ee_body_id].copy().astype(np.float32),  # gripper world xyz (roam metric)
            "safety_reward": r_safe,
            "object_contacts": np.int64(n_contact),
            "table_contacts": np.int64(n_table),
            "object_motion": np.float32(obj_motion),
        }
        if self.parent_arm:
            info["parent_object_contacts"] = np.int64(self._parent_obj_contacts)
        self._prev_qvel = qvel
        return obs, info

    def _maybe_respawn_fallen(self) -> None:
        fell = [i for i in range(self.n_objects)
                if self.data.xpos[self._object_body_ids[i], 2] < self.table_drop_threshold]
        if not fell:
            return
        keep = [i for i in range(self.n_objects) if i not in fell]
        for i in fell:
            self._place_object_safely(i, keep)
            keep.append(i)
        mujoco.mj_forward(self.model, self.data)
        for i in fell:  # don't count the teleport as object motion next step
            self._prev_obj_xpos[i] = self.data.xpos[self._object_body_ids[i]]

    # --- Observation ----------------------------------------------------------

    def _get_obs(self, render: bool = True) -> dict[str, np.ndarray]:
        if render:
            # render the configured encode camera (wrist by default; overhead = fixed view) through the
            # wrist_resolution renderer, so obs["image"] is always (wrist_resolution, wrist_resolution, 3).
            self._wrist_renderer.update_scene(self.data, camera=self._encode_cam_id)
            wrist = self._wrist_renderer.render().copy()
        else:
            wrist = None                       # discarded substep: skip the costly render
        proprio = np.concatenate([
            self.data.qpos[:self.n_dof].astype(np.float32),
            self.data.qvel[:self.n_dof].astype(np.float32),
            self.data.actuator_force[self._child_act_ids].astype(np.float32),
        ])
        return {"image": wrist, "proprio": proprio}

    def render_overhead(self) -> np.ndarray:
        self._overhead_renderer.update_scene(self.data, camera=self._overhead_cam_id)
        return self._overhead_renderer.render().copy()

    def close(self) -> None:
        for r in ("_wrist_renderer", "_overhead_renderer"):
            if hasattr(self, r):
                getattr(self, r).close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
