"""Gymnasium environment for the three-finger MuJoCo grasp task."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import mujoco
import numpy as np

try:
    import mujoco.viewer as mujoco_viewer
except ImportError:  # pragma: no cover
    mujoco_viewer = None

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    import gym
    from gym import spaces


_XML_PATH = os.path.join(
    os.path.dirname(__file__), "..", "mujoco_model", "scene.xml"
)

OBJECT_CONFIGS = {
    "sphere": {"type": "sphere", "size": [0.025], "mass": 0.05, "rgba": "0.9 0.2 0.2 1"},
    "sphere_small": {"type": "sphere", "size": [0.020], "mass": 0.03, "rgba": "0.2 0.8 0.2 1"},
    "cylinder": {"type": "cylinder", "size": [0.01, 0.015], "mass": 0.05, "rgba": "0.2 0.5 0.9 1"},
    "box": {"type": "box", "size": [0.012, 0.012, 0.012], "mass": 0.05, "rgba": "0.9 0.7 0.1 1"},
    "box_large": {"type": "box", "size": [0.020, 0.020, 0.020], "mass": 0.06, "rgba": "0.9 0.7 0.1 1"},
}

_GEOM_TYPE_MAP = {
    "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
    "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
    "box": mujoco.mjtGeom.mjGEOM_BOX,
}

_WRIST_ACTUATORS = [
    "wrist_act_x",
    "wrist_act_y",
    "wrist_act_z",
    "wrist_act_rx",
    "wrist_act_ry",
    "wrist_act_rz",
]
_FINGER_ACTUATORS = [
    "thumb_act_q1",
    "thumb_act_q2",
    "thumb_act_q3",
    "thumb_act_q4",
    "index_act_q1",
    "index_act_q2",
    "index_act_q3",
    "middle_act_q1",
    "middle_act_q2",
    "middle_act_q3",
]
_ALL_ACTUATORS = _WRIST_ACTUATORS + _FINGER_ACTUATORS

_TIP_BODY_NAMES = ["thumb_tip", "index_tip", "middle_tip"]
_TIP_SENSOR_NAMES = ["thumb_touch", "index_touch", "middle_touch"]
_TIP_GEOM_NAMES = ["thumb_tip_geom", "index_tip_geom", "middle_tip_geom"]

_FINGER_GEOM_GROUPS = {
    "thumb": (
        "thumb_tip_geom",
        "thumb_pad_NT",
        "thumb_pad_MN",
        "thumb_pad_OM",
        "thumb_NT",
        "thumb_MN",
        "thumb_OM",
    ),
    "index": (
        "index_tip_geom",
        "index_pad_NT",
        "index_pad_MN",
        "index_pad_PM",
        "index_NT",
        "index_MN",
        "index_PM",
    ),
    "middle": (
        "middle_tip_geom",
        "middle_pad_NT",
        "middle_pad_MN",
        "middle_pad_PM",
        "middle_NT",
        "middle_MN",
        "middle_PM",
    ),
}
_FINGER_TIP_GEOMS = {
    "thumb": "thumb_tip_geom",
    "index": "index_tip_geom",
    "middle": "middle_tip_geom",
}
_FINGER_SUPPORT_GEOMS = {
    "thumb": (
        "thumb_tip_geom",
        "thumb_pad_NT",
        "thumb_pad_MN",
        "thumb_pad_OM",
        "thumb_NT",
        "thumb_MN",
    ),
    "index": (
        "index_tip_geom",
        "index_pad_NT",
        "index_pad_MN",
        "index_pad_PM",
        "index_NT",
        "index_MN",
        "index_PM",
    ),
    "middle": (
        "middle_tip_geom",
        "middle_pad_NT",
        "middle_pad_MN",
        "middle_pad_PM",
        "middle_NT",
        "middle_MN",
        "middle_PM",
    ),
}

_FIXED_PREGRASP = {
    "wrist_offset": np.array([0.0720, -0.0820, 0.0740], dtype=np.float64),
    "wrist_rot": np.array([0.0884, 0.4050, 0.2450], dtype=np.float64),
    "thumb": np.radians(np.array([-60.0, 0.0, -26.0, -8.0], dtype=np.float64)),
    "index": np.radians(np.array([-24.0, -3.0, -15.0], dtype=np.float64)),
    "middle": np.radians(np.array([-24.0, -3.5, -17.0], dtype=np.float64)),
}
_Q3Q4_POLY = np.array([-0.01462, 1.27107, 0.07658, 0.05314, 0.10674], dtype=np.float64)


class JackHandGraspEnv(gym.Env):
    """MuJoCo environment used by the three-finger grasp experiments."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 25}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        object_type: str = "sphere",
        cam_width: int = 320,
        cam_height: int = 240,
        n_substeps: int = 20,
        reward_type: str = "dense",
        include_camera_obs: bool = True,
        randomize_object_radius: bool = False,
        object_radius_range: Tuple[float, float] = (0.023, 0.028),
        randomize_object_friction: bool = False,
        object_friction_range: Tuple[float, float] = (0.7, 1.3),
        init_hand_pose: str = "fixed_pregrasp",
        spawn_xy_range: Tuple[float, float] = (0.008, 0.008),
        object_clearance: float = 0.003,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.object_type = object_type
        self.cam_width = cam_width
        self.cam_height = cam_height
        self.n_substeps = n_substeps
        self.reward_type = reward_type
        self.include_camera_obs = include_camera_obs
        self.randomize_object_radius = randomize_object_radius
        self.object_radius_range = object_radius_range
        self.randomize_object_friction = randomize_object_friction
        self.object_friction_range = object_friction_range
        self.init_hand_pose = init_hand_pose
        self.object_clearance = float(object_clearance)

        self.model = mujoco.MjModel.from_xml_path(_XML_PATH)
        self.data = mujoco.MjData(self.model)

        self._renderer = None
        self._viewer = None
        self._spawn_xy_range = np.array(spawn_xy_range, dtype=np.float64)

        self._cache_ids()
        self._build_spaces()

        self._initial_obj_xy = np.array([-0.02, 0.08], dtype=np.float64)
        self._step_count = 0
        self._max_steps = 250
        self._prev_obj_pos = np.zeros(3, dtype=np.float64)
        self._episode_obj_pos = np.zeros(3, dtype=np.float64)
        self._initial_obj_height = 0.0
        self._reset_action = np.zeros(len(_ALL_ACTUATORS), dtype=np.float32)
        self._last_metrics: Dict[str, Any] = {}
        self._stable_enclosure_steps = 0
        self._max_stable_enclosure_steps = 0
        self._max_support_contacts = 0
        self._max_height_gain = 0.0
        self._max_object_yaw_rate = 0.0
        self._ever_thumb_front_support = False
        self._ever_tip_contact = False
        self._ever_link_contact = False
        self._prev_tip_mean = 0.0
        self._prev_palm_err = 0.0
        self._prev_contact_score = 0.0
        self._prev_support_contacts = 0
        self._prev_enclosure_steps = 0
        self._prev_opposition_active = False
        self._prev_obj_yaw = 0.0

    def _cache_ids(self):
        m = self.model

        self._act_ids = []
        self._act_ranges = np.zeros((len(_ALL_ACTUATORS), 2), dtype=np.float64)
        self._act_joint_names = []
        for i, name in enumerate(_ALL_ACTUATORS):
            aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            self._act_ids.append(aid)
            self._act_ranges[i] = m.actuator_ctrlrange[aid]
            jid = m.actuator_trnid[aid, 0]
            self._act_joint_names.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid))

        self._joint_qposadr = {}
        joint_names = [
            "wrist_x",
            "wrist_y",
            "wrist_z",
            "wrist_rx",
            "wrist_ry",
            "wrist_rz",
            "thumb_q1",
            "thumb_q2",
            "thumb_q3",
            "thumb_q4",
            "index_q1",
            "index_q2",
            "index_q3",
            "index_q4",
            "middle_q1",
            "middle_q2",
            "middle_q3",
            "middle_q4",
        ]
        for name in joint_names:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            self._joint_qposadr[name] = m.jnt_qposadr[jid]

        self._obj_geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "obj_geom")
        self._obj_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "target_object")
        self._palm_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "palm")
        self._cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "palm_cam")
        self._table_geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table_top")

        self._table_center = m.geom_pos[self._table_geom_id].copy()
        self._table_geom_type = int(m.geom_type[self._table_geom_id])
        self._table_radius = float(m.geom_size[self._table_geom_id, 0])
        self._table_size = m.geom_size[self._table_geom_id].copy()

        self._obj_jnt_adr = m.body_jntadr[self._obj_body_id]
        self._obj_qpos_adr = m.jnt_qposadr[self._obj_jnt_adr]
        self._obj_dof_adr = m.jnt_dofadr[self._obj_jnt_adr]
        self._default_obj_size = m.geom_size[self._obj_geom_id].copy()
        self._default_obj_friction = m.geom_friction[self._obj_geom_id].copy()
        self._default_obj_type = int(m.geom_type[self._obj_geom_id])
        self._default_obj_rgba = m.geom_rgba[self._obj_geom_id].copy()
        self._default_obj_mass = float(m.body_mass[self._obj_body_id])

        self._touch_sensor_ids = []
        for name in _TIP_SENSOR_NAMES:
            sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, name)
            self._touch_sensor_ids.append(sid)

        self._tip_body_ids = []
        for name in _TIP_BODY_NAMES:
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
            self._tip_body_ids.append(bid)

        self._tip_geom_ids = []
        for name in _TIP_GEOM_NAMES:
            gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
            self._tip_geom_ids.append(gid)

        self._finger_geom_ids = {}
        self._geom_to_finger = {}
        self._geom_to_name = {}
        self._support_geom_ids = {}
        for finger, geom_names in _FINGER_GEOM_GROUPS.items():
            ids = []
            support_ids = set()
            for geom_name in geom_names:
                gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
                ids.append(gid)
                self._geom_to_finger[gid] = finger
                self._geom_to_name[gid] = geom_name
                if geom_name in _FINGER_SUPPORT_GEOMS[finger]:
                    support_ids.add(gid)
            self._finger_geom_ids[finger] = tuple(ids)
            self._support_geom_ids[finger] = support_ids

    def _build_spaces(self):
        self.observation_space = spaces.Dict(
            {
                "palm_rgb": spaces.Box(
                    0, 255, (self.cam_height, self.cam_width, 3), dtype=np.uint8
                ),
                "palm_depth": spaces.Box(
                    0.0, 2.0, (self.cam_height, self.cam_width), dtype=np.float32
                ),
                "joint_pos": spaces.Box(-np.pi, np.pi, (16,), dtype=np.float32),
                "joint_vel": spaces.Box(-50.0, 50.0, (16,), dtype=np.float32),
                "touch_force": spaces.Box(-50.0, 50.0, (3, 3), dtype=np.float32),
                "obj_pos": spaces.Box(-1.0, 1.0, (3,), dtype=np.float32),
                "obj_quat": spaces.Box(-1.0, 1.0, (4,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Box(-1.0, 1.0, (16,), dtype=np.float32)

    def reset(self, *, seed=None, options=None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)
        self._apply_object_config()
        self.model.geom_friction[self._obj_geom_id] = self._default_obj_friction

        if self.object_type.startswith("sphere") and self.randomize_object_radius:
            radius = float(self.np_random.uniform(*self.object_radius_range))
            self.model.geom_size[self._obj_geom_id, 0] = radius

        if self.randomize_object_friction:
            slide = float(self.np_random.uniform(*self.object_friction_range))
            self.model.geom_friction[self._obj_geom_id, 0] = slide

        obj_pos = np.array(
            [
                self._initial_obj_xy[0],
                self._initial_obj_xy[1],
                self._object_rest_height(),
            ],
            dtype=np.float64,
        )
        if self.np_random is not None:
            obj_pos[:2] += self.np_random.uniform(-self._spawn_xy_range, self._spawn_xy_range)
        obj_pos[2] = self._object_rest_height()

        self._apply_hand_pose(self.init_hand_pose, obj_pos)

        adr = self._obj_qpos_adr
        self.data.qpos[adr : adr + 3] = obj_pos
        self.data.qpos[adr + 3 : adr + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[:] = 0.0

        self._sync_ctrl_to_qpos()
        mujoco.mj_forward(self.model, self.data)
        self._settle_reset_state()

        settled_obj_pos = self.data.qpos[adr : adr + 3].copy()
        self._step_count = 0
        self._prev_obj_pos = settled_obj_pos.copy()
        self._episode_obj_pos = settled_obj_pos.copy()
        self._initial_obj_height = float(settled_obj_pos[2])
        self._stable_enclosure_steps = 0
        self._max_stable_enclosure_steps = 0
        self._max_support_contacts = 0
        self._max_height_gain = 0.0
        self._max_object_yaw_rate = 0.0
        self._ever_thumb_front_support = False
        self._ever_tip_contact = False
        self._ever_link_contact = False

        obs = self._get_obs()
        metrics = self._compute_step_metrics(obs)
        self._update_episode_stats(metrics)
        self._last_metrics = metrics
        self._prev_tip_mean = float(metrics["mean_tip_to_obj"])
        self._prev_palm_err = float(metrics["palm_err"])
        self._prev_contact_score = float(
            metrics["n_tip_contacts"] + 1.5 * metrics["n_link_contacts"]
        )
        self._prev_support_contacts = int(metrics["n_support_contacts"])
        self._prev_enclosure_steps = int(metrics["stable_enclosure_steps"])
        self._prev_opposition_active = bool(metrics["thumb_support"] and metrics["front_support"])
        self._prev_obj_yaw = float(metrics["object_yaw"])
        info = self._get_info(obs, metrics)
        info["reset_action"] = self.get_reset_action()
        info["xml_path"] = _XML_PATH
        return obs, info

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        ctrl = np.zeros(self.model.nu, dtype=np.float64)
        for i, aid in enumerate(self._act_ids):
            lo, hi = self._act_ranges[i]
            ctrl[aid] = lo + (float(action[i]) + 1.0) * 0.5 * (hi - lo)

        self.data.ctrl[:] = ctrl
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        obs = self._get_obs()
        metrics = self._compute_step_metrics(obs)
        self._update_episode_stats(metrics)
        self._last_metrics = metrics

        reward = self._compute_reward(obs, metrics)
        terminated = self._check_terminated(metrics)
        truncated = self._step_count >= self._max_steps
        info = self._get_info(obs, metrics)

        self._prev_obj_pos = obs["obj_pos"].astype(np.float64)
        self._prev_tip_mean = float(metrics["mean_tip_to_obj"])
        self._prev_palm_err = float(metrics["palm_err"])
        self._prev_contact_score = float(
            metrics["n_tip_contacts"] + 1.5 * metrics["n_link_contacts"]
        )
        self._prev_support_contacts = int(metrics["n_support_contacts"])
        self._prev_enclosure_steps = int(metrics["stable_enclosure_steps"])
        self._prev_opposition_active = bool(metrics["thumb_support"] and metrics["front_support"])
        self._prev_obj_yaw = float(metrics["object_yaw"])
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_camera_rgb("palm_cam")
        if self.render_mode == "human":
            if self._viewer is None:
                if mujoco_viewer is None:
                    raise RuntimeError("mujoco.viewer is unavailable in this environment")
                self._viewer = mujoco_viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
        return None

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def _get_obs(self) -> Dict[str, np.ndarray]:
        if self.include_camera_obs:
            rgb, depth = self._render_palm_camera()
        else:
            rgb = np.zeros((self.cam_height, self.cam_width, 3), dtype=np.uint8)
            depth = np.zeros((self.cam_height, self.cam_width), dtype=np.float32)

        joint_pos = np.zeros(16, dtype=np.float32)
        joint_vel = np.zeros(16, dtype=np.float32)
        for i, aid in enumerate(self._act_ids):
            jid = self.model.actuator_trnid[aid, 0]
            qadr = self.model.jnt_qposadr[jid]
            vadr = self.model.jnt_dofadr[jid]
            joint_pos[i] = self.data.qpos[qadr]
            joint_vel[i] = self.data.qvel[vadr]

        touch_force = self._get_touch_forces()
        adr = self._obj_qpos_adr
        obj_pos = self.data.qpos[adr : adr + 3].copy().astype(np.float32)
        obj_quat = self.data.qpos[adr + 3 : adr + 7].copy().astype(np.float32)

        return {
            "palm_rgb": rgb,
            "palm_depth": depth,
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "touch_force": touch_force,
            "obj_pos": obj_pos,
            "obj_quat": obj_quat,
        }

    def _get_renderer(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model, height=self.cam_height, width=self.cam_width
            )
        return self._renderer

    def _render_palm_camera(self) -> Tuple[np.ndarray, np.ndarray]:
        renderer = self._get_renderer()

        renderer.update_scene(self.data, camera=self._cam_id)
        rgb = renderer.render().copy()

        renderer.enable_depth_rendering()
        renderer.update_scene(self.data, camera=self._cam_id)
        depth = renderer.render().copy().astype(np.float32)
        renderer.disable_depth_rendering()
        return rgb, depth

    def _render_camera_rgb(self, camera_name: str) -> np.ndarray:
        renderer = self._get_renderer()
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        renderer.update_scene(self.data, camera=cam_id)
        return renderer.render().copy()

    def _settle_reset_state(self):
        for _ in range(180):
            mujoco.mj_step(self.model, self.data)
            obj_pos = self.data.qpos[self._obj_qpos_adr : self._obj_qpos_adr + 3]
            linear_vel = self.data.qvel[self._obj_dof_adr : self._obj_dof_adr + 3]
            angular_vel = self.data.qvel[self._obj_dof_adr + 3 : self._obj_dof_adr + 6]
            gap = float(
                obj_pos[2] - self._current_object_vertical_extent() - self._table_surface_z()
            )
            if gap <= 5e-4 and np.linalg.norm(linear_vel) < 1e-3 and np.linalg.norm(angular_vel) < 1e-2:
                break

        current_gap = float(
            self.data.qpos[self._obj_qpos_adr + 2]
            - self._current_object_vertical_extent()
            - self._table_surface_z()
        )
        target_gap = self._reset_supported_gap()
        self.data.qpos[self._obj_qpos_adr + 2] += target_gap - current_gap
        self.data.qvel[self._obj_dof_adr : self._obj_dof_adr + 6] = 0.0
        self._sync_ctrl_to_qpos()
        mujoco.mj_forward(self.model, self.data)

    def _get_touch_forces(self) -> np.ndarray:
        forces = np.zeros((3, 3), dtype=np.float32)

        for ic in range(self.data.ncon):
            contact = self.data.contact[ic]
            g1, g2 = contact.geom1, contact.geom2

            for fi, tip_gid in enumerate(self._tip_geom_ids):
                if g1 != tip_gid and g2 != tip_gid:
                    continue
                if g1 != self._obj_geom_id and g2 != self._obj_geom_id:
                    continue

                c_force = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(self.model, self.data, ic, c_force)
                fn, ft1, ft2 = c_force[:3]
                frame = contact.frame.reshape(3, 3)
                tip_body_id = self._tip_body_ids[fi]
                tip_xmat = self.data.xmat[tip_body_id].reshape(3, 3)
                f_world = frame.T @ np.array([fn, ft1, ft2], dtype=np.float64)
                f_local = tip_xmat.T @ f_world
                forces[fi] = f_local.astype(np.float32)
                break

        return forces

    def get_touch_normal_forces(self) -> np.ndarray:
        forces = np.zeros(3, dtype=np.float32)
        for i, sid in enumerate(self._touch_sensor_ids):
            adr = self.model.sensor_adr[sid]
            forces[i] = self.data.sensordata[adr]
        return forces

    def _table_surface_z(self) -> float:
        if self._table_geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            half_height = float(self.model.geom_size[self._table_geom_id, 1])
        elif self._table_geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            half_height = float(self.model.geom_size[self._table_geom_id, 2])
        else:
            half_height = float(self._table_size[2])
        return float(self._table_center[2] + half_height)

    def _current_object_vertical_extent(self) -> float:
        geom_type = int(self.model.geom_type[self._obj_geom_id])
        geom_size = self.model.geom_size[self._obj_geom_id]
        if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            return float(geom_size[0])
        if geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            return float(geom_size[1])
        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            return float(geom_size[2])
        return float(geom_size[0])

    def _current_object_planar_extent(self) -> float:
        geom_type = int(self.model.geom_type[self._obj_geom_id])
        geom_size = self.model.geom_size[self._obj_geom_id]
        if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            return float(geom_size[0])
        if geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            return float(geom_size[0])
        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            return float(np.linalg.norm(geom_size[:2]))
        return float(geom_size[0])

    def _object_rest_margin(self) -> float:
        geom_type = int(self.model.geom_type[self._obj_geom_id])
        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            return 0.0030
        if geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            return 0.0008
        return 0.0

    def _reset_supported_gap(self) -> float:
        geom_type = int(self.model.geom_type[self._obj_geom_id])
        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            return 0.0015
        if geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            return 0.0008
        return 0.0005

    def _object_rest_height(self) -> float:
        return (
            self._table_surface_z()
            + self._current_object_vertical_extent()
            + self.object_clearance
            + self._object_rest_margin()
        )

    def _apply_object_config(self):
        cfg = OBJECT_CONFIGS[self.object_type]
        geom_type = _GEOM_TYPE_MAP[cfg["type"]]

        self.model.geom_type[self._obj_geom_id] = geom_type
        self.model.geom_size[self._obj_geom_id] = 0.0
        size = np.asarray(cfg["size"], dtype=np.float64)
        self.model.geom_size[self._obj_geom_id, : size.shape[0]] = size
        self.model.geom_rgba[self._obj_geom_id] = np.fromstring(cfg["rgba"], sep=" ", dtype=np.float64)
        self.model.body_mass[self._obj_body_id] = float(cfg["mass"])

    @staticmethod
    def _coupled_q4(q3_rad: float) -> float:
        powers = np.array([1.0, q3_rad, q3_rad**2, q3_rad**3, q3_rad**4], dtype=np.float64)
        return float(np.dot(_Q3Q4_POLY, powers))

    def _set_joint_qpos(self, joint_name: str, value: float):
        self.data.qpos[self._joint_qposadr[joint_name]] = value

    def _apply_hand_pose(self, pose_name: str, obj_pos: Optional[np.ndarray] = None):
        if pose_name not in {"open", "fixed_pregrasp"}:
            raise ValueError(f"Unknown init_hand_pose: {pose_name}")

        wrist = np.zeros(6, dtype=np.float64)
        thumb = np.zeros(4, dtype=np.float64)
        index = np.zeros(3, dtype=np.float64)
        middle = np.zeros(3, dtype=np.float64)

        if pose_name == "fixed_pregrasp":
            if obj_pos is None:
                obj_pos = np.array(
                    [self._initial_obj_xy[0], self._initial_obj_xy[1], self._object_rest_height()],
                    dtype=np.float64,
                )
            wrist_xyz = np.array(
                [
                    obj_pos[0] + _FIXED_PREGRASP["wrist_offset"][0],
                    obj_pos[1] + _FIXED_PREGRASP["wrist_offset"][1],
                    obj_pos[2] + _FIXED_PREGRASP["wrist_offset"][2] - 0.45,
                ],
                dtype=np.float64,
            )
            wrist = np.concatenate([wrist_xyz, _FIXED_PREGRASP["wrist_rot"]])
            thumb = _FIXED_PREGRASP["thumb"].copy()
            index = _FIXED_PREGRASP["index"].copy()
            middle = _FIXED_PREGRASP["middle"].copy()

        for joint_name, value in zip(
            ["wrist_x", "wrist_y", "wrist_z", "wrist_rx", "wrist_ry", "wrist_rz"], wrist
        ):
            self._set_joint_qpos(joint_name, float(value))

        for joint_name, value in zip(["thumb_q1", "thumb_q2", "thumb_q3", "thumb_q4"], thumb):
            self._set_joint_qpos(joint_name, float(value))

        self._set_joint_qpos("index_q1", float(index[0]))
        self._set_joint_qpos("index_q2", float(index[1]))
        self._set_joint_qpos("index_q3", float(index[2]))
        self._set_joint_qpos("index_q4", self._coupled_q4(float(index[2])))

        self._set_joint_qpos("middle_q1", float(middle[0]))
        self._set_joint_qpos("middle_q2", float(middle[1]))
        self._set_joint_qpos("middle_q3", float(middle[2]))
        self._set_joint_qpos("middle_q4", self._coupled_q4(float(middle[2])))

    def _sync_ctrl_to_qpos(self):
        for aid, joint_name in zip(self._act_ids, self._act_joint_names):
            qpos = float(self.data.qpos[self._joint_qposadr[joint_name]])
            lo, hi = self.model.actuator_ctrlrange[aid]
            self.data.ctrl[aid] = float(np.clip(qpos, lo, hi))
        self._reset_action = self.current_normalized_action()

    def current_normalized_action(self) -> np.ndarray:
        action = np.zeros(len(_ALL_ACTUATORS), dtype=np.float32)
        for i, aid in enumerate(self._act_ids):
            lo, hi = self._act_ranges[i]
            ctrl = float(self.data.ctrl[aid])
            action[i] = np.clip(2.0 * (ctrl - lo) / (hi - lo) - 1.0, -1.0, 1.0)
        return action

    def get_reset_action(self) -> np.ndarray:
        return self._reset_action.copy()

    def _object_contact_summary(self) -> Dict[str, Dict[str, Any]]:
        summary: Dict[str, Dict[str, Any]] = {
            finger: {"tip": False, "link": False, "support": False, "geoms": set()}
            for finger in _FINGER_GEOM_GROUPS
        }
        for ic in range(self.data.ncon):
            contact = self.data.contact[ic]
            g1, g2 = contact.geom1, contact.geom2
            if g1 != self._obj_geom_id and g2 != self._obj_geom_id:
                continue
            other = g2 if g1 == self._obj_geom_id else g1
            finger = self._geom_to_finger.get(other)
            if finger is None:
                continue
            geom_name = self._geom_to_name[other]
            entry = summary[finger]
            entry["geoms"].add(geom_name)
            if other == mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, _FINGER_TIP_GEOMS[finger]):
                entry["tip"] = True
            else:
                entry["link"] = True
            if other in self._support_geom_ids[finger]:
                entry["support"] = True
        return summary

    @staticmethod
    def _quat_to_yaw(quat: np.ndarray) -> float:
        w, x, y, z = quat
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return float(np.arctan2(siny_cosp, cosy_cosp))

    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    def _compute_step_metrics(self, obs: Dict[str, np.ndarray]) -> Dict[str, Any]:
        obj_pos = obs["obj_pos"].astype(np.float64)
        obj_quat = obs["obj_quat"].astype(np.float64)
        palm_pos = self.data.xpos[self._palm_body_id].copy()
        palm_mat = self.data.xmat[self._palm_body_id].reshape(3, 3).copy()
        palm_local_obj = palm_mat.T @ (obj_pos - palm_pos)

        tip_positions = np.array([self.data.xpos[bid] for bid in self._tip_body_ids], dtype=np.float64)
        tip_to_obj = np.linalg.norm(tip_positions - obj_pos[None, :], axis=1)
        mean_tip_to_obj = float(np.mean(tip_to_obj))
        palm_target = np.array([-0.015, 0.105, 0.075], dtype=np.float64)
        palm_err = float(
            np.linalg.norm((palm_local_obj - palm_target) * np.array([8.0, 6.0, 12.0]))
        )

        contact_summary = self._object_contact_summary()
        tip_contact_count = int(sum(bool(v["tip"]) for v in contact_summary.values()))
        link_contact_count = int(sum(bool(v["link"]) for v in contact_summary.values()))
        support_fingers = int(sum(bool(v["support"]) for v in contact_summary.values()))

        thumb_support = bool(contact_summary["thumb"]["support"])
        front_support = bool(contact_summary["index"]["support"] or contact_summary["middle"]["support"])
        lateral_drift = float(np.linalg.norm(obj_pos[:2] - self._episode_obj_pos[:2]))
        height_gain = float(obj_pos[2] - self._initial_obj_height)
        height_delta = float(obj_pos[2] - self._prev_obj_pos[2])
        object_yaw = self._quat_to_yaw(obj_quat)
        dt = max(float(self.model.opt.timestep * self.n_substeps), 1e-6)
        object_yaw_rate = self._wrap_to_pi(object_yaw - self._prev_obj_yaw) / dt
        embedded_depth = max(
            self._table_surface_z() + self._current_object_vertical_extent() - float(obj_pos[2]),
            0.0,
        )
        radial_from_table_center = float(np.linalg.norm(obj_pos[:2] - self._table_center[:2]))
        support_radius = self._table_radius - 0.50 * self._current_object_planar_extent()
        object_off_table = radial_from_table_center > support_radius
        object_dropped = float(obj_pos[2]) < (self._table_surface_z() - 0.01)

        return {
            "obj_pos": obj_pos,
            "palm_local_obj": palm_local_obj,
            "tip_positions": tip_positions,
            "tip_to_obj": tip_to_obj,
            "mean_tip_to_obj": mean_tip_to_obj,
            "touch_normal": self.get_touch_normal_forces(),
            "contact_summary": contact_summary,
            "n_tip_contacts": tip_contact_count,
            "n_link_contacts": link_contact_count,
            "n_support_contacts": support_fingers,
            "thumb_support": thumb_support,
            "front_support": front_support,
            "height_gain": height_gain,
            "height_delta": height_delta,
            "object_yaw": float(object_yaw),
            "object_yaw_rate": float(object_yaw_rate),
            "lateral_drift": lateral_drift,
            "embedded_depth": float(embedded_depth),
            "object_off_table": bool(object_off_table),
            "object_dropped": bool(object_dropped),
            "palm_target": palm_target,
            "palm_err": palm_err,
        }

    def _update_episode_stats(self, metrics: Dict[str, Any]):
        enclosure_active = bool(
            metrics["thumb_support"]
            and metrics["front_support"]
            and metrics["n_support_contacts"] >= 2
        )
        if enclosure_active:
            self._stable_enclosure_steps += 1
        else:
            self._stable_enclosure_steps = 0

        self._max_stable_enclosure_steps = max(
            self._max_stable_enclosure_steps, self._stable_enclosure_steps
        )
        self._max_support_contacts = max(
            self._max_support_contacts, int(metrics["n_support_contacts"])
        )
        self._max_height_gain = max(self._max_height_gain, float(metrics["height_gain"]))
        self._max_object_yaw_rate = max(
            self._max_object_yaw_rate, abs(float(metrics["object_yaw_rate"]))
        )
        self._ever_thumb_front_support |= enclosure_active
        self._ever_tip_contact |= metrics["n_tip_contacts"] > 0
        self._ever_link_contact |= metrics["n_link_contacts"] > 0

        metrics["stable_enclosure_steps"] = self._stable_enclosure_steps
        metrics["max_stable_enclosure_steps"] = self._max_stable_enclosure_steps
        metrics["max_support_contacts"] = self._max_support_contacts
        metrics["max_height_gain"] = self._max_height_gain
        metrics["max_object_yaw_rate"] = self._max_object_yaw_rate
        metrics["ever_thumb_front_support"] = self._ever_thumb_front_support
        metrics["ever_tip_contact"] = self._ever_tip_contact
        metrics["ever_link_contact"] = self._ever_link_contact
        metrics["is_success"] = bool(
            self._max_height_gain > 0.02
            and self._max_stable_enclosure_steps >= 6
            and metrics["lateral_drift"] < 0.04
            and metrics["thumb_support"]
            and metrics["front_support"]
        )

    def _compute_reward(self, obs: Dict[str, np.ndarray], metrics: Dict[str, Any]) -> float:
        if self.reward_type == "sparse":
            return self._sparse_reward(metrics)
        return self._dense_reward(metrics)

    def _dense_reward(self, metrics: Dict[str, Any]) -> float:
        delta_tip = float(np.clip(self._prev_tip_mean - metrics["mean_tip_to_obj"], -0.01, 0.01))
        delta_palm = float(np.clip(self._prev_palm_err - metrics["palm_err"], -0.02, 0.02))
        clipped_height_delta = float(np.clip(metrics["height_delta"], -0.004, 0.004))
        opposition_active = bool(metrics["thumb_support"] and metrics["front_support"])
        enclosure_active = bool(opposition_active and metrics["stable_enclosure_steps"] >= 4)
        contact_score = float(metrics["n_tip_contacts"] + 1.5 * metrics["n_link_contacts"])
        delta_contact = float(np.clip(contact_score - self._prev_contact_score, -3.0, 3.0))
        delta_support = float(
            np.clip(metrics["n_support_contacts"] - self._prev_support_contacts, -3.0, 3.0)
        )
        delta_enclosure = float(
            np.clip(metrics["stable_enclosure_steps"] - self._prev_enclosure_steps, -4.0, 4.0)
        )

        r_align_static = 0.08 * np.exp(-12.0 * metrics["mean_tip_to_obj"])
        r_palm_static = 0.12 * np.exp(-metrics["palm_err"])
        r_align_progress = 8.0 * delta_tip
        r_palm_progress = 4.0 * delta_palm
        r_contact = 0.06 * metrics["n_tip_contacts"] + 0.08 * metrics["n_link_contacts"]
        r_contact_progress = 1.20 * max(delta_contact, 0.0)
        r_support = 0.05 * metrics["n_support_contacts"]
        r_support_progress = 1.55 * max(delta_support, 0.0)
        r_opposition = 2.20 if opposition_active and not self._prev_opposition_active else 0.0
        r_opposition_hold = 0.08 if opposition_active else 0.0
        r_enclosure_progress = 0.70 * max(delta_enclosure, 0.0)
        r_hold = 0.03 * min(metrics["stable_enclosure_steps"], 4) if opposition_active else 0.0
        r_lift_abs = 20.0 * max(metrics["height_gain"], 0.0) if enclosure_active else 0.0
        r_lift_delta = 80.0 * max(clipped_height_delta, 0.0) if enclosure_active else 0.0
        success_bonus = 80.0 if metrics["is_success"] else 0.0

        p_drift = 8.0 * metrics["lateral_drift"]
        spin_gate = bool(metrics["n_support_contacts"] >= 2)
        p_spin = (
            0.05 * max(abs(metrics["object_yaw_rate"]) - 1.2, 0.0)
            if spin_gate
            else 0.0
        )
        p_embed = 80.0 * min(metrics["embedded_depth"], 0.002)
        p_contact_stall = (
            0.30 * metrics["n_support_contacts"]
            if metrics["n_support_contacts"] >= 2
            and not opposition_active
            and metrics["height_gain"] < 0.002
            else 0.0
        )
        lazy_hold_deficit = max(0.005 - metrics["height_gain"], 0.0)
        p_lazy_hold = (
            0.22 + 12.0 * lazy_hold_deficit
            if opposition_active and metrics["stable_enclosure_steps"] >= 6
            else 0.0
        )
        p_drop = 18.0 if metrics["object_dropped"] else 0.0
        p_off_table = 10.0 if metrics["object_off_table"] else 0.0
        p_time = 0.08

        return float(
            r_align_static
            + r_palm_static
            + r_align_progress
            + r_palm_progress
            + r_contact
            + r_contact_progress
            + r_support
            + r_support_progress
            + r_opposition
            + r_opposition_hold
            + r_enclosure_progress
            + r_hold
            + r_lift_abs
            + r_lift_delta
            + success_bonus
            - p_drift
            - p_spin
            - p_embed
            - p_contact_stall
            - p_lazy_hold
            - p_drop
            - p_off_table
            - p_time
        )

    @staticmethod
    def _sparse_reward(metrics: Dict[str, Any]) -> float:
        return 1.0 if metrics["is_success"] else 0.0

    def _check_terminated(self, metrics: Dict[str, Any]) -> bool:
        if metrics["is_success"]:
            return True
        if metrics["object_dropped"]:
            return True
        if metrics["object_off_table"]:
            return True
        if metrics["lateral_drift"] > 0.12:
            return True
        return False

    def _get_info(self, obs: Dict[str, np.ndarray], metrics: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "touch_normal": metrics["touch_normal"].copy(),
            "obj_height": float(obs["obj_pos"][2]),
            "height_gain": float(metrics["height_gain"]),
            "height_delta": float(metrics["height_delta"]),
            "max_height_gain": float(metrics["max_height_gain"]),
            "object_yaw_rate": float(metrics["object_yaw_rate"]),
            "max_object_yaw_rate": float(metrics["max_object_yaw_rate"]),
            "n_contacts": int(metrics["n_support_contacts"]),
            "max_contacts": int(metrics["max_support_contacts"]),
            "n_tip_contacts": int(metrics["n_tip_contacts"]),
            "n_link_contacts": int(metrics["n_link_contacts"]),
            "thumb_contact": bool(metrics["contact_summary"]["thumb"]["tip"]),
            "front_contact": bool(
                metrics["contact_summary"]["index"]["tip"] or metrics["contact_summary"]["middle"]["tip"]
            ),
            "thumb_support": bool(metrics["thumb_support"]),
            "front_support": bool(metrics["front_support"]),
            "stable_enclosure_steps": int(metrics["stable_enclosure_steps"]),
            "max_stable_enclosure_steps": int(metrics["max_stable_enclosure_steps"]),
            "lateral_drift": float(metrics["lateral_drift"]),
            "palm_local_obj": metrics["palm_local_obj"].astype(np.float32),
            "mean_tip_to_obj": float(metrics["mean_tip_to_obj"]),
            "palm_err": float(metrics["palm_err"]),
            "object_embedded": bool(metrics["embedded_depth"] > 1e-4),
            "object_off_table": bool(metrics["object_off_table"]),
            "object_type": self.object_type,
            "contact_summary": {
                finger: {
                    "tip": bool(entry["tip"]),
                    "link": bool(entry["link"]),
                    "support": bool(entry["support"]),
                    "geoms": sorted(entry["geoms"]),
                }
                for finger, entry in metrics["contact_summary"].items()
            },
            "is_success": bool(metrics["is_success"]),
        }

    def get_camera_intrinsics(self) -> np.ndarray:
        fovy = self.model.cam_fovy[self._cam_id]
        f = self.cam_height / (2.0 * np.tan(np.radians(fovy) / 2.0))
        cx = self.cam_width / 2.0
        cy = self.cam_height / 2.0
        return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)

    def depth_to_pointcloud(self, depth: np.ndarray) -> np.ndarray:
        K = self.get_camera_intrinsics()
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        h, w = depth.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        z = depth
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return np.stack([x, y, z], axis=-1)

    def set_object_type(self, obj_type: str):
        if obj_type not in OBJECT_CONFIGS:
            raise ValueError(
                f"Unknown object type: {obj_type}. Available: {list(OBJECT_CONFIGS.keys())}"
            )
        self.object_type = obj_type
