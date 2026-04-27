"""
grasp_env.py - 三指手抓取 Gymnasium 环境

提供标准 Gymnasium 接口，支持：
- 掌心相机 RGB + 深度图观测
- 16-DOF 动作空间（手腕6 + 手指10）
- 三指尖三轴触觉力反馈
- 分阶段奖励（接近→接触→抬升→保持）
- 兼容 Stable-Baselines3 / CleanRL 等 RL 框架
"""

import numpy as np
import mujoco
import os
from typing import Optional, Dict, Any, Tuple

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

_XML_PATH = os.path.join(os.path.dirname(__file__), '..', 'mujoco_model', 'jack_hand_3f.xml')

OBJECT_CONFIGS = {
    'sphere': {'type': 'sphere', 'size': [0.015], 'mass': 0.05,
               'rgba': '0.9 0.2 0.2 1'},
    'sphere_small': {'type': 'sphere', 'size': [0.012], 'mass': 0.03,
                     'rgba': '0.2 0.8 0.2 1'},
    'cylinder': {'type': 'cylinder', 'size': [0.01, 0.015], 'mass': 0.05,
                 'rgba': '0.2 0.5 0.9 1'},
    'box': {'type': 'box', 'size': [0.012, 0.012, 0.012], 'mass': 0.05,
            'rgba': '0.9 0.7 0.1 1'},
}

_FINGERTIP_GEOMS = ['thumb_tip_geom', 'index_tip_geom', 'middle_tip_geom']

_WRIST_ACTUATORS = [
    'wrist_act_x', 'wrist_act_y', 'wrist_act_z',
    'wrist_act_rx', 'wrist_act_ry', 'wrist_act_rz',
]
_FINGER_ACTUATORS = [
    'thumb_act_q1', 'thumb_act_q2', 'thumb_act_q3', 'thumb_act_q4',
    'index_act_q1', 'index_act_q2', 'index_act_q3',
    'middle_act_q1', 'middle_act_q2', 'middle_act_q3',
]
_ALL_ACTUATORS = _WRIST_ACTUATORS + _FINGER_ACTUATORS


class JackHandGraspEnv(gym.Env):
    """三指手抓取 Gymnasium 环境

    Observation:
        palm_rgb:    (240, 320, 3) uint8  — 掌心相机 RGB
        palm_depth:  (240, 320) float32   — 掌心相机深度图 (m)
        joint_pos:   (16,) float32        — 关节角 (手腕6+手指10)
        joint_vel:   (16,) float32        — 关节速度
        touch_force: (3, 3) float32       — 三指×三轴力 (N)
        obj_pos:     (3,) float32         — 物体位置 (m)
        obj_quat:    (4,) float32         — 物体姿态

    Action:
        (16,) float32 in [-1, 1] — 归一化关节位置目标
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 25}

    def __init__(self, render_mode: Optional[str] = None,
                 object_type: str = "sphere",
                 cam_width: int = 320, cam_height: int = 240,
                 n_substeps: int = 20,
                 reward_type: str = "dense",
                 randomize_object_radius: bool = False,
                 object_radius_range: Tuple[float, float] = (0.016, 0.024),
                 randomize_object_friction: bool = False,
                 object_friction_range: Tuple[float, float] = (0.7, 1.3)):
        super().__init__()

        self.render_mode = render_mode
        self.object_type = object_type
        self.cam_width = cam_width
        self.cam_height = cam_height
        self.n_substeps = n_substeps
        self.reward_type = reward_type
        self.randomize_object_radius = randomize_object_radius
        self.object_radius_range = object_radius_range
        self.randomize_object_friction = randomize_object_friction
        self.object_friction_range = object_friction_range

        self.model = mujoco.MjModel.from_xml_path(_XML_PATH)
        self.data = mujoco.MjData(self.model)

        self._renderer = None
        self._viewer = None

        self._cache_ids()
        self._build_spaces()

        self._initial_obj_pos = np.array([-0.02, 0.08, 0.265])
        self._obj_spawn_range = np.array([0.03, 0.03, 0.02])

        self._step_count = 0
        self._max_steps = 250

    def _cache_ids(self):
        m = self.model
        self._act_ids = []
        self._act_ranges = np.zeros((len(_ALL_ACTUATORS), 2))
        for i, name in enumerate(_ALL_ACTUATORS):
            aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            self._act_ids.append(aid)
            self._act_ranges[i] = m.actuator_ctrlrange[aid]

        self._tip_geom_ids = []
        for name in _FINGERTIP_GEOMS:
            self._tip_geom_ids.append(
                mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name))

        self._obj_geom_id = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_GEOM, 'obj_geom')
        self._obj_body_id = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_BODY, 'target_object')
        self._palm_body_id = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_BODY, 'palm')
        self._cam_id = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_CAMERA, 'palm_cam')

        self._obj_jnt_adr = m.body_jntadr[self._obj_body_id]
        self._obj_qpos_adr = m.jnt_qposadr[self._obj_jnt_adr]
        self._default_obj_size = m.geom_size[self._obj_geom_id].copy()
        self._default_obj_friction = m.geom_friction[self._obj_geom_id].copy()

        self._touch_sensor_ids = []
        for name in ['thumb_touch', 'index_touch', 'middle_touch']:
            self._touch_sensor_ids.append(
                mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, name))

        self._tip_body_ids = []
        for name in ['thumb_tip', 'index_tip', 'middle_tip']:
            self._tip_body_ids.append(
                mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name))

    def _build_spaces(self):
        self.observation_space = spaces.Dict({
            "palm_rgb": spaces.Box(0, 255,
                                   (self.cam_height, self.cam_width, 3),
                                   dtype=np.uint8),
            "palm_depth": spaces.Box(0.0, 2.0,
                                     (self.cam_height, self.cam_width),
                                     dtype=np.float32),
            "joint_pos": spaces.Box(-np.pi, np.pi, (16,), dtype=np.float32),
            "joint_vel": spaces.Box(-50.0, 50.0, (16,), dtype=np.float32),
            "touch_force": spaces.Box(-50.0, 50.0, (3, 3), dtype=np.float32),
            "obj_pos": spaces.Box(-1.0, 1.0, (3,), dtype=np.float32),
            "obj_quat": spaces.Box(-1.0, 1.0, (4,), dtype=np.float32),
        })
        self.action_space = spaces.Box(-1.0, 1.0, (16,), dtype=np.float32)

    # ===== Gymnasium interface =====

    def reset(self, *, seed=None, options=None) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        self.model.geom_size[self._obj_geom_id] = self._default_obj_size
        self.model.geom_friction[self._obj_geom_id] = self._default_obj_friction

        if self.object_type.startswith("sphere") and self.randomize_object_radius:
            radius = float(self.np_random.uniform(*self.object_radius_range))
            self.model.geom_size[self._obj_geom_id, 0] = radius

        if self.randomize_object_friction:
            slide = float(self.np_random.uniform(*self.object_friction_range))
            self.model.geom_friction[self._obj_geom_id, 0] = slide

        obj_pos = self._initial_obj_pos.copy()
        if self.np_random is not None:
            obj_pos += self.np_random.uniform(
                -self._obj_spawn_range, self._obj_spawn_range)

        adr = self._obj_qpos_adr
        self.data.qpos[adr:adr + 3] = obj_pos
        self.data.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]

        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        self._prev_obj_pos = obj_pos.copy()
        self._episode_obj_pos = obj_pos.copy()
        self._initial_obj_height = obj_pos[2]

        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        action = np.clip(action, -1.0, 1.0)

        ctrl = np.zeros(self.model.nu)
        for i, aid in enumerate(self._act_ids):
            lo, hi = self._act_ranges[i]
            ctrl[aid] = lo + (action[i] + 1.0) * 0.5 * (hi - lo)

        self.data.ctrl[:] = ctrl
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        obs = self._get_obs()
        reward = self._compute_reward(obs)
        terminated = self._check_terminated(obs)
        truncated = self._step_count >= self._max_steps
        info = self._get_info(obs)

        self._prev_obj_pos = obs["obj_pos"].copy()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_camera_rgb("palm_cam")
        elif self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(
                    self.model, self.data)
            self._viewer.sync()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    # ===== Observation =====

    def _get_obs(self) -> Dict[str, np.ndarray]:
        rgb, depth = self._render_palm_camera()

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
        obj_pos = self.data.qpos[adr:adr + 3].copy().astype(np.float32)
        obj_quat = self.data.qpos[adr + 3:adr + 7].copy().astype(np.float32)

        return {
            "palm_rgb": rgb,
            "palm_depth": depth,
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "touch_force": touch_force,
            "obj_pos": obj_pos,
            "obj_quat": obj_quat,
        }

    # ===== Camera =====

    def _get_renderer(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model, height=self.cam_height, width=self.cam_width)
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
        cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        renderer.update_scene(self.data, camera=cam_id)
        return renderer.render().copy()

    # ===== Touch forces =====

    def _get_touch_forces(self) -> np.ndarray:
        """提取三指尖的三轴接触力 (fx, fy, fz)。

        fz 为法向力（指向指尖内表面），fx/fy 为切向力。
        使用 mj_contactForce 提取并投影到指尖局部坐标系。
        """
        forces = np.zeros((3, 3), dtype=np.float32)

        for ic in range(self.data.ncon):
            contact = self.data.contact[ic]
            g1, g2 = contact.geom1, contact.geom2

            for fi, tip_gid in enumerate(self._tip_geom_ids):
                if g1 == tip_gid or g2 == tip_gid:
                    if not (g1 == self._obj_geom_id or g2 == self._obj_geom_id):
                        continue
                    c_force = np.zeros(6)
                    mujoco.mj_contactForce(self.model, self.data, ic, c_force)
                    fn = c_force[0]
                    ft1 = c_force[1]
                    ft2 = c_force[2]
                    frame = contact.frame.reshape(3, 3)
                    tip_body_id = self._tip_body_ids[fi]
                    tip_xmat = self.data.xmat[tip_body_id].reshape(3, 3)
                    f_world = frame.T @ np.array([fn, ft1, ft2])
                    f_local = tip_xmat.T @ f_world
                    forces[fi] = f_local.astype(np.float32)
                    break

        return forces

    def get_touch_normal_forces(self) -> np.ndarray:
        """简化版：只返回三指法向力标量 (N)。"""
        forces = np.zeros(3, dtype=np.float32)
        for i, sid in enumerate(self._touch_sensor_ids):
            adr = self.model.sensor_adr[sid]
            forces[i] = self.data.sensordata[adr]
        return forces

    # ===== Reward =====

    def _compute_reward(self, obs: Dict) -> float:
        if self.reward_type == "sparse":
            return self._sparse_reward(obs)
        return self._dense_reward(obs)

    def _dense_reward(self, obs: Dict) -> float:
        obj_pos = obs["obj_pos"]
        palm_pos = self.data.xpos[self._palm_body_id]

        tip_positions = np.array([
            self.data.xpos[bid] for bid in self._tip_body_ids])
        tip_to_obj = np.linalg.norm(tip_positions - obj_pos, axis=1)
        r_approach = -np.mean(tip_to_obj) * 10.0

        touch = self.get_touch_normal_forces()
        n_contacts = np.sum(touch > 0.01)
        r_contact = n_contacts * 2.0

        height_gain = obj_pos[2] - self._initial_obj_height
        r_lift = max(0, height_gain) * 100.0

        if n_contacts >= 2 and height_gain > 0.005:
            r_hold = 5.0
        else:
            r_hold = 0.0

        return r_approach + r_contact + r_lift + r_hold

    def _sparse_reward(self, obs: Dict) -> float:
        obj_pos = obs["obj_pos"]
        height_gain = obj_pos[2] - self._initial_obj_height
        if height_gain > 0.02:
            return 1.0
        return 0.0

    def _check_terminated(self, obs: Dict) -> bool:
        obj_pos = obs["obj_pos"]
        if obj_pos[2] < 0.0:
            return True
        if np.linalg.norm(obj_pos - self._initial_obj_pos) > 0.3:
            return True
        return False

    def _get_info(self, obs: Dict) -> Dict[str, Any]:
        touch = self.get_touch_normal_forces()
        obj_pos = obs["obj_pos"]
        return {
            "touch_normal": touch,
            "obj_height": obj_pos[2],
            "height_gain": obj_pos[2] - self._initial_obj_height,
            "n_contacts": int(np.sum(touch > 0.01)),
            "is_success": obj_pos[2] - self._initial_obj_height > 0.02,
        }

    # ===== Utilities =====

    def get_camera_intrinsics(self) -> np.ndarray:
        """返回掌心相机 3x3 内参矩阵。"""
        fovy = self.model.cam_fovy[self._cam_id]
        f = self.cam_height / (2.0 * np.tan(np.radians(fovy) / 2.0))
        cx = self.cam_width / 2.0
        cy = self.cam_height / 2.0
        return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)

    def depth_to_pointcloud(self, depth: np.ndarray) -> np.ndarray:
        """将深度图转换为相机坐标系下的三维点云。"""
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
        """切换物体类型（需要重新加载模型）。"""
        if obj_type not in OBJECT_CONFIGS:
            raise ValueError(f"Unknown object type: {obj_type}. "
                             f"Available: {list(OBJECT_CONFIGS.keys())}")
        self.object_type = obj_type
