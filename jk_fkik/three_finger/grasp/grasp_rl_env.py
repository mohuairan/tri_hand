"""Low-dimensional RL wrapper for the three-finger grasp environment."""

from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    import gym
    from gym import spaces

from grasp_env import JackHandGraspEnv


class JackHandStateEnv(gym.Env):
    """State-only wrapper with delta actions and envelope-aware shaping."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 25}

    def __init__(
        self,
        render_mode=None,
        object_type: str = "sphere",
        n_substeps: int = 20,
        reward_type: str = "dense",
        action_mode: str = "delta",
        wrist_translation_step: float = 0.015,
        finger_action_step: float = 0.06,
        lock_wrist_rotation: bool = True,
        include_last_action: bool = True,
        randomize_object_radius: bool = False,
        object_radius_range=(0.023, 0.028),
        randomize_object_friction: bool = False,
        object_friction_range=(0.7, 1.3),
    ):
        super().__init__()

        self.base_env = JackHandGraspEnv(
            render_mode=render_mode,
            object_type=object_type,
            n_substeps=n_substeps,
            reward_type=reward_type,
            include_camera_obs=False,
            randomize_object_radius=randomize_object_radius,
            object_radius_range=object_radius_range,
            randomize_object_friction=randomize_object_friction,
            object_friction_range=object_friction_range,
            init_hand_pose="fixed_pregrasp",
        )
        self.render_mode = render_mode
        self.action_mode = action_mode
        self.wrist_translation_step = float(wrist_translation_step)
        self.finger_action_step = float(finger_action_step)
        self.lock_wrist_rotation = bool(lock_wrist_rotation)
        self.include_last_action = include_last_action

        self._env_action_dim = 16
        self._policy_action_dim = 13 if self.lock_wrist_rotation else self._env_action_dim
        self.action_space = spaces.Box(-1.0, 1.0, (self._policy_action_dim,), dtype=np.float32)
        obs_dim = self._state_dim()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._last_env_action = np.zeros(self._env_action_dim, dtype=np.float32)
        self._reset_env_action = np.zeros(self._env_action_dim, dtype=np.float32)
        self._wrist_rotation_anchor = np.zeros(3, dtype=np.float32)

    def _state_dim(self) -> int:
        dim = 16 + 16 + 9 + 3 + 9 + 4 + 1 + 9
        if self.include_last_action:
            dim += self._env_action_dim
        return dim

    def _tip_positions(self) -> np.ndarray:
        return np.array(
            [self.base_env.data.xpos[bid] for bid in self.base_env._tip_body_ids],
            dtype=np.float32,
        )

    def _palm_position(self) -> np.ndarray:
        return self.base_env.data.xpos[self.base_env._palm_body_id].astype(np.float32)

    def _contact_features(self) -> np.ndarray:
        summary = self.base_env._last_metrics.get("contact_summary", {})
        feats = []
        for finger in ("thumb", "index", "middle"):
            entry = summary.get(finger, {})
            feats.extend(
                [
                    float(entry.get("tip", False)),
                    float(entry.get("link", False)),
                    float(entry.get("support", False)),
                ]
            )
        return np.asarray(feats, dtype=np.float32)

    def _build_state(self, obs: dict) -> np.ndarray:
        palm_pos = self._palm_position()
        tip_pos = self._tip_positions()
        obj_pos = obs["obj_pos"].astype(np.float32)
        obj_quat = obs["obj_quat"].astype(np.float32)

        obj_rel_palm = obj_pos - palm_pos
        tip_rel_obj = (tip_pos - obj_pos[None, :]).reshape(-1)
        height_gain = np.array(
            [obj_pos[2] - self.base_env._initial_obj_height], dtype=np.float32
        )

        parts = [
            obs["joint_pos"].astype(np.float32),
            obs["joint_vel"].astype(np.float32),
            obs["touch_force"].reshape(-1).astype(np.float32),
            obj_rel_palm.astype(np.float32),
            tip_rel_obj.astype(np.float32),
            obj_quat.astype(np.float32),
            height_gain,
            self._contact_features(),
        ]
        if self.include_last_action:
            parts.append(self._last_env_action.astype(np.float32))
        return np.concatenate(parts, axis=0).astype(np.float32)

    def _compose_env_action(self, action: np.ndarray) -> np.ndarray:
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        env_action = self._last_env_action.copy()
        if self.lock_wrist_rotation:
            wrist_xyz_action = action[:3]
            finger_action = action[3:]
            if self.action_mode == "absolute":
                env_action[:3] = wrist_xyz_action
                env_action[6:] = finger_action
            elif self.action_mode == "delta":
                env_action[:3] = np.clip(
                    self._last_env_action[:3] + self.wrist_translation_step * wrist_xyz_action,
                    -1.0,
                    1.0,
                )
                env_action[6:] = np.clip(
                    self._last_env_action[6:] + self.finger_action_step * finger_action,
                    -1.0,
                    1.0,
                )
            else:
                raise ValueError(f"Unknown action_mode: {self.action_mode}")
            env_action[3:6] = self._wrist_rotation_anchor
        else:
            if self.action_mode == "absolute":
                env_action = action
            elif self.action_mode == "delta":
                env_action = np.clip(
                    self._last_env_action + self.finger_action_step * action, -1.0, 1.0
                )
            else:
                raise ValueError(f"Unknown action_mode: {self.action_mode}")
        self._last_env_action = env_action.astype(np.float32)
        return env_action

    def _shape_reward(self, base_reward: float, info: dict) -> float:
        return float(base_reward)

    def reset(self, *, seed=None, options=None):
        obs, info = self.base_env.reset(seed=seed, options=options)
        self._last_env_action = self.base_env.get_reset_action().astype(np.float32)
        self._reset_env_action = self._last_env_action.copy()
        self._wrist_rotation_anchor = self._reset_env_action[3:6].copy()
        state = self._build_state(obs)
        info = dict(info)
        info["raw_obs"] = obs
        info["initial_action"] = self._last_env_action.copy()
        return state, info

    def step(self, action):
        env_action = self._compose_env_action(action)
        obs, reward, terminated, truncated, info = self.base_env.step(env_action)
        state = self._build_state(obs)
        info = dict(info)
        info["raw_obs"] = obs
        reward = self._shape_reward(reward, info)
        return state, reward, terminated, truncated, info

    def render(self):
        return self.base_env.render()

    def close(self):
        self.base_env.close()
