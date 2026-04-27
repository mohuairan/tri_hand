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
    """State-only wrapper better suited for lightweight RL training.

    The base MuJoCo env exposes images plus proprioception. For the first RL
    stage we use a compact state vector and delta actions, which are far easier
    to optimize than end-to-end pixels with absolute joint targets.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 25}

    def __init__(self,
                 render_mode=None,
                 object_type: str = "sphere",
                 n_substeps: int = 20,
                 reward_type: str = "dense",
                 action_mode: str = "delta",
                 action_step: float = 0.12,
                 include_last_action: bool = True,
                 randomize_object_radius: bool = True,
                 object_radius_range=(0.016, 0.024),
                 randomize_object_friction: bool = True,
                 object_friction_range=(0.7, 1.3)):
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
        )
        self.render_mode = render_mode
        self.action_mode = action_mode
        self.action_step = float(action_step)
        self.include_last_action = include_last_action

        self.action_space = spaces.Box(-1.0, 1.0, (16,), dtype=np.float32)
        obs_dim = self._state_dim()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self._last_env_action = np.zeros(self.action_space.shape, dtype=np.float32)

    def _state_dim(self) -> int:
        dim = 16 + 16 + 9 + 3 + 9 + 4 + 1
        if self.include_last_action:
            dim += 16
        return dim

    def _tip_positions(self) -> np.ndarray:
        return np.array(
            [self.base_env.data.xpos[bid] for bid in self.base_env._tip_body_ids],
            dtype=np.float32,
        )

    def _palm_position(self) -> np.ndarray:
        return self.base_env.data.xpos[self.base_env._palm_body_id].astype(np.float32)

    def _build_state(self, obs: dict) -> np.ndarray:
        palm_pos = self._palm_position()
        tip_pos = self._tip_positions()
        obj_pos = obs["obj_pos"].astype(np.float32)
        obj_quat = obs["obj_quat"].astype(np.float32)

        obj_rel_palm = obj_pos - palm_pos
        tip_rel_obj = (tip_pos - obj_pos[None, :]).reshape(-1)
        height_gain = np.array(
            [obj_pos[2] - self.base_env._initial_obj_height], dtype=np.float32)

        parts = [
            obs["joint_pos"].astype(np.float32),
            obs["joint_vel"].astype(np.float32),
            obs["touch_force"].reshape(-1).astype(np.float32),
            obj_rel_palm.astype(np.float32),
            tip_rel_obj.astype(np.float32),
            obj_quat.astype(np.float32),
            height_gain,
        ]
        if self.include_last_action:
            parts.append(self._last_env_action.astype(np.float32))
        return np.concatenate(parts, axis=0).astype(np.float32)

    def _compose_env_action(self, action: np.ndarray) -> np.ndarray:
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        if self.action_mode == "absolute":
            env_action = action
        elif self.action_mode == "delta":
            env_action = np.clip(
                self._last_env_action + self.action_step * action, -1.0, 1.0)
        else:
            raise ValueError(f"Unknown action_mode: {self.action_mode}")
        self._last_env_action = env_action.astype(np.float32)
        return env_action

    def _shape_reward(self, obs: dict, base_reward: float, info: dict) -> float:
        obj_pos = obs["obj_pos"]
        lateral_drift = np.linalg.norm(
            obj_pos[:2] - self.base_env._episode_obj_pos[:2])
        tip_touch = self.base_env.get_touch_normal_forces()
        thumb_contact = float(tip_touch[0] > 0.01)
        front_contact = float((tip_touch[1] > 0.01) or (tip_touch[2] > 0.01))

        reward = float(base_reward)
        reward += 1.5 * thumb_contact * front_contact
        reward -= 3.0 * lateral_drift
        if obj_pos[2] < self.base_env._initial_obj_height - 0.01:
            reward -= 2.0

        info["thumb_contact"] = bool(thumb_contact)
        info["front_contact"] = bool(front_contact)
        info["lateral_drift"] = float(lateral_drift)
        return reward

    def reset(self, *, seed=None, options=None):
        self._last_env_action[:] = 0.0
        obs, info = self.base_env.reset(seed=seed, options=options)
        state = self._build_state(obs)
        info = dict(info)
        info["raw_obs"] = obs
        return state, info

    def step(self, action):
        env_action = self._compose_env_action(action)
        obs, reward, terminated, truncated, info = self.base_env.step(env_action)
        state = self._build_state(obs)
        info = dict(info)
        info["raw_obs"] = obs
        reward = self._shape_reward(obs, reward, info)
        return state, reward, terminated, truncated, info

    def render(self):
        return self.base_env.render()

    def close(self):
        self.base_env.close()
