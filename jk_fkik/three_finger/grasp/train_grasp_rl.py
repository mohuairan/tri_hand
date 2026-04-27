"""Train a lightweight RL baseline for three-finger grasping.

This script uses Augmented Random Search (ARS), which works well as a no-extra-
dependency baseline for continuous control. It is not the final word in policy
learning, but it gives us a trainable RL loop immediately in the current env.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass

import numpy as np

from grasp_rl_env import JackHandStateEnv


@dataclass
class TrainConfig:
    iterations: int = 200
    directions: int = 12
    top_directions: int = 6
    noise_std: float = 0.03
    step_size: float = 0.025
    seed: int = 0
    eval_every: int = 10
    eval_episodes: int = 5
    action_mode: str = "delta"
    action_step: float = 0.12
    reward_type: str = "dense"
    n_substeps: int = 20
    randomize_object_radius: bool = True
    randomize_object_friction: bool = True
    save_dir: str = "jk_fkik/three_finger/grasp/checkpoints"


class RunningNormalizer:
    def __init__(self, obs_dim: int):
        self.count = 1e-4
        self.mean = np.zeros(obs_dim, dtype=np.float64)
        self.m2 = np.ones(obs_dim, dtype=np.float64)

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        self.count += 1.0
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.m2 += delta * delta2

    def std(self) -> np.ndarray:
        return np.sqrt(np.maximum(self.m2 / self.count, 1e-6))

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std()


class LinearPolicy:
    def __init__(self, obs_dim: int, act_dim: int):
        self.weights = np.zeros((act_dim, obs_dim + 1), dtype=np.float64)

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs_aug = np.concatenate([obs.astype(np.float64), np.ones(1, dtype=np.float64)])
        return np.tanh(self.weights @ obs_aug).astype(np.float32)

    def clone(self) -> "LinearPolicy":
        other = LinearPolicy(self.weights.shape[1] - 1, self.weights.shape[0])
        other.weights[:] = self.weights
        return other


def rollout(env: JackHandStateEnv,
            policy: LinearPolicy,
            normalizer: RunningNormalizer,
            seed: int,
            update_normalizer: bool = True,
            max_steps: int | None = None):
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    total_steps = 0
    last_info = dict(info)

    while True:
        if update_normalizer:
            normalizer.update(obs)
        norm_obs = normalizer.normalize(obs)
        action = policy.act(norm_obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        total_steps += 1
        last_info = dict(info)
        if terminated or truncated or (max_steps is not None and total_steps >= max_steps):
            break

    return total_reward, total_steps, last_info


def evaluate(policy: LinearPolicy,
             normalizer: RunningNormalizer,
             cfg: TrainConfig,
             episodes: int,
             seed_offset: int):
    env = JackHandStateEnv(
        render_mode=None,
        action_mode=cfg.action_mode,
        action_step=cfg.action_step,
        reward_type=cfg.reward_type,
        n_substeps=cfg.n_substeps,
        randomize_object_radius=cfg.randomize_object_radius,
        randomize_object_friction=cfg.randomize_object_friction,
    )
    returns = []
    successes = []
    height_gains = []
    contacts = []
    for ep in range(episodes):
        ret, _, info = rollout(
            env, policy, normalizer, seed=seed_offset + ep, update_normalizer=False)
        returns.append(ret)
        successes.append(float(info.get("is_success", False)))
        height_gains.append(float(info.get("height_gain", 0.0)))
        contacts.append(float(info.get("n_contacts", 0)))
    env.close()
    return {
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "success_rate": float(np.mean(successes)),
        "height_gain_mean": float(np.mean(height_gains)),
        "contacts_mean": float(np.mean(contacts)),
    }


def save_checkpoint(path: str,
                    policy: LinearPolicy,
                    normalizer: RunningNormalizer,
                    cfg: TrainConfig,
                    stats: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(
        path,
        weights=policy.weights,
        obs_mean=normalizer.mean,
        obs_m2=normalizer.m2,
        obs_count=np.array([normalizer.count], dtype=np.float64),
        config_json=np.array([json.dumps(asdict(cfg))]),
        stats_json=np.array([json.dumps(stats)]),
    )


def train(cfg: TrainConfig):
    rng = np.random.default_rng(cfg.seed)
    env = JackHandStateEnv(
        render_mode=None,
        action_mode=cfg.action_mode,
        action_step=cfg.action_step,
        reward_type=cfg.reward_type,
        n_substeps=cfg.n_substeps,
        randomize_object_radius=cfg.randomize_object_radius,
        randomize_object_friction=cfg.randomize_object_friction,
    )
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    normalizer = RunningNormalizer(obs_dim)
    policy = LinearPolicy(obs_dim, act_dim)
    best_success = -1.0
    best_stats = {}

    for it in range(1, cfg.iterations + 1):
        deltas = rng.standard_normal(
            size=(cfg.directions, act_dim, obs_dim + 1)).astype(np.float64)
        rewards_pos = np.zeros(cfg.directions, dtype=np.float64)
        rewards_neg = np.zeros(cfg.directions, dtype=np.float64)

        for k in range(cfg.directions):
            policy_pos = policy.clone()
            policy_neg = policy.clone()
            policy_pos.weights += cfg.noise_std * deltas[k]
            policy_neg.weights -= cfg.noise_std * deltas[k]

            seed_base = cfg.seed * 100000 + it * 1000 + k * 10
            rewards_pos[k], _, _ = rollout(
                env, policy_pos, normalizer, seed=seed_base, update_normalizer=True)
            rewards_neg[k], _, _ = rollout(
                env, policy_neg, normalizer, seed=seed_base + 1, update_normalizer=True)

        scores = np.maximum(rewards_pos, rewards_neg)
        top_idx = np.argsort(scores)[-cfg.top_directions:]
        reward_std = np.std(np.concatenate([rewards_pos[top_idx], rewards_neg[top_idx]]))
        reward_std = max(reward_std, 1e-6)

        step = np.zeros_like(policy.weights)
        for idx in top_idx:
            step += (rewards_pos[idx] - rewards_neg[idx]) * deltas[idx]
        policy.weights += (cfg.step_size / (cfg.top_directions * reward_std)) * step

        train_stats = {
            "iter": it,
            "reward_pos_mean": float(np.mean(rewards_pos)),
            "reward_neg_mean": float(np.mean(rewards_neg)),
            "reward_pos_max": float(np.max(rewards_pos)),
            "reward_neg_max": float(np.max(rewards_neg)),
        }

        if it == 1 or it % cfg.eval_every == 0 or it == cfg.iterations:
            eval_stats = evaluate(policy, normalizer, cfg, cfg.eval_episodes, cfg.seed + it * 100)
            merged = {**train_stats, **eval_stats}
            print(json.dumps(merged, ensure_ascii=True))

            latest_path = os.path.join(cfg.save_dir, "ars_latest.npz")
            save_checkpoint(latest_path, policy, normalizer, cfg, merged)

            if eval_stats["success_rate"] >= best_success:
                best_success = eval_stats["success_rate"]
                best_stats = merged
                best_path = os.path.join(cfg.save_dir, "ars_best.npz")
                save_checkpoint(best_path, policy, normalizer, cfg, merged)
        else:
            print(json.dumps(train_stats, ensure_ascii=True))

    env.close()
    return best_stats


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--directions", type=int, default=12)
    parser.add_argument("--top-directions", type=int, default=6)
    parser.add_argument("--noise-std", type=float, default=0.03)
    parser.add_argument("--step-size", type=float, default=0.025)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--action-mode", type=str, default="delta",
                        choices=["delta", "absolute"])
    parser.add_argument("--action-step", type=float, default=0.12)
    parser.add_argument("--reward-type", type=str, default="dense",
                        choices=["dense", "sparse"])
    parser.add_argument("--n-substeps", type=int, default=20)
    parser.add_argument("--save-dir", type=str,
                        default="jk_fkik/three_finger/grasp/checkpoints")
    parser.add_argument("--no-radius-randomization", action="store_true")
    parser.add_argument("--no-friction-randomization", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = TrainConfig(
        iterations=args.iterations,
        directions=args.directions,
        top_directions=args.top_directions,
        noise_std=args.noise_std,
        step_size=args.step_size,
        seed=args.seed,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        action_mode=args.action_mode,
        action_step=args.action_step,
        reward_type=args.reward_type,
        n_substeps=args.n_substeps,
        randomize_object_radius=not args.no_radius_randomization,
        randomize_object_friction=not args.no_friction_randomization,
        save_dir=args.save_dir,
    )
    best_stats = train(cfg)
    print("best_stats", json.dumps(best_stats, ensure_ascii=True))


if __name__ == "__main__":
    main()
