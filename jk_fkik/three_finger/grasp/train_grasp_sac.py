"""Train a PyTorch Soft Actor-Critic agent for three-finger grasping."""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass

import numpy as np
import torch

from grasp_rl_env import JackHandStateEnv
from sac_agent import ReplayBuffer, SACAgent, SACConfig


@dataclass
class TrainConfig:
    total_steps: int = 300_000
    seed: int = 0
    warmup_steps: int = 5_000
    update_after: int = 2_000
    updates_per_step: int = 1
    eval_interval: int = 10_000
    eval_episodes: int = 10
    action_mode: str = "delta"
    action_step: float = 0.12
    reward_type: str = "dense"
    n_substeps: int = 20
    randomize_object_radius: bool = True
    randomize_object_friction: bool = True
    save_dir: str = "jk_fkik/three_finger/grasp/checkpoints_sac"
    save_best_by: str = "success_rate"


def make_env(cfg: TrainConfig, render_mode=None):
    return JackHandStateEnv(
        render_mode=render_mode,
        action_mode=cfg.action_mode,
        action_step=cfg.action_step,
        reward_type=cfg.reward_type,
        n_substeps=cfg.n_substeps,
        randomize_object_radius=cfg.randomize_object_radius,
        randomize_object_friction=cfg.randomize_object_friction,
    )


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(agent: SACAgent, cfg: TrainConfig, seed_offset: int):
    env = make_env(cfg, render_mode=None)
    returns = []
    successes = []
    height_gains = []
    contacts = []
    drifts = []

    for ep in range(cfg.eval_episodes):
        obs, _ = env.reset(seed=seed_offset + ep)
        done = False
        total_reward = 0.0
        last_info = {}
        while not done:
            action = agent.act(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            done = terminated or truncated
            last_info = info
        returns.append(total_reward)
        successes.append(float(last_info.get("is_success", False)))
        height_gains.append(float(last_info.get("height_gain", 0.0)))
        contacts.append(float(last_info.get("n_contacts", 0)))
        drifts.append(float(last_info.get("lateral_drift", 0.0)))

    env.close()
    return {
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "success_rate": float(np.mean(successes)),
        "height_gain_mean": float(np.mean(height_gains)),
        "contacts_mean": float(np.mean(contacts)),
        "lateral_drift_mean": float(np.mean(drifts)),
    }


def save_checkpoint(path: str, agent: SACAgent, train_cfg: TrainConfig, extra_stats: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = agent.checkpoint_dict()
    ckpt["train_cfg"] = asdict(train_cfg)
    ckpt["stats"] = extra_stats
    torch.save(ckpt, path)


def train(train_cfg: TrainConfig, sac_cfg: SACConfig):
    set_seed(train_cfg.seed)
    env = make_env(train_cfg, render_mode=None)

    obs, _ = env.reset(seed=train_cfg.seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = SACAgent(obs_dim, act_dim, sac_cfg)
    replay = ReplayBuffer(obs_dim, act_dim, sac_cfg.replay_size)

    if sac_cfg.normalize_obs:
        agent.update_obs_norm(obs)

    best_score = -float("inf")
    best_stats = {}
    episode_return = 0.0
    episode_len = 0
    episode_idx = 0
    latest_update = {}

    for step in range(1, train_cfg.total_steps + 1):
        if step <= train_cfg.warmup_steps:
            action = env.action_space.sample().astype(np.float32)
        else:
            action = agent.act(obs, deterministic=False)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated)
        replay.add(obs, action, reward, next_obs, float(done))
        if sac_cfg.normalize_obs:
            agent.update_obs_norm(obs, next_obs)

        obs = next_obs
        episode_return += float(reward)
        episode_len += 1

        if step >= train_cfg.update_after and replay.size >= sac_cfg.batch_size:
            for _ in range(train_cfg.updates_per_step):
                latest_update = agent.update(replay)

        if terminated or truncated:
            episode_idx += 1
            episode_stats = {
                "iter_type": "train_ep",
                "step": step,
                "episode": episode_idx,
                "episode_return": float(episode_return),
                "episode_len": int(episode_len),
                "is_success": bool(info.get("is_success", False)),
                "height_gain": float(info.get("height_gain", 0.0)),
                "lateral_drift": float(info.get("lateral_drift", 0.0)),
            }
            if latest_update:
                episode_stats.update({
                    "actor_loss": latest_update["actor_loss"],
                    "critic_loss": latest_update["critic_loss"],
                    "alpha": latest_update["alpha"],
                })
            print(json.dumps(episode_stats, ensure_ascii=True))
            obs, _ = env.reset(seed=train_cfg.seed + episode_idx)
            episode_return = 0.0
            episode_len = 0
            if sac_cfg.normalize_obs:
                agent.update_obs_norm(obs)

        if step % train_cfg.eval_interval == 0:
            eval_stats = evaluate(agent, train_cfg, train_cfg.seed + step * 13)
            stats = {
                "iter_type": "eval",
                "step": step,
                "replay_size": int(replay.size),
                **eval_stats,
            }
            if latest_update:
                stats.update({
                    "actor_loss": latest_update["actor_loss"],
                    "critic_loss": latest_update["critic_loss"],
                    "alpha": latest_update["alpha"],
                    "q1_mean": latest_update["q1_mean"],
                    "q2_mean": latest_update["q2_mean"],
                    "logp_mean": latest_update["logp_mean"],
                })
            print(json.dumps(stats, ensure_ascii=True))

            latest_path = os.path.join(train_cfg.save_dir, "sac_latest.pt")
            save_checkpoint(latest_path, agent, train_cfg, stats)

            score = eval_stats[train_cfg.save_best_by]
            if score >= best_score:
                best_score = score
                best_stats = stats
                best_path = os.path.join(train_cfg.save_dir, "sac_best.pt")
                save_checkpoint(best_path, agent, train_cfg, stats)

    env.close()
    return best_stats


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=300000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=5000)
    parser.add_argument("--update-after", type=int, default=2000)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--eval-interval", type=int, default=10000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--action-mode", type=str, default="delta", choices=["delta", "absolute"])
    parser.add_argument("--action-step", type=float, default=0.12)
    parser.add_argument("--reward-type", type=str, default="dense", choices=["dense", "sparse"])
    parser.add_argument("--n-substeps", type=int, default=20)
    parser.add_argument("--save-dir", type=str,
                        default="jk_fkik/three_finger/grasp/checkpoints_sac")
    parser.add_argument("--save-best-by", type=str, default="success_rate",
                        choices=["success_rate", "return_mean", "height_gain_mean"])
    parser.add_argument("--no-radius-randomization", action="store_true")
    parser.add_argument("--no-friction-randomization", action="store_true")

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--hidden-depth", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--replay-size", type=int, default=500000)
    parser.add_argument("--init-alpha", type=float, default=0.2)
    parser.add_argument("--target-entropy-scale", type=float, default=1.0)
    parser.add_argument("--no-learnable-alpha", action="store_true")
    parser.add_argument("--no-obs-normalization", action="store_true")
    parser.add_argument("--obs-clip", type=float, default=10.0)
    return parser.parse_args()


def main():
    args = parse_args()
    train_cfg = TrainConfig(
        total_steps=args.total_steps,
        seed=args.seed,
        warmup_steps=args.warmup_steps,
        update_after=args.update_after,
        updates_per_step=args.updates_per_step,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        action_mode=args.action_mode,
        action_step=args.action_step,
        reward_type=args.reward_type,
        n_substeps=args.n_substeps,
        randomize_object_radius=not args.no_radius_randomization,
        randomize_object_friction=not args.no_friction_randomization,
        save_dir=args.save_dir,
        save_best_by=args.save_best_by,
    )
    sac_cfg = SACConfig(
        device=args.device,
        hidden_dim=args.hidden_dim,
        hidden_depth=args.hidden_depth,
        gamma=args.gamma,
        tau=args.tau,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        alpha_lr=args.alpha_lr,
        batch_size=args.batch_size,
        replay_size=args.replay_size,
        learnable_alpha=not args.no_learnable_alpha,
        init_alpha=args.init_alpha,
        target_entropy_scale=args.target_entropy_scale,
        normalize_obs=not args.no_obs_normalization,
        obs_clip=args.obs_clip,
    )

    best_stats = train(train_cfg, sac_cfg)
    print("best_stats", json.dumps(best_stats, ensure_ascii=True))


if __name__ == "__main__":
    main()
