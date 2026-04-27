"""Evaluate a saved SAC checkpoint for the three-finger grasp task."""

from __future__ import annotations

import argparse
import json

import torch

from grasp_rl_env import JackHandStateEnv
from sac_agent import SACAgent, SACConfig


def load_agent(checkpoint_path: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    agent_cfg = SACConfig(**ckpt["agent_cfg"])
    agent = SACAgent(int(ckpt["obs_dim"]), int(ckpt["act_dim"]), agent_cfg)
    agent.load_checkpoint_dict(ckpt)
    train_cfg = ckpt.get("train_cfg", {})
    return agent, train_cfg, ckpt.get("stats", {})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="jk_fkik/three_finger/grasp/checkpoints_sac/sac_best.pt")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    agent, train_cfg, saved_stats = load_agent(args.checkpoint)
    env = JackHandStateEnv(
        render_mode="human" if args.render else None,
        action_mode=train_cfg.get("action_mode", "delta"),
        action_step=float(train_cfg.get("action_step", 0.12)),
        reward_type=train_cfg.get("reward_type", "dense"),
        n_substeps=int(train_cfg.get("n_substeps", 20)),
        randomize_object_radius=bool(train_cfg.get("randomize_object_radius", True)),
        randomize_object_friction=bool(train_cfg.get("randomize_object_friction", True)),
    )

    returns = []
    successes = []
    heights = []
    contacts = []
    drifts = []

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        total_reward = 0.0
        last_info = {}
        while not done:
            action = agent.act(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            done = terminated or truncated
            last_info = info

        episode_stats = {
            "episode": ep,
            "return": float(total_reward),
            "is_success": bool(last_info.get("is_success", False)),
            "height_gain": float(last_info.get("height_gain", 0.0)),
            "n_contacts": int(last_info.get("n_contacts", 0)),
            "lateral_drift": float(last_info.get("lateral_drift", 0.0)),
            "thumb_contact": bool(last_info.get("thumb_contact", False)),
            "front_contact": bool(last_info.get("front_contact", False)),
        }
        print(json.dumps(episode_stats, ensure_ascii=True))

        returns.append(episode_stats["return"])
        successes.append(float(episode_stats["is_success"]))
        heights.append(episode_stats["height_gain"])
        contacts.append(float(episode_stats["n_contacts"]))
        drifts.append(episode_stats["lateral_drift"])

    env.close()
    summary = {
        "checkpoint": args.checkpoint,
        "saved_eval_stats": saved_stats,
        "return_mean": float(sum(returns) / max(len(returns), 1)),
        "success_rate": float(sum(successes) / max(len(successes), 1)),
        "height_gain_mean": float(sum(heights) / max(len(heights), 1)),
        "contacts_mean": float(sum(contacts) / max(len(contacts), 1)),
        "lateral_drift_mean": float(sum(drifts) / max(len(drifts), 1)),
    }
    print("summary", json.dumps(summary, ensure_ascii=True))


if __name__ == "__main__":
    main()
