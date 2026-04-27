"""Evaluate a saved RL policy checkpoint."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from grasp_rl_env import JackHandStateEnv
from train_grasp_rl import LinearPolicy, RunningNormalizer, rollout


def load_checkpoint(path: str):
    data = np.load(path, allow_pickle=True)
    cfg = json.loads(str(data["config_json"][0]))
    weights = data["weights"]
    policy = LinearPolicy(weights.shape[1] - 1, weights.shape[0])
    policy.weights[:] = weights

    normalizer = RunningNormalizer(weights.shape[1] - 1)
    normalizer.mean[:] = data["obs_mean"]
    normalizer.m2[:] = data["obs_m2"]
    normalizer.count = float(data["obs_count"][0])

    return policy, normalizer, cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="jk_fkik/three_finger/grasp/checkpoints/ars_best.npz")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    policy, normalizer, cfg = load_checkpoint(args.checkpoint)
    env = JackHandStateEnv(
        render_mode="human" if args.render else None,
        action_mode=cfg.get("action_mode", "delta"),
        action_step=float(cfg.get("action_step", 0.12)),
        reward_type=cfg.get("reward_type", "dense"),
        n_substeps=int(cfg.get("n_substeps", 20)),
        randomize_object_radius=bool(cfg.get("randomize_object_radius", True)),
        randomize_object_friction=bool(cfg.get("randomize_object_friction", True)),
    )

    returns = []
    successes = []
    height_gains = []
    contacts = []

    for ep in range(args.episodes):
        ret, steps, info = rollout(
            env, policy, normalizer, seed=args.seed + ep, update_normalizer=False)
        returns.append(ret)
        successes.append(float(info.get("is_success", False)))
        height_gains.append(float(info.get("height_gain", 0.0)))
        contacts.append(float(info.get("n_contacts", 0)))
        print(json.dumps({
            "episode": ep,
            "return": float(ret),
            "steps": int(steps),
            "is_success": bool(info.get("is_success", False)),
            "height_gain": float(info.get("height_gain", 0.0)),
            "n_contacts": int(info.get("n_contacts", 0)),
            "lateral_drift": float(info.get("lateral_drift", 0.0)),
            "thumb_contact": bool(info.get("thumb_contact", False)),
            "front_contact": bool(info.get("front_contact", False)),
        }, ensure_ascii=True))

    env.close()
    summary = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "episodes": args.episodes,
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "success_rate": float(np.mean(successes)),
        "height_gain_mean": float(np.mean(height_gains)),
        "contacts_mean": float(np.mean(contacts)),
    }
    print("summary", json.dumps(summary, ensure_ascii=True))


if __name__ == "__main__":
    main()
