"""Evaluate a saved SAC checkpoint for the three-finger grasp task."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np
import torch

from grasp_rl_env import JackHandStateEnv
from sac_agent import SACAgent, SACConfig


def load_agent(checkpoint_path: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    agent_cfg = SACConfig(**ckpt["agent_cfg"])
    agent = SACAgent(int(ckpt["obs_dim"]), int(ckpt["act_dim"]), agent_cfg)
    agent.load_checkpoint_dict(ckpt)
    train_cfg = ckpt.get("train_cfg", {})
    return agent, train_cfg, ckpt.get("stats", {}), int(ckpt["act_dim"])


def make_free_camera():
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = np.array([-0.02, 0.08, 0.26], dtype=np.float32)
    cam.distance = 0.35
    cam.azimuth = 135
    cam.elevation = -25
    return cam


def render_free_camera(env: JackHandStateEnv, renderer, camera):
    renderer.update_scene(env.base_env.data, camera=camera)
    return renderer.render().copy()


def write_frame(writer, frame_rgb: np.ndarray, repeats: int = 1):
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    for _ in range(max(1, repeats)):
        writer.write(frame_bgr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="jk_fkik/three_finger/grasp/checkpoints_sac/sac_best.pt")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--record-dir", type=str, default="")
    parser.add_argument("--record-width", type=int, default=640)
    parser.add_argument("--record-height", type=int, default=480)
    parser.add_argument("--record-fps", type=int, default=20)
    parser.add_argument("--slowdown", type=float, default=1.0)
    parser.add_argument("--start-hold", type=float, default=0.5)
    parser.add_argument("--end-hold", type=float, default=1.0)
    args = parser.parse_args()

    agent, train_cfg, saved_stats, act_dim = load_agent(args.checkpoint)
    lock_wrist_rotation = bool(
        train_cfg.get("lock_wrist_rotation", act_dim == 13)
    )
    env = JackHandStateEnv(
        render_mode="human" if args.render else None,
        object_type=train_cfg.get("object_type", "sphere"),
        action_mode=train_cfg.get("action_mode", "delta"),
        wrist_translation_step=float(train_cfg.get("wrist_translation_step", 0.015)),
        finger_action_step=float(train_cfg.get("finger_action_step", 0.06)),
        lock_wrist_rotation=lock_wrist_rotation,
        reward_type=train_cfg.get("reward_type", "dense"),
        n_substeps=int(train_cfg.get("n_substeps", 20)),
        randomize_object_radius=bool(train_cfg.get("randomize_object_radius", False)),
        randomize_object_friction=bool(train_cfg.get("randomize_object_friction", False)),
    )
    step_dt = env.base_env.model.opt.timestep * env.base_env.n_substeps
    render_sleep = max(0.0, step_dt * (args.slowdown - 1.0))

    record_dir = Path(args.record_dir) if args.record_dir else None
    if record_dir is not None:
        record_dir.mkdir(parents=True, exist_ok=True)
        record_renderer = mujoco.Renderer(
            env.base_env.model, args.record_height, args.record_width)
        record_camera = make_free_camera()
        frames_per_step = max(
            1, int(round(args.record_fps * step_dt * args.slowdown)))
        start_hold_frames = max(1, int(round(args.record_fps * args.start_hold)))
        end_hold_frames = max(1, int(round(args.record_fps * args.end_hold)))
    else:
        record_renderer = None
        record_camera = None
        frames_per_step = 1
        start_hold_frames = 1
        end_hold_frames = 1

    returns = []
    successes = []
    heights = []
    contacts = []
    drifts = []

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        if args.render:
            env.render()
            if render_sleep > 0.0:
                time.sleep(render_sleep)
        if record_dir is not None:
            video_path = record_dir / f"episode_{ep:02d}.mp4"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                args.record_fps,
                (args.record_width, args.record_height),
            )
            first_frame = render_free_camera(env, record_renderer, record_camera)
            write_frame(writer, first_frame, start_hold_frames)
        else:
            writer = None
        done = False
        total_reward = 0.0
        last_info = {}
        while not done:
            action = agent.act(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if args.render:
                env.render()
                if render_sleep > 0.0:
                    time.sleep(render_sleep)
            if writer is not None:
                frame = render_free_camera(env, record_renderer, record_camera)
                write_frame(writer, frame, frames_per_step)
            total_reward += float(reward)
            done = terminated or truncated
            last_info = info

        if writer is not None:
            final_frame = render_free_camera(env, record_renderer, record_camera)
            write_frame(writer, final_frame, end_hold_frames)
            writer.release()

        episode_stats = {
            "episode": ep,
            "return": float(total_reward),
            "is_success": bool(last_info.get("is_success", False)),
            "height_gain": float(last_info.get("height_gain", 0.0)),
            "n_contacts": int(last_info.get("max_contacts", last_info.get("n_contacts", 0))),
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

    if record_renderer is not None:
        record_renderer.close()
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
