"""Train a PyTorch Soft Actor-Critic agent for three-finger grasping."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from grasp_rl_env import JackHandStateEnv
from sac_agent import ReplayBuffer, SACAgent, SACConfig

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None


@dataclass
class TrainConfig:
    total_steps: int = 300_000
    seed: int = 0
    warmup_steps: int = 5_000
    update_after: int = 2_000
    updates_per_step: int = 1
    eval_interval: int = 10_000
    eval_episodes: int = 10
    object_type: str = "sphere"
    action_mode: str = "delta"
    wrist_translation_step: float = 0.015
    finger_action_step: float = 0.06
    lock_wrist_rotation: bool = True
    reward_type: str = "dense"
    n_substeps: int = 20
    randomize_object_radius: bool = False
    randomize_object_friction: bool = False
    save_dir: str = "jk_fkik/three_finger/grasp/checkpoints_sac"
    save_best_by: str = "success_rate"
    render: bool = False
    use_tensorboard: bool = True
    tensorboard_logdir: str = "jk_fkik/three_finger/grasp/tb_runs"
    run_name: str = ""
    use_success_alpha_schedule: bool = False
    success_rate_ema_beta: float = 0.8
    alpha_phase1_threshold: float = 0.03
    alpha_phase2_threshold: float = 0.10
    alpha_phase3_threshold: float = 0.20
    alpha_phase1_min: float = 0.01
    alpha_phase1_max: float = 0.06
    alpha_phase1_entropy: float = 0.8
    alpha_phase2_min: float = 0.005
    alpha_phase2_max: float = 0.03
    alpha_phase2_entropy: float = 0.45
    alpha_phase3_min: float = 0.002
    alpha_phase3_max: float = 0.015
    alpha_phase3_entropy: float = 0.2


def make_env(cfg: TrainConfig, render_mode=None):
    return JackHandStateEnv(
        render_mode=render_mode,
        object_type=cfg.object_type,
        action_mode=cfg.action_mode,
        wrist_translation_step=cfg.wrist_translation_step,
        finger_action_step=cfg.finger_action_step,
        lock_wrist_rotation=cfg.lock_wrist_rotation,
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
    spin_rates = []

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
        contacts.append(float(last_info.get("max_contacts", last_info.get("n_contacts", 0))))
        drifts.append(float(last_info.get("lateral_drift", 0.0)))
        spin_rates.append(float(last_info.get("max_object_yaw_rate", last_info.get("object_yaw_rate", 0.0))))

    env.close()
    return {
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "success_rate": float(np.mean(successes)),
        "height_gain_mean": float(np.mean(height_gains)),
        "contacts_mean": float(np.mean(contacts)),
        "lateral_drift_mean": float(np.mean(drifts)),
        "spin_rate_mean": float(np.mean(spin_rates)),
    }


def save_checkpoint(path: str, agent: SACAgent, train_cfg: TrainConfig, extra_stats: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = agent.checkpoint_dict()
    ckpt["train_cfg"] = asdict(train_cfg)
    ckpt["stats"] = extra_stats
    torch.save(ckpt, path)


def make_run_name(cfg: TrainConfig) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    if cfg.run_name:
        return f"{cfg.run_name}_{ts}"
    return f"sac_{cfg.object_type}_{cfg.action_mode}_{ts}"


def create_summary_writer(cfg: TrainConfig):
    if not cfg.use_tensorboard:
        return None, None
    if SummaryWriter is None:
        print(
            json.dumps(
                {
                    "iter_type": "warning",
                    "message": "TensorBoard writer unavailable; install tensorboard to enable the dashboard.",
                },
                ensure_ascii=True,
            )
        )
        return None, None
    run_name = cfg.run_name if cfg.run_name else make_run_name(cfg)
    log_dir = os.path.join(cfg.tensorboard_logdir, run_name)
    writer = SummaryWriter(log_dir=log_dir)
    return writer, log_dir


def writer_add_scalar_dict(writer, prefix: str, stats: dict, step: int):
    for key, value in stats.items():
        if isinstance(value, (int, float, np.floating, np.integer, bool)):
            writer.add_scalar(f"{prefix}/{key}", float(value), step)


def get_success_alpha_phase(train_cfg: TrainConfig, success_rate_ema: float, current_phase: int) -> int:
    phase = current_phase
    if phase < 1 and success_rate_ema >= train_cfg.alpha_phase1_threshold:
        phase = 1
    if phase < 2 and success_rate_ema >= train_cfg.alpha_phase2_threshold:
        phase = 2
    if phase < 3 and success_rate_ema >= train_cfg.alpha_phase3_threshold:
        phase = 3
    return phase


def apply_success_alpha_phase(agent: SACAgent, train_cfg: TrainConfig, sac_cfg: SACConfig, phase: int):
    if phase <= 0:
        agent.set_alpha_schedule_phase(
            learnable_alpha=False,
            target_entropy_scale=sac_cfg.target_entropy_scale,
            alpha_min=sac_cfg.init_alpha,
            alpha_max=sac_cfg.init_alpha,
            fixed_alpha=sac_cfg.init_alpha,
        )
    elif phase == 1:
        agent.set_alpha_schedule_phase(
            learnable_alpha=True,
            target_entropy_scale=train_cfg.alpha_phase1_entropy,
            alpha_min=train_cfg.alpha_phase1_min,
            alpha_max=train_cfg.alpha_phase1_max,
        )
    elif phase == 2:
        agent.set_alpha_schedule_phase(
            learnable_alpha=True,
            target_entropy_scale=train_cfg.alpha_phase2_entropy,
            alpha_min=train_cfg.alpha_phase2_min,
            alpha_max=train_cfg.alpha_phase2_max,
        )
    else:
        agent.set_alpha_schedule_phase(
            learnable_alpha=True,
            target_entropy_scale=train_cfg.alpha_phase3_entropy,
            alpha_min=train_cfg.alpha_phase3_min,
            alpha_max=train_cfg.alpha_phase3_max,
        )


def train(train_cfg: TrainConfig, sac_cfg: SACConfig):
    set_seed(train_cfg.seed)
    train_render = "human" if train_cfg.render else None
    env = make_env(train_cfg, render_mode=train_render)

    obs, _ = env.reset(seed=train_cfg.seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = SACAgent(obs_dim, act_dim, sac_cfg)
    replay = ReplayBuffer(obs_dim, act_dim, sac_cfg.replay_size)
    
    # 纭繚姣忎釜 run 鏈夌嫭绔嬬殑 logdir 鍜?save_dir
    run_name = make_run_name(train_cfg)
    train_cfg.run_name = run_name
    train_cfg.save_dir = os.path.join(train_cfg.save_dir, run_name)
    
    writer, tb_log_dir = create_summary_writer(train_cfg)

    if sac_cfg.normalize_obs:
        agent.update_obs_norm(obs)

    if writer is not None:
        writer.add_text("config/train_cfg", json.dumps(asdict(train_cfg), ensure_ascii=True, indent=2))
        writer.add_text("config/agent_cfg", json.dumps(asdict(sac_cfg), ensure_ascii=True, indent=2))
        writer.add_text("run/info", f"log_dir={tb_log_dir}")
        print(json.dumps({"iter_type": "tensorboard", "log_dir": tb_log_dir}, ensure_ascii=True))

    best_score = -float("inf")
    best_stats = {}
    episode_return = 0.0
    episode_len = 0
    episode_idx = 0
    latest_update = {}
    success_rate_ema = 0.0
    alpha_phase = 0

    if train_cfg.use_success_alpha_schedule:
        apply_success_alpha_phase(agent, train_cfg, sac_cfg, alpha_phase)

    for step in range(1, train_cfg.total_steps + 1):
        if step <= train_cfg.warmup_steps:
            action = env.action_space.sample().astype(np.float32)
        else:
            action = agent.act(obs, deterministic=False)

        next_obs, reward, terminated, truncated, info = env.step(action)
        if train_cfg.render:
            env.render()
            time.sleep(0.04)
        done = bool(terminated or truncated)
        replay.add(obs, action, reward, next_obs, float(done))
        if sac_cfg.normalize_obs:
            agent.update_obs_norm(obs, next_obs)

        obs = next_obs
        episode_return += float(reward)
        episode_len += 1

        if step >= train_cfg.update_after and replay.size >= sac_cfg.batch_size:
            for _ in range(train_cfg.updates_per_step):
                latest_update = agent.update(replay)
            if writer is not None and latest_update and step % 50 == 0:
                writer_add_scalar_dict(writer, "update", latest_update, step)

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
                "max_object_yaw_rate": float(
                    info.get("max_object_yaw_rate", info.get("object_yaw_rate", 0.0))
                ),
            }
            if latest_update:
                episode_stats.update({
                    "actor_loss": latest_update["actor_loss"],
                    "critic_loss": latest_update["critic_loss"],
                    "alpha": latest_update["alpha"],
                })
            print(json.dumps(episode_stats, ensure_ascii=True))
            if writer is not None:
                writer_add_scalar_dict(writer, "train_episode", episode_stats, step)
            obs, _ = env.reset(seed=train_cfg.seed + episode_idx)
            episode_return = 0.0
            episode_len = 0
            if sac_cfg.normalize_obs:
                agent.update_obs_norm(obs)

        if step % train_cfg.eval_interval == 0:
            eval_stats = evaluate(agent, train_cfg, train_cfg.seed + step * 13)
            if train_cfg.use_success_alpha_schedule:
                success_rate_ema = (
                    train_cfg.success_rate_ema_beta * success_rate_ema
                    + (1.0 - train_cfg.success_rate_ema_beta) * eval_stats["success_rate"]
                )
                new_phase = get_success_alpha_phase(train_cfg, success_rate_ema, alpha_phase)
                if new_phase != alpha_phase:
                    alpha_phase = new_phase
                    apply_success_alpha_phase(agent, train_cfg, sac_cfg, alpha_phase)
            stats = {
                "iter_type": "eval",
                "step": step,
                "replay_size": int(replay.size),
                "success_rate_ema": float(success_rate_ema),
                "alpha_phase": int(alpha_phase),
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
            if writer is not None:
                writer_add_scalar_dict(writer, "eval", stats, step)

            latest_path = os.path.join(train_cfg.save_dir, "sac_latest.pt")
            save_checkpoint(latest_path, agent, train_cfg, stats)

            score = eval_stats[train_cfg.save_best_by]
            if score >= best_score:
                best_score = score
                best_stats = stats
                best_path = os.path.join(train_cfg.save_dir, "sac_best.pt")
                save_checkpoint(best_path, agent, train_cfg, stats)

    env.close()
    if writer is not None:
        writer.close()
    return best_stats


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="", help="Path to JSON config file")
    parser.add_argument("--total-steps", type=int, default=300000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=5000)
    parser.add_argument("--update-after", type=int, default=2000)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--eval-interval", type=int, default=10000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument(
        "--object-type",
        type=str,
        default="sphere",
        choices=["sphere", "sphere_small", "cylinder", "box", "box_large"],
    )
    parser.add_argument("--action-mode", type=str, default="delta", choices=["delta", "absolute"])
    parser.add_argument("--wrist-translation-step", type=float, default=0.015)
    parser.add_argument("--finger-action-step", type=float, default=0.06)
    parser.add_argument("--lock-wrist-rotation", dest="lock_wrist_rotation", action="store_true")
    parser.add_argument("--free-wrist-rotation", dest="lock_wrist_rotation", action="store_false")
    parser.add_argument("--reward-type", type=str, default="dense", choices=["dense", "sparse"])
    parser.add_argument("--n-substeps", type=int, default=20)
    parser.add_argument("--save-dir", type=str,
                        default="jk_fkik/three_finger/grasp/checkpoints_sac")
    parser.add_argument("--save-best-by", type=str, default="success_rate",
                        choices=["success_rate", "return_mean", "height_gain_mean"])
    parser.add_argument("--radius-randomization", dest="radius_randomization",
                        action="store_true")
    parser.add_argument("--no-radius-randomization", dest="radius_randomization",
                        action="store_false")
    parser.add_argument("--friction-randomization", dest="friction_randomization",
                        action="store_true")
    parser.add_argument("--no-friction-randomization", dest="friction_randomization",
                        action="store_false")
    parser.add_argument("--render", action="store_true", help="Open MuJoCo viewer during training")
    parser.add_argument("--tensorboard-logdir", type=str, default="jk_fkik/three_finger/grasp/tb_runs")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--use-success-alpha-schedule", dest="use_success_alpha_schedule", action="store_true")
    parser.add_argument("--no-success-alpha-schedule", dest="use_success_alpha_schedule", action="store_false")
    parser.add_argument("--success-rate-ema-beta", type=float, default=0.8)
    parser.add_argument("--alpha-phase1-threshold", type=float, default=0.03)
    parser.add_argument("--alpha-phase2-threshold", type=float, default=0.10)
    parser.add_argument("--alpha-phase3-threshold", type=float, default=0.20)
    parser.add_argument("--alpha-phase1-min", type=float, default=0.01)
    parser.add_argument("--alpha-phase1-max", type=float, default=0.06)
    parser.add_argument("--alpha-phase1-entropy", type=float, default=0.8)
    parser.add_argument("--alpha-phase2-min", type=float, default=0.005)
    parser.add_argument("--alpha-phase2-max", type=float, default=0.03)
    parser.add_argument("--alpha-phase2-entropy", type=float, default=0.45)
    parser.add_argument("--alpha-phase3-min", type=float, default=0.002)
    parser.add_argument("--alpha-phase3-max", type=float, default=0.015)
    parser.add_argument("--alpha-phase3-entropy", type=float, default=0.2)
    parser.set_defaults(
        radius_randomization=False,
        friction_randomization=False,
        lock_wrist_rotation=True,
        use_success_alpha_schedule=False,
    )

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--hidden-depth", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--critic-grad-clip", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--replay-size", type=int, default=500000)
    parser.add_argument("--init-alpha", type=float, default=0.05)
    parser.add_argument("--alpha-min", type=float, default=0.002)
    parser.add_argument("--alpha-max", type=float, default=0.1)
    parser.add_argument("--target-entropy-scale", type=float, default=1.0)
    parser.add_argument("--learnable-alpha", dest="learnable_alpha", action="store_true")
    parser.add_argument("--fixed-alpha", dest="learnable_alpha", action="store_false")
    parser.set_defaults(learnable_alpha=False)
    parser.add_argument("--no-obs-normalization", action="store_true")
    parser.add_argument("--obs-clip", type=float, default=10.0)
    parser.add_argument("--auto-start-tensorboard", action="store_true", help="Automatically launch tensorboard in background")
    
    args, remaining_argv = parser.parse_known_args()
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            config_defaults = json.load(f)
        parser.set_defaults(**config_defaults)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.auto_start_tensorboard and not args.no_tensorboard:
        import subprocess
        import atexit
        try:
            tb_process = subprocess.Popen(
                [sys.executable, "-m", "tensorboard.main", "--logdir", args.tensorboard_logdir, "--port", "6006"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print("TensorBoard automatically started at http://localhost:6006")
            atexit.register(lambda: tb_process.kill())
        except Exception as e:
            print(f"Failed to start TensorBoard automatically: {e}")

    train_cfg = TrainConfig(
        total_steps=args.total_steps,
        seed=args.seed,
        warmup_steps=args.warmup_steps,
        update_after=args.update_after,
        updates_per_step=args.updates_per_step,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        object_type=args.object_type,
        action_mode=args.action_mode,
        wrist_translation_step=args.wrist_translation_step,
        finger_action_step=args.finger_action_step,
        lock_wrist_rotation=args.lock_wrist_rotation,
        reward_type=args.reward_type,
        n_substeps=args.n_substeps,
        randomize_object_radius=args.radius_randomization,
        randomize_object_friction=args.friction_randomization,
        save_dir=args.save_dir,
        save_best_by=args.save_best_by,
        render=args.render,
        use_tensorboard=not args.no_tensorboard,
        tensorboard_logdir=args.tensorboard_logdir,
        run_name=args.run_name,
        use_success_alpha_schedule=args.use_success_alpha_schedule,
        success_rate_ema_beta=args.success_rate_ema_beta,
        alpha_phase1_threshold=args.alpha_phase1_threshold,
        alpha_phase2_threshold=args.alpha_phase2_threshold,
        alpha_phase3_threshold=args.alpha_phase3_threshold,
        alpha_phase1_min=args.alpha_phase1_min,
        alpha_phase1_max=args.alpha_phase1_max,
        alpha_phase1_entropy=args.alpha_phase1_entropy,
        alpha_phase2_min=args.alpha_phase2_min,
        alpha_phase2_max=args.alpha_phase2_max,
        alpha_phase2_entropy=args.alpha_phase2_entropy,
        alpha_phase3_min=args.alpha_phase3_min,
        alpha_phase3_max=args.alpha_phase3_max,
        alpha_phase3_entropy=args.alpha_phase3_entropy,
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
        learnable_alpha=args.learnable_alpha,
        init_alpha=args.init_alpha,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        target_entropy_scale=args.target_entropy_scale,
        critic_grad_clip=args.critic_grad_clip,
        normalize_obs=not args.no_obs_normalization,
        obs_clip=args.obs_clip,
    )

    best_stats = train(train_cfg, sac_cfg)
    print("best_stats", json.dumps(best_stats, ensure_ascii=True))


if __name__ == "__main__":
    main()

