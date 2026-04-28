"""Soft Actor-Critic components for the three-finger grasp task."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Normal
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyTorch is required for SAC training. Install it first, then rerun the SAC scripts."
    ) from exc


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


@dataclass
class SACConfig:
    device: str = "auto"
    hidden_dim: int = 256
    hidden_depth: int = 2
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    batch_size: int = 256
    replay_size: int = 500_000
    learnable_alpha: bool = False
    init_alpha: float = 0.05
    target_entropy_scale: float = 1.0
    normalize_obs: bool = True
    obs_clip: float = 10.0


class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, capacity: int):
        self.capacity = int(capacity)
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.as_tensor(self.obs[idx], device=device),
            "actions": torch.as_tensor(self.actions[idx], device=device),
            "rewards": torch.as_tensor(self.rewards[idx], device=device),
            "next_obs": torch.as_tensor(self.next_obs[idx], device=device),
            "dones": torch.as_tensor(self.dones[idx], device=device),
        }


class RunningNorm:
    def __init__(self, dim: int, clip: float = 10.0, eps: float = 1e-4):
        self.dim = int(dim)
        self.clip = float(clip)
        self.eps = float(eps)
        self.count = eps
        self.mean = np.zeros(dim, dtype=np.float64)
        self.var = np.ones(dim, dtype=np.float64)

    def update(self, batch: np.ndarray):
        batch = np.asarray(batch, dtype=np.float64)
        if batch.ndim == 1:
            batch = batch[None, :]
        batch_count = batch.shape[0]
        batch_mean = np.mean(batch, axis=0)
        batch_var = np.var(batch, axis=0)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + (delta ** 2) * self.count * batch_count / total_count
        new_var = m2 / total_count

        self.mean = new_mean
        self.var = np.maximum(new_var, 1e-6)
        self.count = total_count

    def normalize_np(self, obs: np.ndarray) -> np.ndarray:
        obs = (obs - self.mean) / np.sqrt(self.var + 1e-8)
        return np.clip(obs, -self.clip, self.clip)

    def normalize_tensor(self, obs: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, dtype=obs.dtype, device=obs.device)
        var = torch.as_tensor(self.var, dtype=obs.dtype, device=obs.device)
        obs = (obs - mean) / torch.sqrt(var + 1e-8)
        return torch.clamp(obs, -self.clip, self.clip)

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {
            "count": np.array([self.count], dtype=np.float64),
            "mean": self.mean.copy(),
            "var": self.var.copy(),
        }

    def load_state_dict(self, state: Dict[str, np.ndarray]):
        self.count = float(np.asarray(state["count"]).reshape(-1)[0])
        self.mean = np.asarray(state["mean"], dtype=np.float64).copy()
        self.var = np.asarray(state["var"], dtype=np.float64).copy()


def mlp(in_dim: int, out_dim: int, hidden_dim: int, hidden_depth: int) -> nn.Sequential:
    layers = []
    last_dim = in_dim
    for _ in range(hidden_depth):
        layers.extend([nn.Linear(last_dim, hidden_dim), nn.ReLU()])
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, out_dim))
    return nn.Sequential(*layers)


class SquashedGaussianActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int, hidden_depth: int):
        super().__init__()
        self.trunk = mlp(obs_dim, hidden_dim, hidden_dim, hidden_depth)
        self.mu_layer = nn.Linear(hidden_dim, act_dim)
        self.log_std_layer = nn.Linear(hidden_dim, act_dim)

    def forward(self, obs: torch.Tensor, deterministic: bool = False, with_logprob: bool = True):
        h = self.trunk(obs)
        mu = self.mu_layer(h)
        log_std = torch.clamp(self.log_std_layer(h), LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        dist = Normal(mu, std)

        pre_tanh = mu if deterministic else dist.rsample()
        action = torch.tanh(pre_tanh)
        log_prob = None
        if with_logprob:
            log_prob = dist.log_prob(pre_tanh).sum(dim=-1, keepdim=True)
            log_prob -= torch.log(1.0 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        return action, log_prob, mu


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int, hidden_depth: int):
        super().__init__()
        self.net = mlp(obs_dim + act_dim, 1, hidden_dim, hidden_depth)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, act], dim=-1))


class SACAgent:
    def __init__(self, obs_dim: int, act_dim: int, cfg: SACConfig):
        self.cfg = cfg
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = self._resolve_device(cfg.device)

        self.actor = SquashedGaussianActor(
            obs_dim, act_dim, cfg.hidden_dim, cfg.hidden_depth).to(self.device)
        self.q1 = QNetwork(obs_dim, act_dim, cfg.hidden_dim, cfg.hidden_depth).to(self.device)
        self.q2 = QNetwork(obs_dim, act_dim, cfg.hidden_dim, cfg.hidden_depth).to(self.device)
        self.q1_target = QNetwork(
            obs_dim, act_dim, cfg.hidden_dim, cfg.hidden_depth).to(self.device)
        self.q2_target = QNetwork(
            obs_dim, act_dim, cfg.hidden_dim, cfg.hidden_depth).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=cfg.critic_lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=cfg.critic_lr)

        self.target_entropy = -cfg.target_entropy_scale * act_dim
        self.log_alpha = torch.tensor(
            np.log(cfg.init_alpha), device=self.device, dtype=torch.float32, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)

        self.obs_norm = RunningNorm(obs_dim, clip=cfg.obs_clip)

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def update_obs_norm(self, *obs_batches: np.ndarray):
        if not self.cfg.normalize_obs:
            return
        for batch in obs_batches:
            self.obs_norm.update(batch)

    def _normalize_tensor(self, obs: torch.Tensor) -> torch.Tensor:
        if not self.cfg.normalize_obs:
            return obs
        return self.obs_norm.normalize_tensor(obs)

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        obs_t = self._normalize_tensor(obs_t)
        with torch.no_grad():
            action, _, _ = self.actor(obs_t, deterministic=deterministic, with_logprob=False)
        return action.squeeze(0).cpu().numpy().astype(np.float32)

    def update(self, replay: ReplayBuffer) -> Dict[str, float]:
        batch = replay.sample(self.cfg.batch_size, self.device)
        obs = self._normalize_tensor(batch["obs"])
        next_obs = self._normalize_tensor(batch["next_obs"])
        actions = batch["actions"]
        rewards = batch["rewards"]
        dones = batch["dones"]

        with torch.no_grad():
            next_actions, next_logp, _ = self.actor(next_obs, deterministic=False, with_logprob=True)
            target_q1 = self.q1_target(next_obs, next_actions)
            target_q2 = self.q2_target(next_obs, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha.detach() * next_logp
            backup = rewards + (1.0 - dones) * self.cfg.gamma * target_q

        q1_loss = F.mse_loss(self.q1(obs, actions), backup)
        q2_loss = F.mse_loss(self.q2(obs, actions), backup)

        self.q1_opt.zero_grad(set_to_none=True)
        q1_loss.backward()
        self.q1_opt.step()

        self.q2_opt.zero_grad(set_to_none=True)
        q2_loss.backward()
        self.q2_opt.step()

        new_actions, logp, _ = self.actor(obs, deterministic=False, with_logprob=True)
        q_pi = torch.min(self.q1(obs, new_actions), self.q2(obs, new_actions))
        actor_loss = (self.alpha.detach() * logp - q_pi).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss_value = 0.0
        if self.cfg.learnable_alpha:
            alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()
            alpha_loss_value = float(alpha_loss.item())

        self._soft_update(self.q1, self.q1_target, self.cfg.tau)
        self._soft_update(self.q2, self.q2_target, self.cfg.tau)

        return {
            "critic_loss": float((q1_loss + q2_loss).item() * 0.5),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(self.alpha.detach().item()),
            "alpha_loss": alpha_loss_value,
            "q1_mean": float(self.q1(obs, actions).mean().item()),
            "q2_mean": float(self.q2(obs, actions).mean().item()),
            "logp_mean": float(logp.mean().item()),
        }

    @staticmethod
    def _soft_update(src: nn.Module, dst: nn.Module, tau: float):
        for src_param, dst_param in zip(src.parameters(), dst.parameters()):
            dst_param.data.mul_(1.0 - tau)
            dst_param.data.add_(tau * src_param.data)

    def checkpoint_dict(self) -> Dict[str, object]:
        return {
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "agent_cfg": asdict(self.cfg),
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "q1_opt": self.q1_opt.state_dict(),
            "q2_opt": self.q2_opt.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha_opt": self.alpha_opt.state_dict(),
            "obs_norm": self.obs_norm.state_dict(),
        }

    def load_checkpoint_dict(self, ckpt: Dict[str, object]):
        self.actor.load_state_dict(ckpt["actor"])
        self.q1.load_state_dict(ckpt["q1"])
        self.q2.load_state_dict(ckpt["q2"])
        self.q1_target.load_state_dict(ckpt["q1_target"])
        self.q2_target.load_state_dict(ckpt["q2_target"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.q1_opt.load_state_dict(ckpt["q1_opt"])
        self.q2_opt.load_state_dict(ckpt["q2_opt"])
        self.log_alpha.data.copy_(ckpt["log_alpha"].to(self.device))
        self.alpha_opt.load_state_dict(ckpt["alpha_opt"])
        self.obs_norm.load_state_dict(ckpt["obs_norm"])
