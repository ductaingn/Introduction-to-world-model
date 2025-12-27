import attrs

import numpy as np

import torch

import gymnasium as gym

from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)
from .nn.policy_model import PolicyNetwork
from .nn.reasoning_model import MDNRNN
from .nn.vision_model import ConvVAE
from .utils.replay_buffer import ReplayBuffer


@attrs.define
class WorldModel:
    observation_space: gym.spaces.Dict
    action_space: gym.spaces.Discrete
    replay_buffer_size: int = 10000
    vision_model: ConvVAE = attrs.field(init=False)
    reasoning_model: MDNRNN = attrs.field(init=False)
    policy_model: PolicyNetwork = attrs.field(init=False)
    replay_buffer: ReplayBuffer = attrs.field(init=False)

    def __attrs_post_init__(self):
        self.vision_model = ConvVAE(self.observation_space)
        self.reasoning_model = MDNRNN(self.action_space, self.vision_model.latent_dim)
        self.policy_model = PolicyNetwork(
            self.vision_model.latent_dim,
            self.reasoning_model.hidden_size,
            self.action_space,
        )
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)

    def act(self, obs: torch.Tensor, h: torch.Tensor):
        z = self.vision_model.encode(obs)
        a = self.policy_model(z, h)

        return a

    def collect_data(self, env: gym.Env, n_step):
        obs, _ = env.reset()

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        ) as progress:
            step_task = progress.add_task("[green]Collecting data", total=n_step)

            for _ in range(n_step):
                action = env.action_space.sample()
                next_obs, reward, terminated, truncated, info = env.step(action)
                self.replay_buffer.add(
                    obs.astype(np.float32),
                    next_obs.astype(np.float32),
                    np.eye(int(self.action_space.n))[action],
                    np.array(reward).astype(np.float32),
                    np.array(terminated).astype(np.float32),
                    np.array(truncated).astype(np.float32),
                )
                obs = next_obs

                progress.update(step_task, advance=1)

    def save_checkpoint(
        self,
        path: str,
        *,
        vision_optimizer: torch.optim.Optimizer | None = None,
        reasoning_optimizer: torch.optim.Optimizer | None = None,
        policy_optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        checkpoint = {
            "vision_model": self.vision_model.state_dict(),
            "reasoning_model": self.reasoning_model.state_dict(),
            "policy_model": self.policy_model.state_dict(),
            "replay_buffer_size": self.replay_buffer_size,
            "replay_buffer": {
                "obs": list(self.replay_buffer.obs),
                "next_obs": list(self.replay_buffer.next_obs),
                "act": list(self.replay_buffer.act),
                "reward": list(self.replay_buffer.reward),
                "terminated": list(self.replay_buffer.terminated),
                "truncated": list(self.replay_buffer.truncated),
            },
        }

        if vision_optimizer is not None:
            checkpoint["vision_optimizer"] = vision_optimizer.state_dict()

        if reasoning_optimizer is not None:
            checkpoint["reasoning_optimizer"] = reasoning_optimizer.state_dict()

        if policy_optimizer is not None:
            checkpoint["policy_optimizer"] = policy_optimizer.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(
        self,
        path: str,
        *,
        vision_optimizer: torch.optim.Optimizer | None = None,
        reasoning_optimizer: torch.optim.Optimizer | None = None,
        policy_optimizer: torch.optim.Optimizer | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        checkpoint = torch.load(path, map_location=device)

        # ---- models ----
        self.vision_model.load_state_dict(checkpoint["vision_model"])
        self.reasoning_model.load_state_dict(checkpoint["reasoning_model"])
        self.policy_model.load_state_dict(checkpoint["policy_model"])

        # ---- replay buffer ----
        self.replay_buffer_size = checkpoint["replay_buffer_size"]
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)

        rb = checkpoint["replay_buffer"]
        for obs, next_obs, act, reward, terminated, truncated in zip(
            rb["obs"],
            rb["next_obs"],
            rb["act"],
            rb["reward"],
            rb["terminated"],
            rb["truncated"],
        ):
            self.replay_buffer.add(
                obs,
                next_obs,
                act,
                reward,
                terminated,
                truncated,
            )

        # ---- optimizers ----
        if vision_optimizer is not None and "vision_optimizer" in checkpoint:
            vision_optimizer.load_state_dict(checkpoint["vision_optimizer"])

        if reasoning_optimizer is not None and "reasoning_optimizer" in checkpoint:
            reasoning_optimizer.load_state_dict(checkpoint["reasoning_optimizer"])

        if policy_optimizer is not None and "policy_optimizer" in checkpoint:
            policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
