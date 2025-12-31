from pathlib import Path

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
    observation_space: gym.spaces.Box
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

    def to(self, device: str = "cpu"):
        self.vision_model.to(device)
        self.reasoning_model.to(device)
        self.policy_model.to(device)

    def train(self, train: bool = True):
        self.vision_model.train(train)
        self.reasoning_model.train(train)
        self.policy_model.train(train)

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
                    np.eye(int(self.action_space.n))[action].astype(
                        np.float32
                    ),  # One-hot encoding
                    np.array(reward).astype(np.float32),
                    np.array(terminated).astype(np.float32),
                    np.array(truncated).astype(np.float32),
                )
                obs = next_obs

                progress.update(step_task, advance=1)

    def save_checkpoint(
        self,
        path: str | Path,
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

        try:
            print(f"Saving model to {path}")
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, save_path)
            print(f"Model saved to {path}")
        except Exception as e:
            print("Error occurred while saving model!")
            print(e)

    def load_checkpoint(
        self,
        path: str,
        *,
        load_vision_model: bool = True,
        load_reasoning_model: bool = True,
        load_policy_model: bool = True,
        vision_optimizer: torch.optim.Optimizer | None = None,
        reasoning_optimizer: torch.optim.Optimizer | None = None,
        policy_optimizer: torch.optim.Optimizer | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        try:
            print(f"Loading model from {path}")
            checkpoint = torch.load(path, weights_only=False, map_location=device)
        except Exception as e:
            print("Error occurred while loading model!")
            print(e)

            return

        # ---- models ----
        if load_vision_model:
            print("Loading vision model...")
            self.vision_model.load_state_dict(checkpoint["vision_model"])
            print("Vision model loaded!")
        if load_reasoning_model:
            print("Loading reasoning model...")
            self.reasoning_model.load_state_dict(checkpoint["reasoning_model"])
            print("Reasoning model loaded!")
        if load_policy_model:
            print("Loading policy model...")
            self.policy_model.load_state_dict(checkpoint["policy_model"])
            print("Policy model loaded!")

        # ---- replay buffer ----
        self.replay_buffer_size = checkpoint["replay_buffer_size"]
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)

        print("Loading replay buffer...")
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
        print(f"Replay buffer loaded!\nBuffer length: {len(self.replay_buffer)}")

        # ---- optimizers ----
        print("Loading optimizers...")
        if vision_optimizer is not None and "vision_optimizer" in checkpoint:
            vision_optimizer.load_state_dict(checkpoint["vision_optimizer"])
            print("Vision Model Optimizer loaded!")

        if reasoning_optimizer is not None and "reasoning_optimizer" in checkpoint:
            reasoning_optimizer.load_state_dict(checkpoint["reasoning_optimizer"])
            print("Reasoning Model Optimizer loaded!")

        if policy_optimizer is not None and "policy_optimizer" in checkpoint:
            policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
            print("Policy Model Optimizer loaded!")