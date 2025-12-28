from collections import deque
from typing import Deque, Tuple
import attrs

import torch

import numpy as np


@attrs.define
class ReplayBuffer:
    buffer_size: int = 10000

    obs: Deque[np.ndarray] = attrs.field(init=False)
    next_obs: Deque[np.ndarray] = attrs.field(init=False)
    act: Deque[np.ndarray] = attrs.field(init=False)
    reward: Deque[np.ndarray] = attrs.field(init=False)
    terminated: Deque[np.ndarray] = attrs.field(init=False)
    truncated: Deque[np.ndarray] = attrs.field(init=False)

    def __attrs_post_init__(self) -> None:
        self.obs = deque(maxlen=self.buffer_size)
        self.next_obs = deque(maxlen=self.buffer_size)
        self.act = deque(maxlen=self.buffer_size)
        self.reward = deque(maxlen=self.buffer_size)
        self.terminated = deque(maxlen=self.buffer_size)
        self.truncated = deque(maxlen=self.buffer_size)

    def __len__(self):
        return len(self.obs)

    def add(
        self,
        obs: np.ndarray | torch.Tensor,
        next_obs: np.ndarray | torch.Tensor,
        act: np.ndarray | torch.Tensor,
        reward: np.ndarray | torch.Tensor,
        terminated: np.ndarray | torch.Tensor,
        truncated: np.ndarray | torch.Tensor,
    ):
        if isinstance(obs, torch.Tensor):
            obs = obs.detach().numpy()
        self.obs.append(obs)

        if isinstance(next_obs, torch.Tensor):
            next_obs = next_obs.detach().numpy()
        self.next_obs.append(next_obs)

        if isinstance(act, torch.Tensor):
            act = act.detach().numpy()
        self.act.append(act)

        if isinstance(reward, torch.Tensor):
            reward = reward.detach().numpy()
        self.reward.append(reward)

        if isinstance(terminated, torch.Tensor):
            terminated = terminated.detach().numpy()
        self.terminated.append(terminated)

        if isinstance(truncated, torch.Tensor):
            truncated = truncated.detach().numpy()
        self.truncated.append(truncated)

    def get_batch(
        self, batch_indices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if len(self) <= batch_indices.max():
            raise RuntimeError("Batch indices is greater than length of buffer!")

        return (
            np.array(self.obs)[batch_indices],
            np.array(self.next_obs)[batch_indices],
            np.array(self.act)[batch_indices],
            np.array(self.reward)[batch_indices],
            np.array(self.terminated)[batch_indices],
            np.array(self.truncated)[batch_indices],
        )

    def get_rollout_batch(
        self,
        batch_indices: np.ndarray,
        time_length: int,
        next_z: np.ndarray,
        z_indices: np.ndarray,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """
        Parameters:
        next_z: np.ndarray
            The latent vector of the next observation comes from the Vision Model, corresponding to the batch_indices
        """
        max_index = batch_indices.max() + time_length
        if len(self) <= max_index:
            raise RuntimeError("Batch indices is greater than length of buffer!")

        # Shape: (B, T)
        time_offsets = np.arange(time_length)
        rollout_indices = batch_indices[:, None] + time_offsets[None, :]
        z_rollout_indices = z_indices[:, None] + time_offsets[None, :]

        # Convert buffers once
        obs = np.asarray(self.obs)
        next_obs = np.asarray(self.next_obs)
        act = np.asarray(self.act)
        reward = np.asarray(self.reward)
        terminated = np.asarray(self.terminated)
        truncated = np.asarray(self.truncated)

        return (
            obs[rollout_indices],
            next_obs[rollout_indices],
            act[rollout_indices],
            reward[rollout_indices],
            terminated[rollout_indices],
            truncated[rollout_indices],
            next_z[z_rollout_indices],
        )

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if len(self.obs) < 1:
            raise RuntimeError("Buffer is empty!")

        batch_indices = np.random.randint(low=0, high=len(self.obs), size=batch_size)

        return self.get_batch(batch_indices)


if __name__ == "__main__":
    replay_buffer = ReplayBuffer(3)
    for i in range(5):
        obs = next_obs = act = reward = terminated = truncated = np.array([i])
        replay_buffer.add(obs, next_obs, act, reward, terminated, truncated)

        print(replay_buffer.obs)

        samples = replay_buffer.sample(5)
        print(f"samples: \nobs: {samples[0]}\next_obs: {samples[1]}\nact: {samples[2]}")
