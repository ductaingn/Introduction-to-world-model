from typing import Any

import numpy as np
import gymnasium as gym
import miniworld


class Env(gym.Env):
    """
    MiniWorld environment wrapper that pads image observations
    to make them suitable for a VAE.
    """

    metadata = {"render_modes": []}

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__()

        self._env: gym.Env = gym.make("MiniWorld-Sign-v0", render_mode=render_mode)

        # Expose action space
        self.action_space: gym.spaces.Discrete = self._env.action_space

        # Pad width: (H, W, C)
        self.pad_width = (
            (2, 2),  # H
            (0, 0),  # W
            (0, 0),  # C
        )

        # Original observation shape
        h, w, c = self._env.observation_space["obs"].shape
        pad_h = self.pad_width[0][0] + self.pad_width[0][1]
        pad_w = self.pad_width[1][0] + self.pad_width[1][1]

        padded_shape = (h + pad_h, w + pad_w, c)

        self.observation_space: gym.spaces.Box = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=padded_shape,
            dtype=np.float32,
        )

    def _pad_obs(self, obs: np.ndarray) -> np.ndarray:
        return np.pad(
            obs,
            pad_width=self.pad_width,
            mode="constant",
            constant_values=0,
        ).astype(np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        obs, info = self._env.reset(seed=seed, options=options)
        obs = self._pad_obs(obs["obs"].astype(np.float32)) / 255.0  # Normalize
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)

        obs = self._pad_obs(obs["obs"].astype(np.float32)) / 255.0  # Normalize

        return (
            obs,
            reward,
            terminated,
            truncated,
            info,
        )

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()


if __name__ == "__main__":
    env = Env(render_mode="human")
    obs, info = env.reset()
    for i in range(1000):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        env.render()
        print(obs.shape, reward, term, trunc)
