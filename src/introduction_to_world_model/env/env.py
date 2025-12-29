from typing import Any, Tuple

import numpy as np

import cv2

import gymnasium as gym
import miniworld


class Env(gym.Env):
    """
    MiniWorld environment wrapper that pads image observations
    to make them suitable for a VAE.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        img_shape: Tuple[int, int, int] = (64, 64, 3),
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        self._env: gym.Env = gym.make("MiniWorld-Sign-v0", render_mode=render_mode)

        # Expose action space
        self.action_space: gym.spaces.Discrete = self._env.action_space

        # H, W, C
        self.img_shape = img_shape

        self.observation_space: gym.spaces.Box = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=img_shape,
            dtype=np.float32,
        )

    def _resize_obs(self, obs: np.ndarray) -> np.ndarray:
        obs = cv2.resize(
            obs, (self.img_shape[1], self.img_shape[0]), interpolation=cv2.INTER_LINEAR
        )

        return obs

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        obs, info = self._env.reset(seed=seed, options=options)
        obs = (
            self._resize_obs(obs["obs"].astype(np.float32)) / 255.0
        )  # Normalize [0.0, 1.0]
        obs = obs * 2 - 1.0  # [-1.0, 1.0]
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)

        obs = (
            self._resize_obs(obs["obs"].astype(np.float32)) / 255.0
        )  # Normalize [0.0, 1.0]
        obs = obs * 2 - 1.0  # [-1.0, 1.0]

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
