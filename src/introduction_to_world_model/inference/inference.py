import numpy as np

import gymnasium as gym
import miniworld

import matplotlib.pyplot as plt
from introduction_to_world_model.env.env import Env


if __name__ == "__main__":
    env = gym.make("MiniWorld-Sign-v0")
    env = Env()
    obs, _ = env.reset()

    for _ in range(10):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)

        plt.imshow(next_obs)
        plt.show(block=False)  # Show without blocking code execution
        plt.pause(0.1)  # Brief pause to allow the window to render

        print("Press [Enter] to see the next image...")
        input()  # Wait for user input
