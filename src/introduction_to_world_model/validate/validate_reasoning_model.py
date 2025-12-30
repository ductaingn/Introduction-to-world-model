"""
Play in dream
"""

import numpy as np

import torch

import gymnasium as gym

import matplotlib.pyplot as plt

from introduction_to_world_model.model.world_model import WorldModel
from introduction_to_world_model.env.env import Env
from introduction_to_world_model.validate import ValidateMode


def validate_vision_model(
    agent: WorldModel,
    n_steps: int,
    mode: ValidateMode = ValidateMode.ID,
    env: gym.Env | None = None,
    device: str = "cpu",
):
    agent.vision_model.eval()
    agent.reasoning_model.eval()
    agent.policy_model.eval()
    agent.to(device)

    obs, _ = env.reset()
    dream_obs = obs.copy()
    h = None
    for step in range(n_steps):
        a = env.action_space.sample()
        a_torch = torch.tensor(a).to(device)
        a_torch = (
            torch.nn.functional.one_hot(a_torch, num_classes=env.action_space.n)
            .view(1, 1, -1)
            .to(torch.float32)
        )
        with torch.no_grad():
            mu, log_std = agent.vision_model.encode(
                torch.tensor(obs, device=device).unsqueeze(0)
            )
            z = agent.vision_model.reparameterize(mu, log_std).unsqueeze(0)
            mu, log_std, log_weights, h, _, _ = agent.reasoning_model.forward(
                z, a_torch
            )
            dream_z = agent.reasoning_model.predict_next_z(mu, log_std, log_weights)
            dream_obs = agent.vision_model.decode(dream_z)

            true_obs, _, _, _, _ = env.step(a)
            print(
                f"Dream-truth diff: {torch.nn.functional.mse_loss(dream_obs, torch.tensor(true_obs).to(device)).cpu().numpy():.3f}"
            )

        dream_obs = (dream_obs.squeeze(0).cpu().numpy() + 1.0) / 2.0
        true_obs = (true_obs + 1.0) / 2.0

        # Convert tensors to numpy for plotting
        # Create the figure
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(true_obs)
        axes[0].set_title(f"Ground truth observation ({mode.value})")
        axes[0].axis("off")

        axes[1].imshow(dream_obs)
        axes[1].set_title("Dream observation")
        axes[1].axis("off")

        plt.show(block=False)  # Show without blocking code execution
        plt.pause(1)  # Brief pause to allow the window to render

        plt.close(fig)  # Close the window before the next iteration


if __name__ == "__main__":
    env = Env(render_mode="rgb_array")

    agent = WorldModel(env.observation_space, env.action_space)
    agent.load_checkpoint("checkpoint/trained_reasoning_model.pt", device="cuda:0")

    validate_vision_model(agent, 1000, ValidateMode.ID, env=env, device="cuda:0")
