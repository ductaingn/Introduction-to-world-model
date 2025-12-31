"""
Play in dream
"""
from collections import deque
from typing import Deque

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
    h = agent.reasoning_model.get_initial_state(batch=False)
    z_by_time: Deque[torch.Tensor] = deque(maxlen=agent.reasoning_model.rollout_time_length)
    a_by_time: Deque[torch.Tensor] = deque(maxlen=agent.reasoning_model.rollout_time_length)
    h_by_time: Deque[torch.Tensor] = deque(maxlen=agent.reasoning_model.rollout_time_length)
    
    # Play in dream
    for step in range(n_steps):
        a = env.action_space.sample()
        a_torch = torch.tensor(a).to(device)
        a_torch = (
            torch.nn.functional.one_hot(a_torch, num_classes=env.action_space.n)
            .view(1, 1, -1)
            .to(torch.float32)
        )
        a_by_time.append(a_torch)
        h_by_time.append(h)
        with torch.no_grad():
            # Get latent from Vision Model's encoder
            mu, log_std = agent.vision_model.encode(
                torch.tensor(dream_obs, device=device).unsqueeze(0)
            )
            z = agent.vision_model.reparameterize(mu, log_std)
            z_by_time.append(z)

            # MDN-RNN inference
            z_accumulate = torch.tensor(z_by_time).to(device).unsqueeze(0)
            a_accumulate = torch.tensor(a_by_time).to(device).unsqueeze(0)
            h_accumulate = torch.tensor(h_by_time).to(device).unsqueeze(0)
            
            mu, log_std, log_weights, h_n, _, _ = agent.reasoning_model.forward(
                z_accumulate, a_accumulate, h_accumulate
            )
            h = h_n
            dream_z = agent.reasoning_model.predict_next_z(mu, log_std, log_weights, True)
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
