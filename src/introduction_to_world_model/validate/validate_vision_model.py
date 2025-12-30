import numpy as np

import torch

import gymnasium as gym

import matplotlib.pyplot as plt

from introduction_to_world_model.model.world_model import WorldModel
from introduction_to_world_model.env.env import Env
from introduction_to_world_model.validate import ValidateMode


def validate_vision_model(
    agent: WorldModel,
    n_examples: int,
    mode: ValidateMode = ValidateMode.ID,
    env: gym.Env | None = None,
):
    agent.vision_model.eval()

    if mode == ValidateMode.OOD:
        if env is None:
            raise RuntimeError("Must provide a environment in mode ValidateMode.OOD")

        obs = []
        agent.replay_buffer.reset()
        agent.collect_data(env, 100_000)

    random_indices = np.random.randint(0, len(agent.replay_buffer), size=n_examples)
    obs, _, _, _, _, _ = agent.replay_buffer.get_batch(random_indices)
    obs = torch.tensor(obs)

    for idx, img in enumerate(obs):
        with torch.no_grad():
            reconstructed_img, mu, log_std = agent.vision_model.forward(
                img.unsqueeze(0)
            )

            print(
                f"Mean Mu: {mu.squeeze(0).mean().numpy()}\nMean LogVar: {log_std.squeeze(0).mean().numpy()}"
            )
            print(
                f"Reconstruction diff: {torch.nn.functional.mse_loss(reconstructed_img.squeeze(0), img).numpy():.3f}"
            )

        reconstructed_img = (reconstructed_img.squeeze(0).numpy() + 1.0) / 2.0
        img = (img.numpy() + 1.0) / 2.0

        # Convert tensors to numpy for plotting
        # Create the figure
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(img)
        axes[0].set_title(f"Original ({mode.value})")
        axes[0].axis("off")

        axes[1].imshow(reconstructed_img)
        axes[1].set_title("Reconstructed")
        axes[1].axis("off")

        fig.savefig(f"results/vision_model/img_{idx}.png")
        plt.show(block=False)  # Show without blocking code execution
        plt.pause(0.1)  # Brief pause to allow the window to render

        print("Press [Enter] to see the next image...")
        input()  # Wait for user input
        plt.close(fig)  # Close the window before the next iteration


if __name__ == "__main__":
    env = Env(render_mode="rgb_array")

    agent = WorldModel(env.observation_space, env.action_space)
    agent.load_checkpoint("checkpoint/trained_vision_model.pt")

    validate_vision_model(agent, 10, ValidateMode.OOD, env=env)
