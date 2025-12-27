import numpy as np

import torch
from torch.optim.adamw import AdamW

import gymnasium as gym

from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)

from introduction_to_world_model.model.world_model import WorldModel
from introduction_to_world_model.env.env import Env


def main(n_episodes: int, batch_size: int, device: str):
    env = Env()

    agent = WorldModel(env.observation_space, env.action_space)
    optimizer = AdamW(agent.vision_model.parameters())

    agent.collect_data(env, 100)

    losses = []

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        TextColumn("{task.fields[loss_info]}"),
    ) as progress:
        episode_task = progress.add_task(
            "[green]Episode", total=n_episodes, loss_info=""
        )

        for ep in range(n_episodes):
            n_step = len(agent.replay_buffer) // batch_size

            step_task = progress.add_task("[cyan]Steps", total=n_episodes, loss_info="")
            for step in range(n_step):
                batch_indices = np.arange(
                    step * batch_size, step * batch_size + batch_size
                )
                batch = agent.replay_buffer.get_batch(batch_indices)
                obs_img = batch[0]
                obs_img = torch.tensor(obs_img).to(device)

                reconstructed_obs_img, mu, log_var = agent.vision_model.forward(obs_img)
                loss, recons_loss, kl_loss = agent.vision_model.calculate_loss(
                    obs_img, reconstructed_obs_img, mu, log_var
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                current_loss = loss.item()
                progress.update(
                    step_task, advance=1, loss_info=f"[yellow]Loss: {current_loss:.4f}"
                )

                losses.append(
                    {
                        "loss": current_loss,
                        "reconstruction loss": recons_loss.item(),
                        "KL Divergence Loss": kl_loss.item(),
                    }
                )
                progress.update(step_task, advance=1)

            progress.update(episode_task, advance=1)
            progress.remove_task(step_task)


if __name__ == "__main__":
    main(1, batch_size=64, device="cpu")
