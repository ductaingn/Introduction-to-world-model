import numpy as np

import torch
from torch.optim.adamw import AdamW

import wandb

from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
)

from introduction_to_world_model.model.world_model import WorldModel
from introduction_to_world_model.env.env import Env


def train_policy_model(
    agent: WorldModel,
    n_episodes: int,
    batch_size: int,
    learning_rate: float,
    device: str = "cpu",
    save_path: str | None = None,
):
    wandb.init(
        project="introduction_to_world_model",
    )
    wandb.watch(agent.vision_model, log_freq=10)

    agent.to(device)
    agent.train(False)
    agent.policy_model.train(True)

    optimizer = AdamW(agent.vision_model.parameters(), lr=learning_rate)

    ep_sum_losses = [
        {"Loss": 0.0, "Reconstruction Loss": 0.0, "KL Divergence Loss": 0.0}
        for _ in range(n_episodes)
    ]

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[bold]{task.completed}/{task.total}"),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
        TextColumn("{task.fields[loss_info]}"),
    ) as progress:
        episode_task = progress.add_task(
            "[green]Episode",
            total=n_episodes,
            loss_info="[yellow]Loss: -- | [cyan]Reconstruction Loss: -- | [magenta]KL Divergence Loss: --",
        )

        print("Training vision model...")
        for ep in range(n_episodes):
            n_step = len(agent.replay_buffer) // batch_size

            step_task = progress.add_task(
                "[cyan]Step",
                total=n_step,
                loss_info="[yellow]Loss: -- | [cyan]Reconstruction Loss: -- | [magenta]KL Divergence Loss: --",
            )
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

                ep_sum_losses[ep]["Loss"] += current_loss
                ep_sum_losses[ep]["Reconstruction Loss"] += recons_loss.item()
                ep_sum_losses[ep]["KL Divergence Loss"] += kl_loss.item()

                progress.update(
                    step_task,
                    advance=1,
                    loss_info=(
                        f"[yellow]Loss: {current_loss:.3f} | "
                        f"[cyan]Reconstruction Loss: {recons_loss.item():.3f} | "
                        f"[magenta]KL Divergence Loss: {kl_loss.item():.3f} | "
                    ),
                )

                wandb.log(
                    {
                        "Loss": current_loss,
                        "Reconstruction Loss": recons_loss.item(),
                        "KL Divergence Loss": kl_loss.item(),
                    }
                )

            progress.update(
                episode_task,
                advance=1,
                loss_info=(
                    f"[yellow]Loss: {ep_sum_losses[ep]['Loss']:.3f} | "
                    f"[cyan]Reconstruction Loss: {ep_sum_losses[ep]['Reconstruction Loss']:.3f} | "
                    f"[magenta]KL Divergence Loss: {ep_sum_losses[ep]['KL Divergence Loss']:.3f} | "
                ),
            )
            progress.remove_task(step_task)

    if save_path is not None:
        agent.save_checkpoint(save_path, vision_optimizer=optimizer)


if __name__ == "__main__":
    env = Env()

    agent = WorldModel(env.observation_space, env.action_space)

    print("Collecting rollout data")
    agent.collect_data(env, n_step=10000)

    train_vision_model(
        agent,
        10,
        batch_size=128,
        learning_rate=1e-4,
        device="cpu",
        save_path="checkpoint/trained_vision_model.pt",
    )
