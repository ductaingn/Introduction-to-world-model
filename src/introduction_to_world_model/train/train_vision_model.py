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


def train_vision_model(
    agent: WorldModel,
    n_epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str = "cpu",
    save_path: str | None = None,
    load_path: str | None = None,
):
    if len(agent.replay_buffer) == 0:
        raise RuntimeError("Agent replay buffer is empty! You must collect data first! Hint: Use <WorldModel>.collect_data(<env>).")

    wandb.init(
        project="introduction_to_world_model",
    )
    wandb.watch(agent.vision_model, log_freq=10)

    optimizer = AdamW(agent.vision_model.parameters(), lr=learning_rate)

    if load_path is not None:
        agent.load_checkpoint(load_path, vision_optimizer=optimizer, device=device)

    agent.to(device)
    agent.train(False)
    agent.vision_model.train(True)

    ep_sum_losses = [
        {"Loss": 0.0, "Reconstruction Loss": 0.0, "KL Divergence Loss": 0.0}
        for _ in range(n_epochs)
    ]

    n_steps_per_epoch = len(agent.replay_buffer) // batch_size
    n_steps = n_steps_per_epoch * n_epochs
    warmup_steps = 10_000
    beta_max = 0.01

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
            total=n_epochs,
            loss_info="[yellow]Episode Total Loss: -- | [cyan]Reconstruction Loss: -- | [magenta]KL Divergence Loss: --",
        )

        print("Training vision model...")
        for ep in range(n_epochs):
            step_task = progress.add_task(
                "[cyan]Step",
                total=n_steps_per_epoch,
                loss_info="[yellow]Loss: -- | [cyan]Reconstruction Loss: -- | [magenta]KL Divergence Loss: --",
            )
            for step in range(n_steps_per_epoch):
                batch_indices = np.random.choice(
                    len(agent.replay_buffer), size=batch_size, replace=False
                )
                batch = agent.replay_buffer.get_batch(batch_indices)
                obs_img = batch[0]
                obs_img = torch.tensor(obs_img).to(device)

                reconstructed_obs_img, mu, log_var = agent.vision_model.forward(obs_img)

                current_step = ep * n_steps_per_epoch + step
                beta = min(beta_max, beta_max * current_step / warmup_steps)
                loss, recons_loss, kl_loss = agent.vision_model.calculate_loss(
                    obs_img, reconstructed_obs_img, mu, log_var, beta
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

                progress.update(
                    episode_task,
                    completed=round(current_step/n_steps*n_epochs, 3),
                    loss_info=(
                        f"[yellow]Episode Total Loss: {ep_sum_losses[ep]['Loss']:.3f} | "
                        f"[cyan]Reconstruction Loss: {ep_sum_losses[ep]['Reconstruction Loss']:.3f} | "
                        f"[magenta]KL Divergence Loss: {ep_sum_losses[ep]['KL Divergence Loss']:.3f} | "
                    ),
                )

                wandb.log(
                    {
                        "Loss": current_loss,
                        "Reconstruction Loss": recons_loss.item(),
                        "KL Divergence Loss": kl_loss.item(),
                        "Mean absolute Mu": mu.detach().abs().mean().item(),
                        "Mean absolute LogVar": log_var.detach().abs().mean().item(),
                        "Beta": beta,
                    }
                )

            progress.remove_task(step_task)

    if save_path is not None:
        agent.save_checkpoint(save_path, vision_optimizer=optimizer)

    return optimizer

if __name__ == "__main__":
    env = Env()

    agent = WorldModel(env.observation_space, env.action_space)

    print("Collecting rollout data")
    agent.collect_data(env, n_step=10000)

    train_vision_model(
        agent,
        20,
        batch_size=256,
        learning_rate=1e-4,
        device="cuda:0",
        save_path="checkpoint/trained_vision_model.pt",
        # load_path="checkpoint/trained_vision_model.pt",
    )
