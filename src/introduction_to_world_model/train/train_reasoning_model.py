import warnings

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


def get_rollout_batch_with_z(
    agent: WorldModel,
    batch_indices: np.ndarray,
    rollout_time_length: int,
    device: str,
):
    time_offsets = np.arange(rollout_time_length)
    rollout_indices = batch_indices[:, None] + time_offsets[None, :]

    # (B, T, ...)
    _, next_obs, act, reward, terminated, truncated = \
        agent.replay_buffer.get_batch(rollout_indices.flatten())

    # reshape back
    next_obs = next_obs.reshape(
        batch_indices.shape[0], rollout_time_length, *next_obs.shape[1:]
    )

    next_obs_torch = torch.tensor(next_obs).to(device)

    with torch.no_grad():
        B, T = next_obs_torch.shape[:2]
        flat = next_obs_torch.view(B * T, *next_obs_torch.shape[2:])
        mu, log_var = agent.vision_model.encode(flat)
        z = agent.vision_model.reparameterize(mu, log_var)
        z = z.view(B, T, -1)

    return (
        None,               # obs (unused)
        None,               # next_obs (unused)
        act.reshape(B, T, -1),
        reward.reshape(B, T),
        terminated.reshape(B, T),
        truncated.reshape(B, T),
        z.cpu().numpy(),
    )


def train_reasoning_model(
    agent: WorldModel,
    n_epochs: int,
    batch_size: int,
    rollout_time_length: int,
    learning_rate: float,
    device: str = "cpu",
    save_path: str | None = None,
    load_path: str | None = None,
):
    wandb.init(
        project="introduction_to_world_model",
    )
    wandb.watch(agent.reasoning_model, log_freq=10)

    optimizer = AdamW(agent.reasoning_model.parameters(), lr=learning_rate)

    if load_path is not None:
        agent.load_checkpoint(load_path, vision_optimizer=optimizer, device=device)
    else:
        warnings.warn(
            "\nNo pre-trained vision model loaded!\nYou should train vision model before training reasoning model, otherwise the model can not learn meaningful representation and reasoning!"
        )
        agent.collect_data(Env(), 10_000)

    if len(agent.replay_buffer) == 0:
        raise RuntimeError(
            "Agent replay buffer is empty! You must collect data first! Hint: Use <WorldModel>.collect_data(<env>)."
        )

    agent.to(device)
    agent.train(False)
    agent.reasoning_model.train(True)

    ep_sum_losses = [
        {
            "Loss": 0.0,
        }
        for _ in range(n_epochs)
    ]

    n_steps_per_epoch = len(agent.replay_buffer) // batch_size
    n_steps = (
        len(agent.replay_buffer) // batch_size - rollout_time_length
    )  # Prevent out of bound because we need to ensure the future observation are observable within out dataset

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
            loss_info="[yellow]Episode Total Loss: --",
        )

        print("Training reasoning model...")
        for ep in range(n_epochs):
            step_task = progress.add_task(
                "[cyan]Step",
                total=n_steps,
                loss_info="[yellow]Loss: --",
            )

            for step in range(n_steps_per_epoch):
                current_step = ep * n_steps_per_epoch + step

                batch_indices = np.random.choice(
                    len(agent.replay_buffer) - rollout_time_length,
                    size=batch_size,
                    replace=False,
                )

                _, _, act, r_target, term, trunc, z_target = get_rollout_batch_with_z(
                    agent, batch_indices, rollout_time_length, device
                )

                z_target = torch.tensor(z_target).to(device)
                act = torch.tensor(act).to(device)
                r_target = torch.tensor(r_target).to(device)
                done_target = torch.tensor(np.max([term, trunc], axis=0)).to(device)

                mu, log_std, log_weights, h_n, r_predict, done_predict = (
                    agent.reasoning_model.forward(z_target, act)
                )
                loss = agent.reasoning_model.compute_loss(
                    z_target,
                    mu,
                    log_std,
                    log_weights,
                    r_target,
                    r_predict,
                    done_target,
                    done_predict,
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                current_loss = loss.item()

                ep_sum_losses[ep]["Loss"] += current_loss

                progress.update(
                    step_task,
                    advance=1,
                    loss_info=(f"[yellow]Loss: {current_loss:.3f} | "),
                )

                progress.update(
                    episode_task,
                    completed=round(current_step / n_steps * n_epochs, 3),
                    loss_info=(f"[yellow]Episode Total Loss: {ep_sum_losses[ep]['Loss']:.3f} | "),
                )

                wandb.log(
                    {
                        "Loss": current_loss,
                    }
                )

            progress.remove_task(step_task)

    if save_path is not None:
        agent.save_checkpoint(save_path, vision_optimizer=optimizer)


if __name__ == "__main__":
    env = Env()

    agent = WorldModel(env.observation_space, env.action_space)

    train_reasoning_model(
        agent,
        2,
        batch_size=2,
        rollout_time_length=16,
        learning_rate=1e-4,
        device="cpu",
        save_path="checkpoint/trained_reasoning_mode.pt",
        # load_path="checkpoint/trained_vision_model.pt",
    )
