"""
Play in dream
"""
from collections import deque
from typing import Deque

import cv2

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
    
    if mode == ValidateMode.ID:
        from introduction_to_world_model.train.train_reasoning_model import get_rollout_batch_with_z
        batch_size = 1
        rollout_time_length = agent.reasoning_model.rollout_time_length
        batch_indices = np.random.choice(
            len(agent.replay_buffer) - rollout_time_length,
            size=batch_size,
            replace=False,
        )

        with torch.no_grad():
            seq_obs, seq_true_obs, seq_act, _, _, _, seq_z, seq_next_z = get_rollout_batch_with_z(
                agent, batch_indices, rollout_time_length, device
            )

            seq_z = torch.tensor(seq_z).to(device)
            seq_next_z = torch.tensor(seq_next_z).to(device)
            seq_act = torch.tensor(seq_act).to(device)

            seq_mu, seq_log_std, seq_log_weights, h_n, _, _ = (
                agent.reasoning_model.forward(seq_z, seq_act)
            )
            seq_dream_z = agent.reasoning_model.predict_next_z(seq_mu, seq_log_std, seq_log_weights) # (B, T, H)
            seq_dream_obs = agent.vision_model.decode(seq_dream_z).squeeze(0) # (T, H)
        
        seq_dream_obs = seq_dream_obs.cpu().numpy()
        seq_true_obs = seq_true_obs # (T, H)

        window_name = "Dream obs"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        for dream_obs, true_obs in zip(seq_dream_obs, seq_true_obs):
            dream_obs = ((dream_obs + 1.0) / 2.0 + 255).astype(np.uint8)
            true_obs = ((true_obs + 1.0) / 2.0 + 255).astype(np.uint8)

            cv2.imshow(window_name, dream_obs)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
            # Convert tensors to numpy for plotting
            # Create the figure
            # fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            # axes[0].imshow(true_obs)
            # axes[0].set_title(f"Ground truth observation ({mode.value})")
            # axes[0].axis("off")

            # axes[1].imshow(dream_obs)
            # axes[1].set_title("Dream observation")
            # axes[1].axis("off")

            # plt.show(block=False)  # Show without blocking code execution
            # plt.pause(1)  # Brief pause to allow the window to render

            # plt.close(fig)  # Close the window before the next iteration
    
    elif mode == ValidateMode.OOD:
        seq_obs, _ = env.reset()
        dream_obs = seq_obs.copy()
        h = agent.reasoning_model.get_initial_state(batch=True, batch_size=1, device=device)
        z_by_time: Deque[torch.Tensor] = deque(maxlen=agent.reasoning_model.rollout_time_length) # (T, H)
        a_by_time: Deque[torch.Tensor] = deque(maxlen=agent.reasoning_model.rollout_time_length) # (T, H)

        # Play in dream
        for step in range(n_steps):
            a = env.action_space.sample()
            a_torch = torch.tensor(a).to(device)
            a_torch = (
                torch.nn.functional.one_hot(a_torch, num_classes=env.action_space.n)
                .to(torch.float32)
            ) # (T, H)
            a_by_time.append(a_torch)
            with torch.no_grad():
                # Get latent from Vision Model's encoder
                mu, log_std = agent.vision_model.encode(
                    torch.tensor(dream_obs, device=device).unsqueeze(0) # (B, H)
                )
                seq_z = agent.vision_model.reparameterize(mu, log_std) # (T, H) (Actually it's (B, H) but B=1=T so we can consider it (T, H))
                z_by_time.append(seq_z.squeeze(0))

                # MDN-RNN inference
                z_rollout = torch.stack(list(z_by_time), dim=0).unsqueeze(0).to(device) # (B, T, H)
                a_rollout = torch.stack(list(a_by_time), dim=0).unsqueeze(0).to(device)
                h = h
                
                seq_mu, seq_log_std, seq_log_weights, h_n, _, _ = agent.reasoning_model.forward(
                    z_rollout, a_rollout, h
                )
                mu, log_std, log_weights, h = seq_mu[-1].unsqueeze(0), seq_log_std[-1].unsqueeze(0), seq_log_weights[-1].unsqueeze(0), h_n
                dream_z = agent.reasoning_model.predict_next_z(mu, log_std, log_weights, True) # (B, H)
                dream_obs = agent.vision_model.decode(dream_z).squeeze(0)

            seq_true_obs, _, _, _, _ = env.step(a)

            print(
                f"Dream-truth diff: {torch.nn.functional.mse_loss(dream_obs, torch.tensor(seq_true_obs).to(device)).cpu().numpy():.3f}"
            )

            dream_obs = (dream_obs.cpu().numpy() + 1.0) / 2.0
            seq_true_obs = (seq_true_obs + 1.0) / 2.0

            # Convert tensors to numpy for plotting
            # Create the figure
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            axes[0].imshow(seq_true_obs)
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
