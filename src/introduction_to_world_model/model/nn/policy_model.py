import torch
import torch.nn as nn

import gymnasium as gym


class PolicyNetwork(nn.Module):
    def __init__(
        self,
        z_size: int,
        rnn_hidden_size,
        action_space: gym.spaces.Discrete,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.action_size = int(action_space.n)

        self.net = nn.Linear(z_size + rnn_hidden_size, self.action_size)

    def forward(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        action = self.net(torch.cat([z, h], dim=-1))

        return action
