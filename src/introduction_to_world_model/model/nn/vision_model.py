from typing import List, Tuple
from warnings import warn

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import gymnasium as gym


class ConvVAE(nn.Module):
    """
    The Convolutional VAE Model to encode the observation into a latent vector
    """

    def __init__(
        self,
        obs_space: gym.spaces.Box,
        hidden_dims: List = [32, 64, 128, 256],
        latent_dim: int = 128,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.obs_space = obs_space  # HxWxC
        self.latent_dim = latent_dim
        in_channels = self.obs_space.shape[-1]

        # Build Encoder
        modules: List[nn.Module] = []
        H, W, C = self.obs_space.shape
        output_shape = np.array([C, H, W])  # CxHxW
        kernel_size = np.array([3, 3])
        stride = 2
        padding = 1

        for hidden_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=hidden_dim,
                        kernel_size=kernel_size.tolist(),
                        stride=stride,
                        padding=padding,
                    ),
                    nn.ReLU(),
                )
            )

            in_channels = hidden_dim
            output_shape[1:] = (
                np.floor(
                    (output_shape[1:] - kernel_size + padding * 2) / stride
                ).astype(int)
                + 1
            )
            output_shape[0] = hidden_dim

        output_dim: int = output_shape[0] * output_shape[1] * output_shape[2]
        self.encoder_last_conv_shape = output_shape.copy()
        self.encoder = nn.Sequential(*modules, nn.Flatten())

        self.mu = nn.Linear(output_dim, latent_dim)
        self.log_var = nn.Linear(output_dim, latent_dim)

        # Build decoder
        modules: List[nn.Module] = []

        self.decoder_input = nn.Linear(latent_dim, output_dim)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=hidden_dims[i],
                        out_channels=hidden_dims[i + 1],
                        kernel_size=kernel_size.tolist(),
                        stride=stride,
                        padding=padding,
                        output_padding=padding,
                    ),
                    nn.ReLU(),
                )
            )

        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=hidden_dims[-1],
                    out_channels=hidden_dims[-1],
                    kernel_size=kernel_size.tolist(),
                    stride=stride,
                    padding=padding,
                    output_padding=padding,
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=hidden_dims[-1],
                    out_channels=3,
                    kernel_size=kernel_size.tolist(),
                    padding=padding,
                ),
                nn.Tanh(),
            )
        )

        self.decoder = nn.Sequential(*modules)

    def encode(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        obs: torch.Tensor
            observation (BxCxHxW)
        """
        if len(obs.shape) < 4:
            obs = obs.unsqueeze(0)
            warn("Inferencing without batch dim")

        obs = obs.permute(0, 3, 1, 2)  # BxHxWxC
        latent = self.encoder(obs)

        mu = self.mu(latent)
        log_var = self.log_var(latent)

        return mu, log_var

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        """
        output = self.decoder_input(z)
        output = output.view(-1, *(self.encoder_last_conv_shape.tolist()))
        output = self.decoder(output)  # BxCxHxW
        output = output.permute(0, 2, 3, 1)  # BxHxWxC

        return output

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        """
        std = torch.exp(0.5 * log_var)
        distr = Normal(loc=mu, scale=std)

        return distr.rsample()

    def forward(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(obs)
        z = self.reparameterize(mu, log_var)
        reconstructed_img = self.decode(z)

        return reconstructed_img, mu, log_var

    def generate_from_obs(self, obs: torch.Tensor) -> torch.Tensor:
        output, _, _ = self.forward(obs)

        return output

    def sample(self, num_samples: int, device: str = "cpu") -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(device)

        samples = self.decode(z)

        return samples

    def calculate_loss(
        self,
        obs: torch.Tensor,
        reconstructed_img: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        beta: float = 0.01,
        kl_tolerance: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        reconstruction_loss = torch.sum(
            (reconstructed_img - obs) ** 2, dim=[1, 2, 3]
        ).mean()
        kld_loss = -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=-1)
        kld_loss = torch.maximum(
            kld_loss,
            torch.tensor(kl_tolerance * self.latent_dim, device=kld_loss.device),
        ).mean()

        loss = reconstruction_loss + beta * kld_loss

        return loss, reconstruction_loss.detach(), kld_loss.detach()


if __name__ == "__main__":
    obs_space = gym.spaces.Box(
        low=np.zeros(shape=(128, 128, 3)), high=np.ones(shape=(128, 128, 3))
    )
    obs = torch.tensor(obs_space.sample())
    conv_vae = ConvVAE(obs_space)
    mu, log_var = conv_vae.encode(obs)
    z = conv_vae.reparameterize(mu, log_var)
    sample = conv_vae.decode(z)

    print(sample.shape)

    from torchsummary import summary
    summary(conv_vae, obs)