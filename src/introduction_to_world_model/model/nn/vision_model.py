from typing import List, Tuple
from warnings import warn

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import gymnasium as gym


class VAE(nn.Module):
    """
    The Convolutional VAE Model to encode the observation into a latent vector
    """

    def __init__(
        self,
        obs_space: gym.spaces.Box,
        latent_dim: int,
        hidden_dims: List = [32, 64, 128, 256, 256],
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


class VectorQuantizer(nn.Module):
    def __init__(
        self, n_embeddings: int, embedding_dim: int, beta: float, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.K = n_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latents = latents.permute(0, 2, 3, 1).contiguous()
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)

        dist = (
            torch.sum(flat_latents**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_latents, self.embedding.weight.t())
        )

        encodings_inds = torch.argmin(dist, dim=1).unsqueeze(1)

        device = latents.device
        encoding_one_hot = torch.zeros(encodings_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encodings_inds, 1)

        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)
        quantized_latents = quantized_latents.view(latents_shape)

        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        quantized_latents = latents + (quantized_latents - latents).detach()

        return quantized_latents.permute(0, 3, 1, 2).contiguous(), vq_loss


class ResidualLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.res_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input + self.res_block(input)


class VQVAE(nn.Module):
    def __init__(
        self,
        obs_space: gym.spaces.Box,
        latent_dim: int,
        n_embeddings: int = 512,
        hidden_dims: List = [32, 64, 128, 256],
        beta: float = 0.25,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.obs_space = obs_space  # HxWxC
        self.latent_dim = latent_dim
        self.n_embeddings = n_embeddings
        in_channels = self.obs_space.shape[-1]

        # Build Encoder
        modules: List[nn.Module] = []
        H, W, C = self.obs_space.shape
        output_shape = np.array([C, H, W])  # CxHxW
        kernel_size = np.array([4, 4])
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
                    nn.LeakyReLU(),
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

        modules.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=np.array([3, 3]).tolist(),
                    stride=1,
                    padding=1,
                )
            )
        )

        for _ in range(6):
            modules.append(ResidualLayer(hidden_dims[-1], hidden_dims[-1]))
        modules.append(nn.LeakyReLU())

        modules.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    latent_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
        )

        self.encoder_last_conv_shape = output_shape.copy()
        self.encoder = nn.Sequential(*modules)

        self.vq_layer = VectorQuantizer(n_embeddings, latent_dim, beta)

        # Build decoder
        modules: List[nn.Module] = []

        modules.append(
            nn.Sequential(
                nn.Conv2d(
                    latent_dim, hidden_dims[-1], kernel_size=3, stride=1, padding=1
                )
            )
        )

        for _ in range(6):
            modules.append(ResidualLayer(hidden_dims[-1], hidden_dims[-1]))
        modules.append(nn.LeakyReLU())

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
                    ),
                    nn.LeakyReLU(),
                )
            )

        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=hidden_dims[-1],
                    out_channels=3,
                    kernel_size=kernel_size.tolist(),
                    stride=stride,
                    padding=padding,
                ),
                nn.Tanh(),
            )
        )

        self.decoder = nn.Sequential(*modules)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        obs: torch.Tensor
            observation (BxCxHxW)
        """
        if len(obs.shape) < 4:
            obs = obs.unsqueeze(0)
            warn("Inferencing without batch dim")

        obs = obs.permute(0, 3, 1, 2)  # BxHxWxC
        latents = self.encoder(obs)

        return latents

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z: torch.Tensor
            Latent representation (BxDxHxW)
        """
        output = self.decoder(z)  # BxCxHxW
        output = output.permute(0, 2, 3, 1)  # BxHxWxC

        return output

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        """
        std = torch.exp(0.5 * log_var)
        distr = Normal(loc=mu, scale=std)

        return distr.rsample()

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoding = self.encode(obs)
        quantized_inputs, vq_loss = self.vq_layer(encoding)
        reconstructed_img = self.decode(quantized_inputs)

        return reconstructed_img, vq_loss

    def generate_from_obs(self, obs: torch.Tensor) -> torch.Tensor:
        output, _ = self.forward(obs)

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
        vq_loss: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        reconstruction_loss = torch.sum((reconstructed_img - obs) ** 2).mean()

        loss = reconstruction_loss + vq_loss

        return loss, reconstruction_loss.detach(), vq_loss.detach()


if __name__ == "__main__":
    from torchsummary import summary

    device = "cpu"

    obs_space = gym.spaces.Box(
        low=np.zeros(shape=(128, 128, 3)), high=np.ones(shape=(128, 128, 3))
    )
    obs = torch.tensor(obs_space.sample()).to(device)
    print("Observation: ", obs.shape)
    vae = VAE(obs_space, 128).to(device)
    mu, log_var = vae.encode(obs)
    z = vae.reparameterize(mu, log_var)
    sample = vae.decode(z)
    print("VAE sample: ", sample.shape)

    summary(vae, obs)

    obs = torch.tensor(obs_space.sample()).to(device)
    vqvae = VQVAE(obs_space, 128).to(device)
    latent = vqvae.encode(obs)
    print("VQVAE Latent: ", latent.shape)
    sample, _ = vqvae.forward(obs)
    print("VQVAE sample: ", sample.shape)

    summary(vqvae, obs)
