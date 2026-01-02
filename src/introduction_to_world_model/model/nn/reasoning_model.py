from typing import Tuple
import math

import gymnasium as gym

import torch
import torch.nn as nn
from torch.nn.functional import mse_loss, binary_cross_entropy_with_logits, log_softmax
from torch.distributions import Categorical

LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


class MixtureDensityHead(nn.Module):
    def __init__(self, rnn_hidden_size, z_size, num_mixture=5, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.z_size = z_size
        self.num_mixture = num_mixture

        self.base = nn.Sequential(
            nn.Linear(rnn_hidden_size, 256),
            nn.ReLU(),
        )
        self.mu = nn.Linear(256, num_mixture * z_size)
        self.log_std = nn.Linear(256, num_mixture * z_size)
        self.log_weights = nn.Linear(256, num_mixture * z_size)

    def forward(self, rnn_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # rnn_output shape can be (B, T, H) or (B*T, H)
        is_3d = len(rnn_output.shape) == 3
        if is_3d:
            B, T, H = rnn_output.shape
        
        flat_rnn_out = rnn_output.reshape(-1, rnn_output.shape[-1])
        x = self.base(flat_rnn_out)

        mu = self.mu(x).view(-1, self.z_size, self.num_mixture)
        log_std = torch.clamp(self.log_std(x).view(-1, self.z_size, self.num_mixture), -5.0, 2.0)
        
        # Calculate log_pi and ensure it sums to 1 across the mixture dim (-1)
        log_pi = self.log_weights(x).view(-1, self.z_size, self.num_mixture)
        log_pi = torch.log_softmax(log_pi, dim=-1)

        # If input was (B, T, H), return (B, T, H, K)
        if is_3d:
            mu = mu.view(B, T, self.z_size, self.num_mixture)
            log_std = log_std.view(B, T, self.z_size, self.num_mixture)
            log_pi = log_pi.view(B, T, self.z_size, self.num_mixture)

        return mu, log_std, log_pi

    def sample(
        self,
        mu: torch.Tensor,        # (B, T, H, K)
        log_std: torch.Tensor,       # (B, T, H, K)
        log_weights: torch.Tensor,   # (B, T, H, K)
        temperature=1.0
    ) -> torch.Tensor:
        log_weights = log_weights / temperature
        weights = torch.softmax(log_weights, dim=-1)
        
        # 2. Scale standard deviation
        std = (log_std.exp()) * math.sqrt(temperature)
        
        # 3. Sample mixture indices for each dimension
        # Shape: (B, T, z_size, 1)
        cat = torch.distributions.Categorical(probs=weights)
        indices = cat.sample().unsqueeze(-1)
        
        # 4. Gather the chosen mu and std
        chosen_mu = torch.gather(mu, -1, indices).squeeze(-1)
        chosen_std = torch.gather(std, -1, indices).squeeze(-1)
        
        # 5. Final Gaussian sample
        epsilon = torch.randn_like(chosen_mu)
        return chosen_mu + epsilon * chosen_std
    
    @classmethod
    def calculate_loss(cls, z_target, mu, log_std, log_pi, mask=None) -> torch.Tensor:
        # Match dimensions: z_target (B, T, Z) -> (B, T, Z, 1)
        z_target = z_target.unsqueeze(-1)

        # Standard MDN Log-Probability
        # Using a small eps for variance stability is a good practice
        variance = (2.0 * log_std).exp()
        log_prob = -0.5 * ((z_target - mu)**2 / variance) - log_std - LOG_SQRT_2PI

        # LogSumExp over the K mixtures
        weighted_log_prob = log_pi + log_prob
        loss = -torch.logsumexp(weighted_log_prob, dim=-1) # (B, T, Z)

        if mask is not None:
            # mask shape (B, T) or (B, T, 1)
            loss = loss * mask.view(loss.size(0), loss.size(1), 1)
            return loss.sum() / (mask.sum() * mu.size(-2)) # Normalized by unmasked elements
            
        return loss.mean()


class MDNRNN(nn.Module):
    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        z_size: int,
        rollout_time_length: int = 512,
        hidden_size: int = 256,
        n_rnn_layers: int = 1,
        predict_reward: bool = False,
        predict_done: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.z_size = z_size
        self.a_size = action_space.n
        self.rollout_time_length = rollout_time_length
        self.predict_reward = predict_reward
        self.predict_done = predict_done
        self.hidden_size = hidden_size
        self.n_rnn_layers = n_rnn_layers

        self.gru = nn.GRU(
            input_size=self.a_size + z_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.mdn = MixtureDensityHead(hidden_size, z_size)

        if predict_reward:
            self.reward_head = nn.Linear(hidden_size, 1)

        if predict_done:
            self.done_head = nn.Linear(hidden_size, 1)

    def forward(
        self, z: torch.Tensor, a: torch.Tensor, h: torch.Tensor | None = None
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """
        Parameters
        z: torch.Tensor
            Latent variable (The output of the encoder of the VAE model) (BxTxH)
        action: torch.Tensor
            Action (The output of the policy model) (BxTxH)
        """
        # TODO: Implement mask
        x = torch.concat([z, a], dim=-1)
        output, h_n = self.gru(x, h)

        mu, log_std, log_weights = self.mdn(output)

        r = None if not self.predict_reward else self.reward_head(output)
        d = None if not self.predict_done else self.done_head(output)

        return mu, log_std, log_weights, h_n, r, d

    def predict_next_z(
        self,
        mu: torch.Tensor,
        log_std: torch.Tensor,
        log_weights: torch.Tensor,
        temparture: float=1.0,
    ) -> torch.Tensor:
        """
        mu: torch.Tensor (B, T, K, H)
        """
        next_z = self.mdn.sample(mu, log_std, log_weights, temparture)

        return next_z

    def compute_loss(
        self,
        z_target: torch.Tensor,
        mu_predict: torch.Tensor,
        log_std_predict: torch.Tensor,
        log_weights_predict: torch.Tensor,
        r_target: torch.Tensor | None,
        r_predict: torch.Tensor | None,
        done_target: torch.Tensor | None,
        done_predict: torch.Tensor | None,
    ) -> torch.Tensor:
        # TODO: Verify
        # TODO: Implement mask
        loss = MixtureDensityHead.calculate_loss(z_target, mu_predict, log_std_predict, log_weights_predict)

        if self.predict_reward and r_target is not None and r_predict is not None:
            loss += mse_loss(r_predict, r_target.view(-1, 1))

        if self.predict_done and done_target is not None and done_predict is not None:
            loss += binary_cross_entropy_with_logits(
                done_predict, done_target.view(-1, 1)
            )

        return loss

    def get_initial_state(
        self, *, batch: bool, batch_size: int=None, device: str="cpu"
    ) -> torch.Tensor:
        if batch:
            assert batch_size is not None, ("You must provide batch size!")
            h_0 = torch.zeros(batch_size, self.n_rnn_layers, self.hidden_size).to(device)
        else:
            h_0 = torch.zeros(self.n_rnn_layers, self.hidden_size).to(device)

        return h_0


if __name__ == "__main__":
    from torchsummary import summary
    device = "cuda:0"

    B, T = 2, 1000
    action_space = gym.spaces.Discrete(3)
    mdnrnn = MDNRNN(action_space, z_size=128, rollout_time_length=T).to(device)
    summary(mdnrnn)

    # latent states
    z = torch.rand(B, T, 128).to(device)
    next_z = torch.rand(B, T, 128).to(device)

    # actions (one-hot)
    a = (
        torch.nn.functional.one_hot(
            torch.tensor([action_space.sample() for i in range(B * T)]),
            num_classes=action_space.n,
        )
        .float()
        .view(B, T, -1)
        .to(device)
    )

    mu, log_std, log_weights, h, r, d = mdnrnn(z, a)

    print("mu:", mu.shape)
    print("log_std:", log_std.shape)
    print("log_weights:", log_weights.shape)
    print("reward:", None if r is None else r.shape)
    print("done:", None if d is None else d.shape)

    next_z_predict = mdnrnn.predict_next_z(mu, log_std, log_weights)
    print("next_z_predict:", next_z_predict.shape)

    loss = mdnrnn.compute_loss(next_z, mu, log_std, log_weights, r, r, d, d)
    print(loss.detach().cpu().item())
