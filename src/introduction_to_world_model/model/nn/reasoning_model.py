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
        self.log_weights = nn.Linear(256, num_mixture)

    def forward(
        self, rnn_output: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        rnn_output: torch.Tensor
            Output of RNN network (B*T, H)
        """
        rnn_output = self.base(rnn_output)

        mu = self.mu(rnn_output)  # (B*T, K*H)
        log_std = self.log_std(rnn_output)  # (B*T, K*H)
        log_weights = self.log_weights(rnn_output)  # (B*T, K)

        mu = mu.view(-1, self.num_mixture, self.z_size)
        log_std = log_std.view(-1, self.num_mixture, self.z_size)
        log_weights = log_softmax(log_weights, dim=-1)

        return mu, log_std, log_weights

    def sample(
        self,
        mu: torch.Tensor,        # (B, K, H)
        log_std: torch.Tensor,       # (B, K, H)
        log_weights: torch.Tensor,   # (B, K)
        deterministic: bool = False,
    ) -> torch.Tensor:
        B, K, H = mu.shape
        std = log_std.exp()
        weights = log_weights.exp()

        if deterministic:
            # Pick the mixture component with the highest weight
            chosen_distr_ind = torch.argmax(weights, dim=-1) # Shape: (B)
        else:
            # Sample a component index based on the weights
            cat = Categorical(probs=weights)
            chosen_distr_ind = cat.sample() # Shape: (B)

        # We need to reshape the index to (B, 1, H) to use torch.gather
        # This allows us to pick the specific K-dimension for each Batch
        idx = chosen_distr_ind.view(B, 1, 1).expand(B, 1, H)
        
        # Gather the chosen mean and std
        chosen_mu = torch.gather(mu, 1, idx).squeeze(1)   # (B, H)
        chosen_std = torch.gather(std, 1, idx).squeeze(1) # (B, H)

        if deterministic:
            return chosen_mu
        else:
            # Sample from the chosen Gaussian
            # Use the reparameterization trick if you need gradients (torch.randn_like)
            eps = torch.randn_like(chosen_mu)
            return chosen_mu + eps * chosen_std
    
    @classmethod
    def calculate_loss(cls, z, mu, log_std, log_pi) -> torch.Tensor:
        """
        z:        (B, z_size)
        mu:       (B, K, z_size)
        log_std:  (B, K, z_size)
        log_pi:   (B, K)          (log-softmaxed!)
        """

        B, K, Z = mu.shape

        # Expand z to (B, 1, z_size) WITHOUT materializing K copies
        z = z.view(-1, 1, z.shape[-1])

        # Gaussian log-likelihood, computed manually
        # (B, K, z_size)
        log_prob = -0.5 * ((z - mu) / log_std.exp()) ** 2 \
                - log_std \
                - LOG_SQRT_2PI

        # Sum over z dimensions → (B, K)
        log_prob = log_prob.sum(dim=-1)

        # Add mixture weights
        log_prob = log_prob + log_pi

        # LogSumExp over mixtures → (B,)
        loss = -torch.logsumexp(log_prob, dim=1)

        return loss.mean()


class MDNRNN(nn.Module):
    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        z_size: int,
        rollout_time_length: int = 512,
        hidden_size: int = 256,
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

        B, T, H = output.shape
        output = output.reshape(B * T, H)

        mu, log_std, log_weights = self.mdn(output)

        r = None if not self.predict_reward else self.reward_head(output)
        d = None if not self.predict_done else self.done_head(output)

        return mu, log_std, log_weights, h_n, r, d

    def predict_next_z(
        self,
        mu: torch.Tensor,
        log_std: torch.Tensor,
        log_weights: torch.Tensor,
        deterministic=True,
    ) -> torch.Tensor:
        """
        mu: torch.Tensor (B, K, H)
        """
        next_z = self.mdn.sample(mu, log_std, log_weights, deterministic)

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
        z_target = z_target.view(-1, 1, z_target.size(-1))  # (B*T, 1, z_size)
        loss = MixtureDensityHead.calculate_loss(z_target, mu_predict, log_std_predict, log_weights_predict)

        if self.predict_reward and r_target is not None and r_predict is not None:
            loss += mse_loss(r_predict, r_target.view(-1, 1))

        if self.predict_done and done_target is not None and done_predict is not None:
            loss += binary_cross_entropy_with_logits(
                done_predict, done_target.view(-1, 1)
            )

        return loss

    def get_initial_state(
        self, batch_size: int, batch: bool, device: str
    ) -> torch.Tensor:
        if batch:
            h_0 = torch.zeros(1, batch_size, self.z_size).to(device)
        else:
            h_0 = torch.zeros(1, self.z_size).to(device)

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
