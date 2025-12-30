from typing import Tuple

import gymnasium as gym
import torch
import torch.nn as nn
from torch.nn.functional import mse_loss, binary_cross_entropy_with_logits, log_softmax
from torch.distributions import Categorical, Normal


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
        self, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.base(h)

        mu = self.mu(h)
        log_std = self.log_std(h)
        log_weights = self.log_weights(h)

        mu = mu.view(-1, self.num_mixture, self.z_size)
        log_std = log_std.view(-1, self.num_mixture, self.z_size)
        log_weights = log_softmax(log_weights, dim=-1)

        return mu, log_std, log_weights
    
    def sample(self, mu: torch.Tensor, log_std: torch.Tensor, log_weights: torch.Tensor, deterministic: bool=True) -> torch.Tensor:
        B, K, H = mu.shape
        uniform_samples = torch.rand(B).to(mu.device)
        cum_weights = log_weights.exp().cumsum(dim=1).to(mu.device)

        sampled_pred = torch.zeros(B, H).to(mu.device)
        for b in range(B):
            if deterministic:
                k = torch.argmax(log_weights[b])
            else:
                k = torch.searchsorted(cum_weights[b], uniform_samples[b]).item()
            sampled_pred[b] = torch.normal(mu[b, k], log_std[b, k].exp())

        return sampled_pred

def mdn_loss(
    z_target: torch.Tensor,
    mu_predict: torch.Tensor,
    log_std_predict: torch.Tensor,
    log_weights_predict: torch.Tensor,
    eps: float=1e-8,
) -> torch.Tensor:
    """
    z_target: (B*T, z_size)
    mu: (B*T, num_mixture, z_size)
    log_std: (B*T, num_mixture, z_size)
    log_weights: (B*T, num_mixture, z_size)
    """
    std_predict = log_std_predict.exp()
    m = Normal(loc=mu_predict, scale=std_predict)
    log_prob = m.log_prob(z_target).sum(dim=-1)
    loss = -torch.logsumexp(log_weights_predict + log_prob, dim=1).mean()

    return loss


class MDNRNN(nn.Module):
    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        z_size: int,
        hidden_size: int = 256,
        predict_reward: bool = True,
        predict_done: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.a_size = action_space.n
        self.predict_reward = predict_reward
        self.predict_done = predict_done
        self.hidden_size = hidden_size

        self.gru = nn.GRU(
            input_size=self.a_size + z_size,
            hidden_size=hidden_size,
            num_layers=3,
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
        self, mu: torch.Tensor, log_std: torch.Tensor, log_weights: torch.Tensor, deterministic=True
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
        loss = mdn_loss(z_target, mu_predict, log_std_predict, log_weights_predict)

        if self.predict_reward and r_target is not None and r_predict is not None:
            loss += mse_loss(r_predict, r_target.view(-1, 1))

        if self.predict_done and done_target is not None and done_predict is not None:
            loss += binary_cross_entropy_with_logits(
                done_predict, done_target.view(-1, 1)
            )

        return loss

    def initial_state(self) -> torch.Tensor: ...


if __name__ == "__main__":
    action_space = gym.spaces.Discrete(3)
    mdnrnn = MDNRNN(action_space, z_size=256)

    B, T = 2, 4

    # latent states
    z = torch.rand(B, T, 256)
    next_z = torch.rand(B, T, 256)

    # actions (one-hot)
    a = (
        torch.nn.functional.one_hot(
            torch.tensor([action_space.sample() for i in range(B * T)]),
            num_classes=action_space.n,
        )
        .float()
        .view(B, T, -1)
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
    print(loss.item())
