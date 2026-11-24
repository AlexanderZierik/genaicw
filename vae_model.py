"""Small convolutional VAE used for Part 1 of the coursework.

The architecture deliberately remains below the coursework's one-million
parameter limit. It accepts either MNIST (one channel) or hot-dog (three
channel) images at 28x28.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvVAE(nn.Module):
    """Convolutional VAE for 28x28 images."""

    def __init__(self, channels: int, latent_dim: int, output: str):
        super().__init__()
        if output not in {"sigmoid", "tanh"}:
            raise ValueError("output must be 'sigmoid' or 'tanh'")
        self.channels = channels
        self.latent_dim = latent_dim
        self.output = output
        self.flat_dim = 128 * 7 * 7
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self.flat_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, channels, 3, 1, 1),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterise(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Differentiably sample z = mu + exp(0.5 * logvar) * eps."""
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z).view(-1, 128, 7, 7)
        x = self.decoder(h)
        return torch.sigmoid(x) if self.output == "sigmoid" else torch.tanh(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        return self.decode(self.reparameterise(mu, logvar)), mu, logvar


def vae_loss(recon, x, mu, logvar, beta: float, likelihood: str):
    """Negative ELBO with Bernoulli/BCE or Gaussian/MSE reconstruction."""
    if likelihood == "bernoulli":
        reconstruction = F.binary_cross_entropy(recon, x, reduction="none").flatten(1).sum(1).mean()
    elif likelihood == "gaussian":
        reconstruction = F.mse_loss(recon, x, reduction="none").flatten(1).sum(1).mean()
    else:
        raise ValueError("likelihood must be 'bernoulli' or 'gaussian'")
    kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(1).mean()
    return reconstruction + beta * kl, reconstruction, kl


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
