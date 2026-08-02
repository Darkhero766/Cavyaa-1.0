"""Variational bottleneck and reconstruction heads."""

from __future__ import annotations

import torch
from torch import nn


class BetaVAE(nn.Module):
    """β-VAE bottleneck for compact latent representation learning."""

    def __init__(self, hidden_dim: int, latent_dim: int, fragment_dim: int, protein_dim: int) -> None:
        """Initialize posterior and decoders."""
        super().__init__()
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.to_hidden = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.GELU())
        self.fragment_decoder = nn.Linear(hidden_dim, fragment_dim)
        self.protein_decoder = nn.Linear(hidden_dim, protein_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample latent vectors with the reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return latent sample and reconstructions."""
        mu = self.mu(hidden)
        logvar = self.logvar(hidden).clamp(-8, 8)
        z = self.reparameterize(mu, logvar)
        decoded = self.to_hidden(z)
        return {
            "z": z,
            "mu": mu,
            "logvar": logvar,
            "fragment_reconstruction": self.fragment_decoder(decoded),
            "protein_reconstruction": self.protein_decoder(decoded),
        }
