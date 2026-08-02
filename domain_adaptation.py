"""Domain-adversarial components for sequencer invariance."""

from __future__ import annotations

import math

import torch
from torch import nn

from layers import GradientReversalLayer


class DomainClassifier(nn.Module):
    """Sequencer classifier attached through a gradient reversal layer."""

    def __init__(self, latent_dim: int, hidden_dim: int, n_domains: int, dropout: float) -> None:
        """Create the adversarial classifier."""
        super().__init__()
        self.grl = GradientReversalLayer()
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, n_domains)
        )

    def set_lambda(self, value: float) -> None:
        """Set gradient reversal strength."""
        self.grl.set_lambda(value)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Predict sequencing domain from latent vectors."""
        return self.classifier(self.grl(z))


def grl_schedule(epoch: int, max_epochs: int) -> float:
    """Smoothly increase GRL strength using the DANN schedule."""
    progress = min(1.0, max(0.0, epoch / max(1, max_epochs)))
    return float(2.0 / (1.0 + math.exp(-10 * progress)) - 1.0)
