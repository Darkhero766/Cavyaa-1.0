"""Feature encoders for fragmentomics, proteomics, cancer, and timeline inputs."""

from __future__ import annotations

import torch
from torch import nn

from layers import ResidualBlock


class NumericEncoder(nn.Module):
    """Residual MLP encoder for dense numeric modalities."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        """Create a numeric encoder."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), ResidualBlock(hidden_dim, dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode numeric features."""
        return self.net(x)


class TimelineEncoder(nn.Module):
    """Encoder for weeks post operation and draw index."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        """Create timeline encoder."""
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, hidden_dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode timeline covariates."""
        return self.net(x)


class CancerEmbedding(nn.Module):
    """Embedding layer for tumor type identity."""

    def __init__(self, n_cancers: int, embedding_dim: int, hidden_dim: int) -> None:
        """Create cancer embedding projection."""
        super().__init__()
        self.embedding = nn.Embedding(n_cancers, embedding_dim)
        self.projection = nn.Linear(embedding_dim, hidden_dim)

    def forward(self, cancer: torch.Tensor) -> torch.Tensor:
        """Embed cancer type indices."""
        return self.projection(self.embedding(cancer))
