"""Cross-modal fusion modules for CAVYAA."""

from __future__ import annotations

import torch
from torch import nn

from layers import ResidualBlock


class CrossModalFusion(nn.Module):
    """Attention-gated fusion over fragment, protein, timeline, and cancer streams."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        """Initialize modality attention and residual mixer."""
        super().__init__()
        self.score = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(), nn.Linear(hidden_dim // 2, 1))
        self.mixer = nn.Sequential(nn.Linear(hidden_dim * 4, hidden_dim), nn.GELU(), ResidualBlock(hidden_dim, dropout))

    def forward(self, fragment: torch.Tensor, protein: torch.Tensor, timeline: torch.Tensor, cancer: torch.Tensor) -> torch.Tensor:
        """Fuse modality embeddings into a single representation."""
        stacked = torch.stack([fragment, protein, timeline, cancer], dim=1)
        weights = torch.softmax(self.score(stacked), dim=1)
        attended = (weights * stacked).sum(dim=1)
        concatenated = torch.cat([fragment, protein, timeline, cancer], dim=-1)
        return attended + self.mixer(concatenated)
