"""Reusable neural-network layers for CAVYAA."""

from __future__ import annotations

import torch
from torch import nn
from torch.autograd import Function


class ResidualBlock(nn.Module):
    """Layer-normalized residual MLP block."""

    def __init__(self, dim: int, dropout: float) -> None:
        """Initialize the residual transformation."""
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual transformation."""
        return x + self.net(x)


class GradientReversalFunction(Function):
    """Autograd function that reverses gradients during backpropagation."""

    @staticmethod
    def forward(ctx: object, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        """Return the input unchanged while storing reversal strength."""
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx: object, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Multiply gradients by negative lambda."""
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """Module wrapper for gradient reversal."""

    def __init__(self) -> None:
        """Initialize with neutral scheduling state."""
        super().__init__()
        self.lambda_ = 0.0

    def set_lambda(self, value: float) -> None:
        """Update the reversal strength."""
        self.lambda_ = float(value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gradient reversal."""
        return GradientReversalFunction.apply(x, self.lambda_)
