"""CAVYAA neural architecture."""

from __future__ import annotations

import torch
from torch import nn

from config import ModelConfig
from domain_adaptation import DomainClassifier
from encoders import CancerEmbedding, NumericEncoder, TimelineEncoder
from fusion import CrossModalFusion
from layers import ResidualBlock
from vae import BetaVAE


class CavyaaModel(nn.Module):
    """Tumor-informed and timeline-aware MRD tracking research model."""

    def __init__(self, config: ModelConfig) -> None:
        """Initialize all encoders, fusion, VAE, domain, and recurrence heads."""
        super().__init__()
        self.fragment_encoder = NumericEncoder(config.fragment_dim, config.hidden_dim, config.dropout)
        self.protein_encoder = NumericEncoder(config.protein_dim, config.hidden_dim, config.dropout)
        self.timeline_encoder = TimelineEncoder(config.hidden_dim, config.dropout)
        self.cancer_embedding = CancerEmbedding(config.n_cancers, config.embedding_dim, config.hidden_dim)
        self.fusion = CrossModalFusion(config.hidden_dim, config.dropout)
        self.residual = nn.Sequential(ResidualBlock(config.hidden_dim, config.dropout), ResidualBlock(config.hidden_dim, config.dropout))
        self.vae = BetaVAE(config.hidden_dim, config.latent_dim, config.fragment_dim, config.protein_dim)
        self.domain_classifier = DomainClassifier(config.latent_dim, config.hidden_dim, config.n_domains, config.dropout)
        self.recurrence_head = nn.Sequential(
            nn.LayerNorm(config.latent_dim),
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        """Initialize linear and embedding layers with stable scales."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            nn.Linear(config.latent_dim, config.hidden_dim), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(config.hidden_dim, 1)
        )

    def set_grl_lambda(self, value: float) -> None:
        """Update adversarial domain strength."""
        self.domain_classifier.set_lambda(value)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run a forward pass for one mini-batch."""
        fragment = self.fragment_encoder(batch["fragment"])
        protein = self.protein_encoder(batch["protein"])
        timeline = self.timeline_encoder(batch["timeline"])
        cancer = self.cancer_embedding(batch["cancer"])
        fused = self.residual(self.fusion(fragment, protein, timeline, cancer))
        latent = self.vae(fused)
        return {
            **latent,
            "recurrence_logit": self.recurrence_head(latent["z"]).squeeze(-1),
            "domain_logit": self.domain_classifier(latent["z"]),
        }
