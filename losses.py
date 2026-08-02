"""Loss functions and annealing schedules for CAVYAA training."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from config import LossConfig
from utils import assert_finite_tensor


def kl_anneal(epoch: int, config: LossConfig) -> float:
    """Linearly anneal β for the VAE KL term."""
    return config.max_beta * min(1.0, (epoch + 1) / max(1, config.kl_warmup_epochs))


def vae_kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Return mean KL divergence from q(z|x) to a unit Gaussian prior."""
    assert_finite_tensor("vae_mu", mu)
    assert_finite_tensor("vae_logvar", logvar)
    return -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())


def cavyaa_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    beta: float,
    config: LossConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute supervised, reconstruction, KL, and domain-adversarial losses."""
    for key, tensor in outputs.items():
        assert_finite_tensor(f"output_{key}", tensor)
    pos_weight = torch.tensor(config.positive_class_weight, device=batch["label"].device)
    cls = F.binary_cross_entropy_with_logits(outputs["recurrence_logit"], batch["label"], pos_weight=pos_weight)
    recon_fragment = F.mse_loss(outputs["fragment_reconstruction"], batch["fragment"])
    recon_protein = F.mse_loss(outputs["protein_reconstruction"], batch["protein"])
    recon = recon_fragment + recon_protein
    kl = vae_kl_divergence(outputs["mu"], outputs["logvar"])
    domain = F.cross_entropy(outputs["domain_logit"], batch["domain"])
    total = cls + config.reconstruction_weight * recon + beta * kl + config.domain_weight * domain
    assert_finite_tensor("total_loss", total)
    return total, {
        "loss": float(total.detach()),
        "classification": float(cls.detach()),
        "reconstruction": float(recon.detach()),
        "kl": float(kl.detach()),
        "domain": float(domain.detach()),
    }
