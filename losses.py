"""Loss functions and annealing schedules for CAVYAA training."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from config import LossConfig


def kl_anneal(epoch: int, config: LossConfig) -> float:
    """Linearly anneal β for the VAE KL term."""
    return config.max_beta * min(1.0, (epoch + 1) / max(1, config.kl_warmup_epochs))


def cavyaa_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    beta: float,
    config: LossConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute supervised, reconstruction, KL, and domain-adversarial losses."""
    pos_weight = torch.tensor(config.positive_class_weight, device=batch["label"].device)
    cls = F.binary_cross_entropy_with_logits(outputs["recurrence_logit"], batch["label"], pos_weight=pos_weight)
    recon = F.mse_loss(outputs["fragment_reconstruction"], batch["fragment"]) + F.mse_loss(
        outputs["protein_reconstruction"], batch["protein"]
    )
    kl = -0.5 * torch.mean(1 + outputs["logvar"] - outputs["mu"].pow(2) - outputs["logvar"].exp())
    domain = F.cross_entropy(outputs["domain_logit"], batch["domain"])
    total = cls + config.reconstruction_weight * recon + beta * kl + config.domain_weight * domain
    return total, {"loss": float(total.detach()), "classification": float(cls.detach()), "reconstruction": float(recon.detach()), "kl": float(kl.detach()), "domain": float(domain.detach())}
