"""Evaluation loops for trained CAVYAA models."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from metrics import compute_metrics, expected_calibration_error, sigmoid


@torch.no_grad()
def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return labels, probabilities, and latent means for a dataloader."""
    model.eval()
    logits, labels, latents = [], [], []
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(batch)
        logits.append(outputs["recurrence_logit"].detach().cpu().numpy())
        labels.append(batch["label"].detach().cpu().numpy())
        latents.append(outputs["mu"].detach().cpu().numpy())
    return np.concatenate(labels), sigmoid(np.concatenate(logits)), np.concatenate(latents)


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """Evaluate recurrence prediction and calibration metrics."""
    labels, probabilities, _ = predict(model, loader, device)
    metrics = compute_metrics(labels, probabilities)
    metrics["ece"] = expected_calibration_error(labels, probabilities)
    return metrics
