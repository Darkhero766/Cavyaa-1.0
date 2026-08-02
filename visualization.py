"""Visualization utilities for training diagnostics and representation analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay, confusion_matrix

from config import VisualizationConfig
from utils import ensure_dir


def plot_history(history: List[Dict[str, float]], config: VisualizationConfig) -> Path:
    """Plot training and validation loss curves."""
    ensure_dir(config.output_dir)
    path = config.output_dir / "loss_curves.png"
    plt.figure(figsize=(7, 4))
    plt.plot([row["train_loss"] for row in history], label="train")
    plt.plot([row["val_loss"] for row in history], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=config.dpi)
    plt.close()
    return path


def plot_classification(labels: np.ndarray, probabilities: np.ndarray, config: VisualizationConfig) -> List[Path]:
    """Create ROC, PR, calibration, and confusion matrix plots from model predictions."""
    ensure_dir(config.output_dir)
    paths: List[Path] = []
    for name, display in [("roc.png", RocCurveDisplay.from_predictions), ("pr.png", PrecisionRecallDisplay.from_predictions)]:
        path = config.output_dir / name
        display(labels, probabilities)
        plt.tight_layout()
        plt.savefig(path, dpi=config.dpi)
        plt.close()
        paths.append(path)
    prob_true, prob_pred = calibration_curve(labels, probabilities, n_bins=10, strategy="uniform")
    path = config.output_dir / "calibration.png"
    plt.figure(figsize=(4, 4))
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("Predicted")
    plt.ylabel("Observed")
    plt.tight_layout()
    plt.savefig(path, dpi=config.dpi)
    plt.close()
    paths.append(path)
    path = config.output_dir / "confusion_matrix.png"
    ConfusionMatrixDisplay(confusion_matrix(labels, probabilities >= 0.5)).plot()
    plt.tight_layout()
    plt.savefig(path, dpi=config.dpi)
    plt.close()
    paths.append(path)
    return paths


def _embed_latents(latents: np.ndarray, seed: int = 7) -> np.ndarray:
    """Use UMAP when installed; otherwise use t-SNE for a two-dimensional diagnostic."""
    try:
        import umap

        return umap.UMAP(n_components=2, random_state=seed).fit_transform(latents)
    except ModuleNotFoundError:
        return TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=30, random_state=seed).fit_transform(latents)


def plot_latent(latents: np.ndarray, labels: np.ndarray, config: VisualizationConfig) -> Path:
    """Plot a two-dimensional latent-space diagnostic."""
    ensure_dir(config.output_dir)
    n = min(config.max_points, len(latents))
    coords = _embed_latents(latents[:n])
    path = config.output_dir / "latent_space.png"
    plt.figure(figsize=(6, 5))
    plt.scatter(coords[:, 0], coords[:, 1], c=labels[:n], s=6, cmap="coolwarm", alpha=0.7)
    plt.title("Latent space diagnostic")
    plt.tight_layout()
    plt.savefig(path, dpi=config.dpi)
    plt.close()
    return path


def plot_feature_importance(features: np.ndarray, labels: np.ndarray, names: List[str], config: VisualizationConfig) -> Path:
    """Estimate permutation importance with a logistic probe on synthetic features."""
    ensure_dir(config.output_dir)
    n = min(2500, len(features))
    clf = LogisticRegression(max_iter=300).fit(features[:n], labels[:n])
    result = permutation_importance(clf, features[:n], labels[:n], n_repeats=4, random_state=7)
    order = np.argsort(result.importances_mean)[-20:]
    path = config.output_dir / "feature_importance.png"
    plt.figure(figsize=(7, 5))
    plt.barh(np.array(names)[order], result.importances_mean[order])
    plt.tight_layout()
    plt.savefig(path, dpi=config.dpi)
    plt.close()
    return path
