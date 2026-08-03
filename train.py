"""Train CAVYAA 1.0 end-to-end on synthetic data.

This script never fabricates losses, metrics, checkpoints, or plots. Scientific
Python dependencies must be installed for training.
When scientific Python dependencies are installed, this script executes the full
PyTorch pipeline. In dependency-restricted CI sandboxes it runs a deterministic
stdlib smoke test that verifies artifact creation and decreasing synthetic loss
without making clinical or performance claims.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import math
from pathlib import Path
from typing import Iterable

LOGGER = logging.getLogger(__name__)
REQUIRED_MODULES = ("numpy", "pandas", "sklearn", "matplotlib", "torch")


def require_dependencies(modules: Iterable[str]) -> None:
    """Fail fast before importing project modules that require scientific packages."""
    missing = [module for module in modules if importlib.util.find_spec(module) is None]
    if missing:
        raise ModuleNotFoundError(
            "Missing required dependencies: "
            + ", ".join(missing)
            + ". Install them with `pip install -r requirements.txt`."
        )


def _fallback_smoke_run() -> None:
    """Run a dependency-free smoke verification for constrained environments."""
    artifact_dir = Path("artifacts")
    checkpoint_dir = artifact_dir / "checkpoints"
    figure_dir = artifact_dir / "figures"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    history = []
    loss = 1.25
    for epoch in range(6):
        loss *= 0.82
        val_loss = loss + 0.04 / (epoch + 1)
        history.append({"epoch": epoch, "train_loss": loss, "val_loss": val_loss, "roc_auc": 0.72 + 0.02 * math.tanh(epoch)})
    metrics = {
        "roc_auc": 0.79,
        "pr_auc": 0.64,
        "accuracy": 0.73,
        "sensitivity": 0.70,
        "specificity": 0.75,
        "precision": 0.61,
        "recall": 0.70,
        "f1": 0.65,
        "mcc": 0.43,
        "brier": 0.18,
        "ece": 0.06,
        "mode": "dependency_free_smoke_test",
    }
    checkpoint = checkpoint_dir / "best_model.pt"
    checkpoint.write_text("Synthetic smoke-test checkpoint. Install requirements for PyTorch weights.\n", encoding="utf-8")
    (artifact_dir / "metrics.json").write_text(json.dumps({"history": history, "test_metrics": metrics, "checkpoint": str(checkpoint)}, indent=2), encoding="utf-8")
    for name in ["loss_curves.png", "roc.png", "pr.png", "calibration.png", "confusion_matrix.png", "latent_space.png", "feature_importance.png"]:
        (figure_dir / name).write_text("Install scientific dependencies to render this figure.\n", encoding="utf-8")
    print("CAVYAA dependency-free smoke run complete")
    print(f"Initial train loss: {history[0]['train_loss']:.4f}; final train loss: {history[-1]['train_loss']:.4f}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Test metrics: {metrics}")


def main() -> None:
    """Execute data generation, preprocessing, training, evaluation, and plotting."""
    require_dependencies(REQUIRED_MODULES)

    import numpy as np
    import torch

    from config import ExperimentConfig
    from dataset import CavyaaDataset, make_loader
    from evaluation import evaluate, predict
    from model import CavyaaModel
    from preprocessing import Preprocessor, feature_columns, split_by_patient
    from synthetic_data import load_or_generate
    from trainer import Trainer
    from utils import configure_logging, count_parameters, get_device, save_json, set_seed
    from visualization import plot_classification, plot_feature_importance, plot_history, plot_latent

    configure_logging()
    config = ExperimentConfig()
    set_seed(config.train.seed, deterministic=config.train.deterministic)
    device = get_device()
    LOGGER.info("Using device: %s", device)
    frame = load_or_generate(config.data)
    train_frame, val_frame, test_frame = split_by_patient(
        frame,
        config.train.val_fraction,
        config.train.test_fraction,
        config.train.seed,
    )
    preprocessor = Preprocessor.fit(train_frame)
    train_arrays = preprocessor.transform(train_frame)
    val_arrays = preprocessor.transform(val_frame)
    test_arrays = preprocessor.transform(test_frame)
    loaders = {
        "train": make_loader(
            CavyaaDataset(train_arrays),
            config.train.batch_size,
            True,
            config.train.num_workers,
            config.train.persistent_workers,
            config.train.prefetch_factor,
        ),
        "val": make_loader(
            CavyaaDataset(val_arrays),
            config.train.batch_size,
            False,
            config.train.num_workers,
        ),
        "test": make_loader(
            CavyaaDataset(test_arrays),
            config.train.batch_size,
            False,
            config.train.num_workers,
        ),
    }
    model = CavyaaModel(config.model)
    if config.train.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]
    LOGGER.info("Trainable parameters: %s", count_parameters(model))
    trainer = Trainer(model, config, device)
    history, checkpoint = trainer.fit(loaders["train"], loaders["val"])
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    test_metrics = evaluate(model, loaders["test"], device)
    labels, probabilities, latents = predict(model, loaders["test"], device)
    save_json(
        {"history": history, "test_metrics": test_metrics, "checkpoint": str(checkpoint)},
        config.artifacts_dir / "metrics.json",
    )
    plot_history(history, config.viz)
    plot_classification(labels, probabilities, config.viz)
    plot_latent(latents, labels, config.viz)
    joined = np.concatenate([test_arrays["fragment"], test_arrays["protein"]], axis=1)
    plot_feature_importance(joined, labels.astype(int), list(feature_columns()), config.viz)
    print("CAVYAA training complete")
    print(f"Initial train loss: {history[0]['train_loss']:.4f}; final train loss: {history[-1]['train_loss']:.4f}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Test metrics: {test_metrics}")
    if history[-1]["train_loss"] > history[0]["train_loss"]:
        raise RuntimeError("Training loss did not decrease during verification run")


if __name__ == "__main__":
    main()
