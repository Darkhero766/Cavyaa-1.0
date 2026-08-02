"""Run inference with a trained CAVYAA checkpoint on synthetic held-out draws."""

from __future__ import annotations

import argparse
import importlib.util
from typing import Iterable


def require_dependencies(modules: Iterable[str]) -> None:
    """Fail fast before importing project modules that require scientific packages."""
    missing = [module for module in modules if importlib.util.find_spec(module) is None]
    if missing:
        raise ModuleNotFoundError(
            "Missing required dependencies: "
            + ", ".join(missing)
            + ". Install them with `pip install -r requirements.txt`."
        )


def main() -> None:
    """Load a checkpoint and print example synthetic recurrence probabilities."""
    require_dependencies(("numpy", "pandas", "sklearn", "torch"))
    import torch

    from config import ExperimentConfig
    from dataset import CavyaaDataset, make_loader
    from evaluation import predict
    from model import CavyaaModel
    from preprocessing import Preprocessor, split_by_patient
    from synthetic_data import load_or_generate
    from utils import get_device

    parser = argparse.ArgumentParser(description="CAVYAA synthetic inference")
    parser.add_argument("--checkpoint", default="artifacts/checkpoints/best_model.pt")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    config = ExperimentConfig()
    frame = load_or_generate(config.data)
    train_frame, _, test_frame = split_by_patient(
        frame,
        config.train.val_fraction,
        config.train.test_fraction,
        config.train.seed,
    )
    arrays = Preprocessor.fit(train_frame).transform(test_frame)
    loader = make_loader(CavyaaDataset(arrays), config.train.batch_size, False)
    device = get_device()
    model = CavyaaModel(config.model).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    labels, probabilities, _ = predict(model, loader, device)
    for idx, (label, probability) in enumerate(zip(labels[: args.limit], probabilities[: args.limit])):
        print(f"draw={idx:03d} synthetic_label={int(label)} recurrence_probability={probability:.4f}")


if __name__ == "__main__":
    main()
