"""General utilities for reproducibility, logging, validation, and artifacts."""

from __future__ import annotations

import csv
import importlib.util
"""General utilities for reproducibility, logging, and artifact management."""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

LOGGER = logging.getLogger(__name__)


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python and PyTorch; seed NumPy when it is installed."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ModuleNotFoundError:
        LOGGER.debug("NumPy is unavailable; skipped NumPy seeding.")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = torch.cuda.is_available()
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def ensure_dir(path: Path | str) -> Path:
    """Create a directory if needed and return it as a ``Path``."""
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def configure_logging(level: int = logging.INFO) -> None:
    """Configure compact console logging for command-line scripts."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def get_device() -> torch.device:
    """Return CUDA when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(payload: Mapping[str, Any], path: Path | str) -> None:
def save_json(payload: Dict[str, Any], path: Path | str) -> None:
    """Write a JSON file with stable formatting."""
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def append_csv(row: Mapping[str, Any], path: Path | str) -> None:
    """Append a row to a CSV file, writing the header on first use."""
    target = Path(path)
    ensure_dir(target.parent)
    exists = target.exists()
    with target.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a PyTorch module."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean environment flag."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def assert_finite_tensor(name: str, tensor: torch.Tensor) -> None:
    """Raise ``FloatingPointError`` when a tensor contains NaN or Inf values."""
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"Non-finite values detected in {name}")


def assert_finite_mapping(values: Mapping[str, float]) -> None:
    """Validate that scalar loss components are finite."""
    for name, value in values.items():
        if not torch.isfinite(torch.tensor(float(value))):
            raise FloatingPointError(f"Non-finite scalar detected in {name}: {value}")


def require_dependencies(modules: Iterable[str]) -> None:
    """Fail fast with an actionable error if required modules are unavailable."""
    missing = [module for module in modules if importlib.util.find_spec(module) is None]
    if missing:
        joined = ", ".join(missing)
        raise ModuleNotFoundError(
            f"Missing required dependencies: {joined}. Install them with `pip install -r requirements.txt`."
        )
