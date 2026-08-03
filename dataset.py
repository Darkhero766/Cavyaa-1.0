"""PyTorch dataset and dataloader helpers for CAVYAA samples."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class CavyaaDataset(Dataset):
    """Dataset representing one synthetic blood draw per item."""

    def __init__(self, arrays: Dict[str, np.ndarray]) -> None:
        """Store transformed arrays from the preprocessing stage."""
        self.arrays = arrays
        expected = len(arrays["label"])
        for key in ("fragment", "protein", "timeline", "cancer", "domain"):
            if len(arrays[key]) != expected:
                raise ValueError(f"Array {key} length does not match labels")

    def __len__(self) -> int:
        """Return number of blood draws."""
        return int(len(self.arrays["label"]))

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Return one sample as typed tensors."""
        return {
            "fragment": torch.as_tensor(self.arrays["fragment"][index], dtype=torch.float32),
            "protein": torch.as_tensor(self.arrays["protein"][index], dtype=torch.float32),
            "timeline": torch.as_tensor(self.arrays["timeline"][index], dtype=torch.float32),
            "cancer": torch.as_tensor(self.arrays["cancer"][index], dtype=torch.long),
            "domain": torch.as_tensor(self.arrays["domain"][index], dtype=torch.long),
            "label": torch.as_tensor(self.arrays["label"][index], dtype=torch.float32),
        }


def make_loader(
    dataset: CavyaaDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    persistent_workers: bool = False,
    prefetch_factor: int | None = None,
) -> DataLoader:
    """Create an optimized dataloader with safe worker settings."""
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": persistent_workers and num_workers > 0,
    }
    if num_workers > 0 and prefetch_factor is not None:
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **kwargs)
