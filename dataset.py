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

    def __len__(self) -> int:
        """Return number of blood draws."""
        return int(len(self.arrays["label"]))

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Return one sample as typed tensors."""
        return {
            "fragment": torch.from_numpy(self.arrays["fragment"][index]),
            "protein": torch.from_numpy(self.arrays["protein"][index]),
            "timeline": torch.from_numpy(self.arrays["timeline"][index]),
            "cancer": torch.tensor(self.arrays["cancer"][index], dtype=torch.long),
            "domain": torch.tensor(self.arrays["domain"][index], dtype=torch.long),
            "label": torch.tensor(self.arrays["label"][index], dtype=torch.float32),
        }


def make_loader(dataset: CavyaaDataset, batch_size: int, shuffle: bool, num_workers: int = 0) -> DataLoader:
    """Create a dataloader with pinned memory when CUDA is available."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=torch.cuda.is_available())
