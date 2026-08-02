"""Configuration objects for the CAVYAA 1.0 research prototype.

CAVYAA is a synthetic-data-only framework for algorithm development in
longitudinal minimal residual disease tracking. Values here intentionally favor
fast reproducible experimentation over clinical realism or deployment use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class DataConfig:
    """Synthetic cohort and feature generation settings."""

    n_patients: int = 10_000
    min_draws: int = 3
    max_draws: int = 8
    fragment_dim: int = 48
    protein_dim: int = 36
    cancer_types: Tuple[str, ...] = (
        "Neuroblastoma",
        "Glioblastoma",
        "Lung Adenocarcinoma",
        "Osteosarcoma",
        "Melanoma",
    )
    sequencers: Tuple[str, ...] = ("NovaSeq", "NextSeq", "MGI", "Element")
    missing_rate: float = 0.055
    seed: int = 2026
    cache_path: Path = Path("artifacts/synthetic_cavyaa.parquet")


@dataclass(frozen=True)
class ModelConfig:
    """Neural-network architecture settings."""

    fragment_dim: int = 48
    protein_dim: int = 36
    hidden_dim: int = 96
    embedding_dim: int = 16
    latent_dim: int = 24
    dropout: float = 0.15
    n_cancers: int = 5
    n_domains: int = 4


@dataclass(frozen=True)
class TrainConfig:
    """Training and optimization settings."""

    epochs: int = 6
    batch_size: int = 512
    learning_rate: float = 2.5e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    early_stopping_patience: int = 4
    num_workers: int = 0
    persistent_workers: bool = False
    prefetch_factor: int | None = None
    warmup_epochs: int = 1
    compile_model: bool = False
    deterministic: bool = False
    seed: int = 2026
    checkpoint_dir: Path = Path("artifacts/checkpoints")
    log_dir: Path = Path("artifacts/tensorboard")
    use_mixed_precision: bool = True
    min_delta: float = 1e-4
    resume_from: Path | None = None
    history_csv: Path = Path("artifacts/training_history.csv")


@dataclass(frozen=True)
class LossConfig:
    """Loss weighting and schedule settings."""

    reconstruction_weight: float = 0.2
    domain_weight: float = 0.08
    max_beta: float = 0.025
    kl_warmup_epochs: int = 4
    grl_warmup_epochs: int = 4
    positive_class_weight: float = 1.5


@dataclass(frozen=True)
class VisualizationConfig:
    """Visualization output settings."""

    output_dir: Path = Path("artifacts/figures")
    dpi: int = 140
    max_points: int = 3000


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level CAVYAA experiment configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    viz: VisualizationConfig = field(default_factory=VisualizationConfig)
    artifacts_dir: Path = Path("artifacts")

    def as_dict(self) -> Dict[str, object]:
        """Return a shallow dictionary useful for logging."""
        return {
            "data": self.data,
            "model": self.model,
            "train": self.train,
            "loss": self.loss,
            "viz": self.viz,
            "artifacts_dir": self.artifacts_dir,
        }


def get_default_config() -> ExperimentConfig:
    """Create the default experiment configuration."""
    return ExperimentConfig()


CANCER_TO_INDEX: Dict[str, int] = {
    cancer: index for index, cancer in enumerate(DataConfig().cancer_types)
}
SEQUENCER_TO_INDEX: Dict[str, int] = {
    sequencer: index for index, sequencer in enumerate(DataConfig().sequencers)
}
FEATURE_GROUPS: Dict[str, List[str]] = {
    "fragmentomics": [f"frag_{i:02d}" for i in range(DataConfig().fragment_dim)],
    "proteomics": [f"prot_{i:02d}" for i in range(DataConfig().protein_dim)],
}
