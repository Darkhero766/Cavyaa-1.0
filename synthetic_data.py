"""Synthetic longitudinal cohort generation for CAVYAA 1.0.

The generator creates algorithm-development data only. It combines simplified
biological motifs--post-operative decay, recurrence growth, inflammation,
patient random effects, sequencer batch effects, measurement noise, and missing
values--without claiming clinical validity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from config import DataConfig, FEATURE_GROUPS
from utils import ensure_dir


@dataclass
class SyntheticTemplates:
    """Latent templates controlling synthetic cancer and batch signatures."""

    cancer_fragment: np.ndarray
    cancer_protein: np.ndarray
    inflammation_fragment: np.ndarray
    inflammation_protein: np.ndarray
    sequencer_fragment: np.ndarray
    sequencer_protein: np.ndarray


def _make_templates(config: DataConfig, rng: np.random.Generator) -> SyntheticTemplates:
    """Construct smooth feature templates for cancers, inflammation, and batches."""
    cancer_fragment = rng.normal(0.0, 0.7, (len(config.cancer_types), config.fragment_dim))
    cancer_protein = rng.normal(0.0, 0.7, (len(config.cancer_types), config.protein_dim))
    grid_f = np.linspace(0, 2 * np.pi, config.fragment_dim)
    grid_p = np.linspace(0, 2 * np.pi, config.protein_dim)
    inflammation_fragment = 0.55 * np.sin(grid_f) + rng.normal(0, 0.08, config.fragment_dim)
    inflammation_protein = 0.55 * np.cos(grid_p) + rng.normal(0, 0.08, config.protein_dim)
    sequencer_fragment = rng.normal(0, 0.22, (len(config.sequencers), config.fragment_dim))
    sequencer_protein = rng.normal(0, 0.12, (len(config.sequencers), config.protein_dim))
    return SyntheticTemplates(
        cancer_fragment=cancer_fragment,
        cancer_protein=cancer_protein,
        inflammation_fragment=inflammation_fragment,
        inflammation_protein=inflammation_protein,
        sequencer_fragment=sequencer_fragment,
        sequencer_protein=sequencer_protein,
    )


def _patient_recurrence_probability(cancer_index: int, burden: float, immune: float) -> float:
    """Return a synthetic recurrence probability for one patient."""
    baseline = np.array([0.28, 0.34, 0.30, 0.24, 0.26])[cancer_index]
    logit = np.log(baseline / (1 - baseline)) + 0.75 * burden - 0.35 * immune
    return float(1 / (1 + np.exp(-logit)))


def generate_synthetic_cohort(config: DataConfig | None = None) -> pd.DataFrame:
    """Generate a complete synthetic longitudinal monitoring table.

    Returns one row per blood draw with patient identifiers, cancer type,
    sequencer, weeks post operation, recurrence label, and feature columns.
    """
    cfg = config or DataConfig()
    rng = np.random.default_rng(cfg.seed)
    templates = _make_templates(cfg, rng)
    rows: List[Dict[str, float | int | str]] = []

    for patient_idx in range(cfg.n_patients):
        cancer_index = int(rng.integers(0, len(cfg.cancer_types)))
        sequencer_index = int(rng.integers(0, len(cfg.sequencers)))
        n_draws = int(rng.integers(cfg.min_draws, cfg.max_draws + 1))
        weeks = np.sort(rng.choice(np.arange(1, 105), size=n_draws, replace=False))
        burden = float(rng.normal(0.0, 1.0))
        immune = float(rng.normal(0.0, 1.0))
        recurrence = int(rng.random() < _patient_recurrence_probability(cancer_index, burden, immune))
        recurrence_week = float(rng.uniform(22, 88)) if recurrence else 140.0
        patient_fragment = rng.normal(0, 0.33, cfg.fragment_dim)
        patient_protein = rng.normal(0, 0.33, cfg.protein_dim)
        inflammation_peak = float(rng.uniform(2, 14))

        for draw_number, week in enumerate(weeks):
            decay = np.exp(-week / rng.uniform(9, 18))
            growth = recurrence * max(0.0, week - recurrence_week) / 38.0
            growth = min(growth, 2.8)
            inflammation = np.exp(-((week - inflammation_peak) ** 2) / (2 * 9.0**2))
            tumor_signal = (0.95 * decay + 1.15 * growth + 0.18 * burden)
            frag = (
                tumor_signal * templates.cancer_fragment[cancer_index]
                + 0.5 * inflammation * templates.inflammation_fragment
                + templates.sequencer_fragment[sequencer_index]
                + patient_fragment
                + rng.normal(0, 0.34, cfg.fragment_dim)
            )
            prot = (
                0.85 * tumor_signal * templates.cancer_protein[cancer_index]
                + 0.75 * inflammation * templates.inflammation_protein
                + templates.sequencer_protein[sequencer_index]
                + patient_protein
                + rng.normal(0, 0.31, cfg.protein_dim)
            )
            row: Dict[str, float | int | str] = {
                "patient_id": f"CAVYAA-{patient_idx:05d}",
                "draw_number": draw_number,
                "weeks_post_operation": float(week),
                "cancer_type": cfg.cancer_types[cancer_index],
                "cancer_index": cancer_index,
                "sequencer": cfg.sequencers[sequencer_index],
                "sequencer_index": sequencer_index,
                "recurrence_label": recurrence,
                "synthetic_tumor_signal": float(tumor_signal),
                "synthetic_inflammation": float(inflammation),
            }
            row.update({name: float(value) for name, value in zip(FEATURE_GROUPS["fragmentomics"], frag)})
            row.update({name: float(value) for name, value in zip(FEATURE_GROUPS["proteomics"], prot)})
            rows.append(row)

    frame = pd.DataFrame(rows)
    feature_cols = FEATURE_GROUPS["fragmentomics"] + FEATURE_GROUPS["proteomics"]
    mask = rng.random((len(frame), len(feature_cols))) < cfg.missing_rate
    frame.loc[:, feature_cols] = frame.loc[:, feature_cols].mask(mask)
    return frame


def load_or_generate(config: DataConfig | None = None, force: bool = False) -> pd.DataFrame:
    """Load a cached cohort when possible; otherwise generate and cache it."""
    cfg = config or DataConfig()
    if cfg.cache_path.exists() and not force:
        return pd.read_parquet(cfg.cache_path)
    frame = generate_synthetic_cohort(cfg)
    ensure_dir(cfg.cache_path.parent)
    try:
        frame.to_parquet(cfg.cache_path, index=False)
    except Exception:
        csv_path = Path(str(cfg.cache_path).replace(".parquet", ".csv"))
        frame.to_csv(csv_path, index=False)
    return frame


if __name__ == "__main__":
    df = load_or_generate(force=True)
    print(df.shape)
