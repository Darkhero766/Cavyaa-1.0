"""Preprocessing utilities for patient-wise splitting and feature scaling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import FEATURE_GROUPS


@dataclass
class Preprocessor:
    """Impute missing features and standardize continuous model inputs."""

    fragment_columns: List[str]
    protein_columns: List[str]
    fragment_scaler: StandardScaler
    protein_scaler: StandardScaler
    timeline_scaler: StandardScaler
    medians: Dict[str, float]

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> "Preprocessor":
        """Fit scalers and median imputers on a training frame."""
        fragment_columns = FEATURE_GROUPS["fragmentomics"]
        protein_columns = FEATURE_GROUPS["proteomics"]
        all_features = fragment_columns + protein_columns
        medians = frame[all_features].median().to_dict()
        filled = frame.copy()
        filled[all_features] = filled[all_features].fillna(medians)
        fragment_scaler = StandardScaler().fit(filled[fragment_columns])
        protein_scaler = StandardScaler().fit(filled[protein_columns])
        timeline_scaler = StandardScaler().fit(filled[["weeks_post_operation", "draw_number"]])
        return cls(fragment_columns, protein_columns, fragment_scaler, protein_scaler, timeline_scaler, medians)

    def transform(self, frame: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Transform a frame into arrays consumed by ``CavyaaDataset``."""
        all_features = self.fragment_columns + self.protein_columns
        filled = frame.copy()
        filled[all_features] = filled[all_features].fillna(self.medians)
        return {
            "fragment": self.fragment_scaler.transform(filled[self.fragment_columns]).astype("float32"),
            "protein": self.protein_scaler.transform(filled[self.protein_columns]).astype("float32"),
            "timeline": self.timeline_scaler.transform(filled[["weeks_post_operation", "draw_number"]]).astype("float32"),
            "cancer": filled["cancer_index"].to_numpy("int64"),
            "domain": filled["sequencer_index"].to_numpy("int64"),
            "label": filled["recurrence_label"].to_numpy("float32"),
            "patient_id": filled["patient_id"].to_numpy(str),
        }


def split_by_patient(
    frame: pd.DataFrame,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split rows by patient id to prevent longitudinal leakage."""
    patient_table = frame.groupby("patient_id")["recurrence_label"].max().reset_index()
    train_ids, test_ids = train_test_split(
        patient_table["patient_id"],
        test_size=test_fraction,
        random_state=seed,
        stratify=patient_table["recurrence_label"],
    )
    remaining = patient_table[patient_table["patient_id"].isin(train_ids)]
    relative_val = val_fraction / (1.0 - test_fraction)
    train_ids, val_ids = train_test_split(
        remaining["patient_id"],
        test_size=relative_val,
        random_state=seed + 1,
        stratify=remaining["recurrence_label"],
    )
    return (
        frame[frame["patient_id"].isin(train_ids)].reset_index(drop=True),
        frame[frame["patient_id"].isin(val_ids)].reset_index(drop=True),
        frame[frame["patient_id"].isin(test_ids)].reset_index(drop=True),
    )


def feature_columns() -> Sequence[str]:
    """Return all model feature column names."""
    return FEATURE_GROUPS["fragmentomics"] + FEATURE_GROUPS["proteomics"]
