from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


DEFAULT_REGRESSION_TARGET_COLUMNS = [
    "mean_length",
    "length_spread",
    "mean_angle_deg",
    "angle_spread_deg",
]


@dataclass
class RegressionTargetScaler:
    """Standard scaler for regression targets.

    The scaler is fitted only on train metadata and then reused for validation.
    """

    target_columns: list[str]
    mean: list[float]
    std: list[float]

    @classmethod
    def fit(
        cls,
        metadata: pd.DataFrame,
        target_columns: list[str],
    ) -> RegressionTargetScaler:
        missing_columns = set(target_columns) - set(metadata.columns)

        if missing_columns:
            raise ValueError(
                f"Metadata is missing regression target columns: "
                f"{sorted(missing_columns)}"
            )

        values = metadata[target_columns].to_numpy(dtype=np.float32)

        mean = values.mean(axis=0)
        std = values.std(axis=0)

        std = np.where(std < 1e-8, 1.0, std)

        return cls(
            target_columns=list(target_columns),
            mean=mean.astype(float).tolist(),
            std=std.astype(float).tolist(),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RegressionTargetScaler:
        return cls(
            target_columns=list(data["target_columns"]),
            mean=[float(value) for value in data["mean"]],
            std=[float(value) for value in data["std"]],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_columns": self.target_columns,
            "mean": self.mean,
            "std": self.std,
        }

    def transform(self, values: np.ndarray) -> np.ndarray:
        mean = np.asarray(self.mean, dtype=np.float32)
        std = np.asarray(self.std, dtype=np.float32)

        return (values.astype(np.float32) - mean) / std

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        mean = np.asarray(self.mean, dtype=np.float32)
        std = np.asarray(self.std, dtype=np.float32)

        return values.astype(np.float32) * std + mean


class SeismoMultiTaskDataset(Dataset):
    """Dataset for classification + regression.

    Returns:
        x: Tensor with shape [2, T, R]
        target: dict with:
            class_id: LongTensor scalar
            regression: FloatTensor with normalized regression targets
    """

    def __init__(
        self,
        metadata_path: str | Path,
        split: str,
        data_root: str | Path = ".",
        regression_target_columns: list[str] | None = None,
        target_scaler: RegressionTargetScaler | None = None,
        normalize_input: bool = True,
        expected_channels: int = 2,
    ) -> None:
        self.metadata_path = Path(metadata_path)
        self.data_root = Path(data_root)
        self.split = split
        self.normalize_input = normalize_input
        self.expected_channels = expected_channels

        if regression_target_columns is None:
            regression_target_columns = DEFAULT_REGRESSION_TARGET_COLUMNS

        self.regression_target_columns = list(regression_target_columns)

        metadata = pd.read_csv(self.metadata_path)

        required_columns = {
            "path",
            "split",
            "class_id",
            *self.regression_target_columns,
        }
        missing_columns = required_columns - set(metadata.columns)

        if missing_columns:
            raise ValueError(
                f"metadata.csv is missing required columns: {sorted(missing_columns)}"
            )

        metadata = metadata[metadata["split"] == split].reset_index(drop=True)

        if len(metadata) == 0:
            raise ValueError(f"No samples found for split={split!r}")

        self.metadata = metadata

        if target_scaler is None:
            target_scaler = RegressionTargetScaler.fit(
                metadata=self.metadata,
                target_columns=self.regression_target_columns,
            )

        self.target_scaler = target_scaler

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        row = self.metadata.iloc[idx]

        sample_path = self.data_root / str(row["path"])

        if not sample_path.exists():
            raise FileNotFoundError(f"Sample file does not exist: {sample_path}")

        x = np.load(sample_path, mmap_mode="r")
        x = np.asarray(x, dtype=np.float32)

        if x.ndim != 3:
            raise ValueError(
                f"Expected 3D tensor with shape (C, T, R), got shape={x.shape}"
            )

        if x.shape[0] != self.expected_channels:
            raise ValueError(
                f"Expected {self.expected_channels} channels, got shape={x.shape}"
            )

        x = np.array(x, dtype=np.float32, copy=True)

        if self.normalize_input:
            mean = float(x.mean())
            std = float(x.std()) + 1e-8
            x = (x - mean) / std

        class_id = int(row["class_id"])

        regression_values = row[self.regression_target_columns].to_numpy(
            dtype=np.float32
        )
        regression_values = self.target_scaler.transform(regression_values)

        target = {
            "class_id": torch.tensor(class_id, dtype=torch.long),
            "regression": torch.from_numpy(regression_values.astype(np.float32)),
        }

        return torch.from_numpy(x), target


def fit_regression_target_scaler_from_metadata(
    metadata_path: str | Path,
    train_split: str = "train",
    target_columns: list[str] | None = None,
) -> RegressionTargetScaler:
    """Fit target scaler on train split only."""
    if target_columns is None:
        target_columns = DEFAULT_REGRESSION_TARGET_COLUMNS

    metadata = pd.read_csv(metadata_path)
    train_metadata = metadata[metadata["split"] == train_split].reset_index(drop=True)

    if len(train_metadata) == 0:
        raise ValueError(f"No samples found for train_split={train_split!r}")

    return RegressionTargetScaler.fit(
        metadata=train_metadata,
        target_columns=list(target_columns),
    )
