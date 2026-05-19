from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SeismoDataset(Dataset):
    """Dataset for SeismoNN seismic .npy samples.

    Expected metadata columns:
    - path
    - split
    - crack_count
    - class_id
    """

    def __init__(
        self,
        metadata_path: str | Path,
        split: str,
        data_root: str | Path = ".",
        normalize: bool = True,
        expected_channels: int = 2,
    ) -> None:
        self.metadata_path = Path(metadata_path)
        self.data_root = Path(data_root)
        self.split = split
        self.normalize = normalize
        self.expected_channels = expected_channels

        metadata = pd.read_csv(self.metadata_path)

        required_columns = {"path", "split", "crack_count", "class_id"}
        missing_columns = required_columns - set(metadata.columns)

        if missing_columns:
            raise ValueError(
                f"metadata.csv is missing required columns: {sorted(missing_columns)}"
            )

        metadata = metadata[metadata["split"] == split].reset_index(drop=True)

        if len(metadata) == 0:
            raise ValueError(f"No samples found for split={split!r}")

        self.metadata = metadata

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
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

        # Make writable contiguous copy after mmap loading.
        x = np.array(x, dtype=np.float32, copy=True)

        if self.normalize:
            mean = float(x.mean())
            std = float(x.std()) + 1e-8
            x = (x - mean) / std

        y = int(row["class_id"])

        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)
