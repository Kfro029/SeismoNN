from __future__ import annotations

from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader

from seismonn.data.dataset import SeismoDataset


class SeismoDataModule(L.LightningDataModule):
    """LightningDataModule for SeismoNN classification task."""

    def __init__(
        self,
        metadata_path: str | Path,
        data_root: str | Path = ".",
        train_split: str = "train",
        val_split: str = "val",
        batch_size: int = 16,
        num_workers: int = 2,
        normalize: bool = True,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()

        self.metadata_path = Path(metadata_path)
        self.data_root = Path(data_root)
        self.train_split = train_split
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normalize = normalize
        self.pin_memory = pin_memory

        self.train_dataset: SeismoDataset | None = None
        self.val_dataset: SeismoDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Create train and validation datasets."""
        if stage in {"fit", None}:
            self.train_dataset = SeismoDataset(
                metadata_path=self.metadata_path,
                split=self.train_split,
                data_root=self.data_root,
                normalize=self.normalize,
            )

            self.val_dataset = SeismoDataset(
                metadata_path=self.metadata_path,
                split=self.val_split,
                data_root=self.data_root,
                normalize=self.normalize,
            )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError(
                "train_dataset is not initialized. Call setup('fit') first."
            )

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError(
                "val_dataset is not initialized. Call setup('fit') first."
            )

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
