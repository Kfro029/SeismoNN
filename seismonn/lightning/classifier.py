from __future__ import annotations

from typing import Any

import lightning as L
import torch
from torch import nn
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

from seismonn.models.factory import create_model


class SeismoClassifierLightningModule(L.LightningModule):
    """LightningModule for SeismoNN classification models."""

    def __init__(
        self,
        model_config: dict[str, Any],
        optimizer_config: dict[str, Any],
    ) -> None:
        super().__init__()

        self.model_config = model_config
        self.optimizer_config = optimizer_config

        self.model = create_model(model_config)
        self.criterion = nn.CrossEntropyLoss()

        num_classes = int(model_config.get("num_classes", 3))

        self.train_accuracy = MulticlassAccuracy(
            num_classes=num_classes,
            average="micro",
        )
        self.val_accuracy = MulticlassAccuracy(
            num_classes=num_classes,
            average="micro",
        )

        self.val_balanced_accuracy = MulticlassAccuracy(
            num_classes=num_classes,
            average="macro",
        )
        self.val_macro_precision = MulticlassPrecision(
            num_classes=num_classes,
            average="macro",
            zero_division=0,
        )
        self.val_macro_recall = MulticlassRecall(
            num_classes=num_classes,
            average="macro",
            zero_division=0,
        )
        self.val_macro_f1 = MulticlassF1Score(
            num_classes=num_classes,
            average="macro",
            zero_division=0,
        )

        self.save_hyperparameters(
            {
                "model_config": model_config,
                "optimizer_config": optimizer_config,
            }
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model(batch)

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        del batch_idx

        features, targets = batch
        logits = self(features)
        loss = self.criterion(logits, targets)

        predictions = torch.argmax(logits, dim=1)

        self.train_accuracy.update(predictions, targets)

        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train_accuracy",
            self.train_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        del batch_idx

        features, targets = batch
        logits = self(features)
        loss = self.criterion(logits, targets)

        predictions = torch.argmax(logits, dim=1)

        self.val_accuracy.update(predictions, targets)
        self.val_balanced_accuracy.update(predictions, targets)
        self.val_macro_precision.update(predictions, targets)
        self.val_macro_recall.update(predictions, targets)
        self.val_macro_f1.update(predictions, targets)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val_accuracy",
            self.val_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val_balanced_accuracy",
            self.val_balanced_accuracy,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_macro_precision",
            self.val_macro_precision,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_macro_recall",
            self.val_macro_recall,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_macro_f1",
            self.val_macro_f1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer_name = str(self.optimizer_config.get("name", "adamw")).lower()
        learning_rate = float(self.optimizer_config.get("lr", 3e-4))
        weight_decay = float(self.optimizer_config.get("weight_decay", 0.0))

        if optimizer_name == "adam":
            return torch.optim.Adam(
                self.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )

        if optimizer_name == "adamw":
            return torch.optim.AdamW(
                self.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )

        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
