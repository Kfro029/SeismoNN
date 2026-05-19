from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader


def evaluate_classifier(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    labels: list[int] | None = None,
) -> dict[str, Any]:
    """Evaluate classifier on a dataloader.

    Returns:
        Dictionary with loss, accuracy, macro_f1 and confusion_matrix.
    """
    model.eval()

    total_loss = 0.0
    total_samples = 0

    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            batch_size = y.size(0)
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size

            predictions = torch.argmax(logits, dim=1)

            y_true.extend(y.detach().cpu().numpy().tolist())
            y_pred.extend(predictions.detach().cpu().numpy().tolist())

    if total_samples == 0:
        raise ValueError("Cannot evaluate on an empty dataloader.")

    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    loss_value = total_loss / total_samples

    return {
        "loss": loss_value,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
        "y_true": np.asarray(y_true),
        "y_pred": np.asarray(y_pred),
    }
