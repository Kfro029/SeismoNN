from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader

from seismonn.data.multitask_dataset import RegressionTargetScaler


def compute_regression_metrics(
    y_true_normalized: np.ndarray,
    y_pred_normalized: np.ndarray,
    target_scaler: RegressionTargetScaler,
) -> dict[str, Any]:
    """Compute regression metrics in original physical units."""
    y_true = target_scaler.inverse_transform(y_true_normalized)
    y_pred = target_scaler.inverse_transform(y_pred_normalized)

    absolute_error = np.abs(y_pred - y_true)
    squared_error = (y_pred - y_true) ** 2

    metrics: dict[str, Any] = {
        "regression_mae_mean": float(absolute_error.mean()),
        "regression_rmse_mean": float(np.sqrt(squared_error.mean())),
        "per_target": {},
    }

    for target_index, target_name in enumerate(target_scaler.target_columns):
        target_abs_error = absolute_error[:, target_index]
        target_squared_error = squared_error[:, target_index]

        metrics["per_target"][target_name] = {
            "mae": float(target_abs_error.mean()),
            "rmse": float(np.sqrt(target_squared_error.mean())),
        }

    return metrics


def compute_multitask_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    classification_criterion: nn.Module,
    regression_criterion: nn.Module,
    regression_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute weighted classification + regression loss."""
    class_loss = classification_criterion(
        outputs["logits"],
        targets["class_id"],
    )

    regression_loss = regression_criterion(
        outputs["regression"],
        targets["regression"],
    )

    total_loss = class_loss + regression_loss_weight * regression_loss

    return total_loss, {
        "classification_loss": float(class_loss.item()),
        "regression_loss": float(regression_loss.item()),
        "total_loss": float(total_loss.item()),
    }


def train_multitask_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    classification_criterion: nn.Module,
    regression_criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    regression_loss_weight: float,
) -> dict[str, float]:
    """Train multi-task model for one epoch."""
    model.train()

    total_loss = 0.0
    total_classification_loss = 0.0
    total_regression_loss = 0.0
    total_samples = 0
    correct = 0

    for x, targets in dataloader:
        x = x.to(device)
        class_id = targets["class_id"].to(device)
        regression = targets["regression"].to(device)

        targets_on_device = {
            "class_id": class_id,
            "regression": regression,
        }

        optimizer.zero_grad(set_to_none=True)

        outputs = model(x)
        loss, loss_parts = compute_multitask_loss(
            outputs=outputs,
            targets=targets_on_device,
            classification_criterion=classification_criterion,
            regression_criterion=regression_criterion,
            regression_loss_weight=regression_loss_weight,
        )

        loss.backward()
        optimizer.step()

        batch_size = x.size(0)
        total_samples += batch_size

        total_loss += loss_parts["total_loss"] * batch_size
        total_classification_loss += loss_parts["classification_loss"] * batch_size
        total_regression_loss += loss_parts["regression_loss"] * batch_size

        predictions = torch.argmax(outputs["logits"], dim=1)
        correct += int((predictions == class_id).sum().item())

    if total_samples == 0:
        raise ValueError("Cannot train on an empty dataloader.")

    return {
        "loss": total_loss / total_samples,
        "classification_loss": total_classification_loss / total_samples,
        "regression_loss": total_regression_loss / total_samples,
        "classification_accuracy": correct / total_samples,
    }


def evaluate_multitask(
    model: nn.Module,
    dataloader: DataLoader,
    classification_criterion: nn.Module,
    regression_criterion: nn.Module,
    device: torch.device,
    target_scaler: RegressionTargetScaler,
    regression_loss_weight: float,
    labels: list[int] | None = None,
) -> dict[str, Any]:
    """Evaluate multi-task model."""
    model.eval()

    total_loss = 0.0
    total_classification_loss = 0.0
    total_regression_loss = 0.0
    total_samples = 0

    y_true_class: list[int] = []
    y_pred_class: list[int] = []
    y_true_regression: list[list[float]] = []
    y_pred_regression: list[list[float]] = []

    with torch.no_grad():
        for x, targets in dataloader:
            x = x.to(device)
            class_id = targets["class_id"].to(device)
            regression = targets["regression"].to(device)

            targets_on_device = {
                "class_id": class_id,
                "regression": regression,
            }

            outputs = model(x)
            loss, loss_parts = compute_multitask_loss(
                outputs=outputs,
                targets=targets_on_device,
                classification_criterion=classification_criterion,
                regression_criterion=regression_criterion,
                regression_loss_weight=regression_loss_weight,
            )

            batch_size = x.size(0)
            total_samples += batch_size

            total_loss += float(loss.item()) * batch_size
            total_classification_loss += loss_parts["classification_loss"] * batch_size
            total_regression_loss += loss_parts["regression_loss"] * batch_size

            predictions = torch.argmax(outputs["logits"], dim=1)

            y_true_class.extend(class_id.detach().cpu().numpy().tolist())
            y_pred_class.extend(predictions.detach().cpu().numpy().tolist())
            y_true_regression.extend(regression.detach().cpu().numpy().tolist())
            y_pred_regression.extend(
                outputs["regression"].detach().cpu().numpy().tolist()
            )

    if total_samples == 0:
        raise ValueError("Cannot evaluate on an empty dataloader.")

    if labels is None:
        labels = sorted(set(y_true_class) | set(y_pred_class))

    y_true_regression_array = np.asarray(y_true_regression, dtype=np.float32)
    y_pred_regression_array = np.asarray(y_pred_regression, dtype=np.float32)

    regression_metrics = compute_regression_metrics(
        y_true_normalized=y_true_regression_array,
        y_pred_normalized=y_pred_regression_array,
        target_scaler=target_scaler,
    )

    return {
        "loss": total_loss / total_samples,
        "classification_loss": total_classification_loss / total_samples,
        "regression_loss": total_regression_loss / total_samples,
        "classification_accuracy": float(accuracy_score(y_true_class, y_pred_class)),
        "classification_balanced_accuracy": float(
            balanced_accuracy_score(y_true_class, y_pred_class)
        ),
        "classification_macro_precision": float(
            precision_score(
                y_true_class,
                y_pred_class,
                average="macro",
                zero_division=0,
            )
        ),
        "classification_macro_recall": float(
            recall_score(
                y_true_class,
                y_pred_class,
                average="macro",
                zero_division=0,
            )
        ),
        "classification_macro_f1": float(
            f1_score(
                y_true_class,
                y_pred_class,
                average="macro",
                zero_division=0,
            )
        ),
        "confusion_matrix": confusion_matrix(
            y_true_class,
            y_pred_class,
            labels=labels,
        ),
        **regression_metrics,
    }
