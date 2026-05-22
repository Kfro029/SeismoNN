from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader

from seismonn.data.multitask_dataset import (
    RegressionTargetScaler,
    SeismoMultiTaskDataset,
)
from seismonn.inference.multitask_predictor import normalize_class_mapping
from seismonn.inference.predictor import load_torch_checkpoint
from seismonn.models.cnn_multitask import SeismoCNNMultiTask
from seismonn.training.multitask import (
    compute_multitask_loss,
    compute_regression_metrics,
)
from seismonn.training.utils import get_device, to_jsonable


def load_multitask_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any], RegressionTargetScaler]:
    """Load multi-task model, checkpoint metadata and target scaler."""
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint file does not exist: {checkpoint_path}. "
            "Train the multi-task model first or provide a valid --checkpoint path."
        )

    checkpoint = load_torch_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    model_config = checkpoint.get("model_config")

    if not isinstance(model_config, dict):
        raise ValueError("Checkpoint must contain dictionary field 'model_config'.")

    model_name = str(model_config.get("name", checkpoint.get("model_name", "")))

    if model_name != "cnn_multitask":
        raise ValueError(
            f"Expected checkpoint with model name 'cnn_multitask', got {model_name!r}."
        )

    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint does not contain 'model_state_dict'.")

    target_scaler_data = checkpoint.get("target_scaler")

    if not isinstance(target_scaler_data, dict):
        raise ValueError(
            "Multi-task checkpoint must contain dictionary field 'target_scaler'."
        )

    target_scaler = RegressionTargetScaler.from_dict(target_scaler_data)

    model = SeismoCNNMultiTask(
        in_channels=int(model_config.get("in_channels", 2)),
        num_classes=int(model_config.get("num_classes", 3)),
        num_regression_targets=int(model_config.get("num_regression_targets", 4)),
        dropout=float(model_config.get("dropout", 0.2)),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint, target_scaler


def build_class_names(class_id_to_crack_count: dict[int, int]) -> list[str]:
    """Build class names sorted by class_id."""
    return [
        str(class_id_to_crack_count[class_id])
        for class_id in sorted(class_id_to_crack_count)
    ]


def evaluate_multitask_checkpoint(
    checkpoint_path: str | Path,
    metadata_path: str | Path,
    split: str = "val",
    data_root: str | Path = ".",
    batch_size: int = 8,
    num_workers: int = 0,
    normalize: bool | None = None,
    device_name: str = "auto",
    regression_loss_weight: float = 1.0,
) -> dict[str, Any]:
    """Evaluate multi-task checkpoint on selected metadata split."""
    device = get_device(device_name)

    model, checkpoint, target_scaler = load_multitask_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    checkpoint_data_config = checkpoint.get("data_config", {})

    if normalize is None:
        if isinstance(checkpoint_data_config, dict):
            normalize = bool(checkpoint_data_config.get("normalize", True))
        else:
            normalize = True

    model_config = checkpoint.get("model_config", {})
    num_classes = int(model_config.get("num_classes", 3))

    class_id_to_crack_count = normalize_class_mapping(
        checkpoint.get("class_id_to_crack_count")
    )

    regression_columns = checkpoint.get("regression_columns")
    if regression_columns is None:
        regression_columns = target_scaler.target_columns
    regression_columns = [str(column) for column in regression_columns]

    dataset = SeismoMultiTaskDataset(
        metadata_path=metadata_path,
        split=split,
        data_root=data_root,
        regression_target_columns=regression_columns,
        target_scaler=target_scaler,
        normalize_input=normalize,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    classification_criterion = nn.CrossEntropyLoss()
    regression_criterion = nn.MSELoss()

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

            batch_size_current = x.size(0)
            total_samples += batch_size_current

            total_loss += float(loss.item()) * batch_size_current
            total_classification_loss += (
                loss_parts["classification_loss"] * batch_size_current
            )
            total_regression_loss += loss_parts["regression_loss"] * batch_size_current

            predictions = torch.argmax(outputs["logits"], dim=1)

            y_true_class.extend(class_id.detach().cpu().numpy().tolist())
            y_pred_class.extend(predictions.detach().cpu().numpy().tolist())
            y_true_regression.extend(regression.detach().cpu().numpy().tolist())
            y_pred_regression.extend(
                outputs["regression"].detach().cpu().numpy().tolist()
            )

    if total_samples == 0:
        raise ValueError("Cannot evaluate on an empty dataloader.")

    labels = list(range(num_classes))
    class_names = build_class_names(class_id_to_crack_count)

    y_true_regression_array = np.asarray(y_true_regression, dtype=np.float32)
    y_pred_regression_array = np.asarray(y_pred_regression, dtype=np.float32)

    regression_metrics = compute_regression_metrics(
        y_true_normalized=y_true_regression_array,
        y_pred_normalized=y_pred_regression_array,
        target_scaler=target_scaler,
    )

    classification_accuracy = float(accuracy_score(y_true_class, y_pred_class))
    classification_balanced_accuracy = float(
        balanced_accuracy_score(y_true_class, y_pred_class)
    )
    classification_macro_precision = float(
        precision_score(y_true_class, y_pred_class, average="macro", zero_division=0)
    )
    classification_macro_recall = float(
        recall_score(y_true_class, y_pred_class, average="macro", zero_division=0)
    )
    classification_macro_f1 = float(
        f1_score(y_true_class, y_pred_class, average="macro", zero_division=0)
    )

    report = classification_report(
        y_true_class,
        y_pred_class,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_true_class, y_pred_class, labels=labels)

    metrics = {
        "loss": total_loss / total_samples,
        "classification_loss": total_classification_loss / total_samples,
        "regression_loss": total_regression_loss / total_samples,
        "classification_accuracy": classification_accuracy,
        "classification_balanced_accuracy": classification_balanced_accuracy,
        "classification_macro_precision": classification_macro_precision,
        "classification_macro_recall": classification_macro_recall,
        "classification_macro_f1": classification_macro_f1,
        "confusion_matrix": cm,
        **regression_metrics,
        # Aliases for compatibility with scripts/compare_evaluations.py
        "accuracy": classification_accuracy,
        "balanced_accuracy": classification_balanced_accuracy,
        "macro_precision": classification_macro_precision,
        "macro_recall": classification_macro_recall,
        "macro_f1": classification_macro_f1,
    }

    result = {
        "checkpoint": {
            "path": str(checkpoint_path),
            "model_name": checkpoint.get("model_name"),
            "epoch": checkpoint.get("epoch"),
            "metric_to_optimize": checkpoint.get("metric_to_optimize"),
            "best_metric": checkpoint.get("best_metric"),
            "input_shape": checkpoint.get("input_shape"),
        },
        "dataset": {
            "metadata_path": str(metadata_path),
            "data_root": str(data_root),
            "split": split,
            "num_samples": len(dataset),
            "batch_size": batch_size,
            "normalize": normalize,
        },
        "device": str(device),
        "regression_loss_weight": regression_loss_weight,
        "class_id_to_crack_count": class_id_to_crack_count,
        "regression_columns": regression_columns,
        "target_scaler": target_scaler.to_dict(),
        "metrics": metrics,
        "classification_report": report,
    }

    return to_jsonable(result)


def save_multitask_evaluation_report(
    report: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Save multi-task evaluation report to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
        file.write("\n")
