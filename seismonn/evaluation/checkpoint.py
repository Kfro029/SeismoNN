from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader

from seismonn.data.dataset import SeismoDataset
from seismonn.inference.predictor import load_torch_checkpoint
from seismonn.models.factory import create_model
from seismonn.training.evaluate import evaluate_classifier
from seismonn.training.utils import get_device, to_jsonable


def normalize_class_mapping(raw_mapping: dict[Any, Any] | None) -> dict[int, int]:
    """Convert checkpoint class mapping to int -> int."""
    if raw_mapping is None:
        return {
            0: 3,
            1: 4,
            2: 5,
        }

    return {
        int(class_id): int(crack_count)
        for class_id, crack_count in raw_mapping.items()
    }


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load model and metadata from checkpoint."""
    checkpoint = load_torch_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint file does not exist: {checkpoint_path}. "
            "Train the model first or provide a valid --checkpoint path."
        )

    model_config = checkpoint.get("model_config")

    if not isinstance(model_config, dict):
        raise ValueError("Checkpoint must contain dictionary field 'model_config'.")

    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint does not contain 'model_state_dict'.")

    model = create_model(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint


def build_class_names(class_id_to_crack_count: dict[int, int]) -> list[str]:
    """Build class names sorted by class_id."""
    return [
        str(class_id_to_crack_count[class_id])
        for class_id in sorted(class_id_to_crack_count)
    ]


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    metadata_path: str | Path,
    split: str = "val",
    data_root: str | Path = ".",
    batch_size: int = 16,
    num_workers: int = 0,
    normalize: bool | None = None,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Evaluate trained checkpoint on selected metadata split."""
    device = get_device(device_name)

    model, checkpoint = load_model_from_checkpoint(
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

    dataset = SeismoDataset(
        metadata_path=metadata_path,
        split=split,
        data_root=data_root,
        normalize=normalize,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    criterion = nn.CrossEntropyLoss()
    labels = list(range(num_classes))

    raw_metrics = evaluate_classifier(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        device=device,
        labels=labels,
    )

    class_names = build_class_names(class_id_to_crack_count)

    report = classification_report(
        raw_metrics["y_true"],
        raw_metrics["y_pred"],
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "loss": raw_metrics["loss"],
        "accuracy": raw_metrics["accuracy"],
        "balanced_accuracy": raw_metrics["balanced_accuracy"],
        "macro_precision": raw_metrics["macro_precision"],
        "macro_recall": raw_metrics["macro_recall"],
        "macro_f1": raw_metrics["macro_f1"],
        "confusion_matrix": raw_metrics["confusion_matrix"],
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
        "class_id_to_crack_count": class_id_to_crack_count,
        "metrics": metrics,
        "classification_report": report,
    }

    return to_jsonable(result)


def save_evaluation_report(
    report: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Save evaluation report to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
        file.write("\n")