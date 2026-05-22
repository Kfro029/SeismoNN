from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from seismonn.data.multitask_dataset import SeismoMultiTaskDataset  # noqa: E402
from seismonn.evaluation.multitask_checkpoint import (  # noqa: E402
    load_multitask_model_from_checkpoint,
)
from seismonn.inference.multitask_predictor import normalize_class_mapping  # noqa: E402
from seismonn.training.utils import get_device, to_jsonable  # noqa: E402


def format_sample_id(value: Any) -> str:
    """Format sample_id in a stable way."""
    if isinstance(value, (int, np.integer)):
        return f"{int(value):06d}"

    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return f"{int(value):06d}"

    value_str = str(value)

    if value_str.isdigit():
        return value_str.zfill(6)

    return value_str


def collect_multitask_predictions(
    checkpoint_path: str | Path,
    metadata_path: str | Path,
    split: str = "val",
    data_root: str | Path = ".",
    batch_size: int = 8,
    num_workers: int = 0,
    normalize: bool | None = None,
    device_name: str = "auto",
) -> pd.DataFrame:
    """Collect per-sample predictions for a multi-task checkpoint."""
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

    rows: list[dict[str, Any]] = []
    offset = 0

    model.eval()

    with torch.no_grad():
        for x, targets in dataloader:
            x = x.to(device)
            class_id = targets["class_id"].to(device)
            regression_target_normalized = targets["regression"].to(device)

            outputs = model(x)

            probabilities = torch.softmax(outputs["logits"], dim=1)
            predicted_class_id = torch.argmax(probabilities, dim=1)

            predicted_regression_normalized = outputs["regression"]

            probabilities_np = probabilities.detach().cpu().numpy()
            predicted_class_id_np = predicted_class_id.detach().cpu().numpy()
            true_class_id_np = class_id.detach().cpu().numpy()

            true_regression_normalized_np = (
                regression_target_normalized.detach().cpu().numpy().astype(np.float32)
            )
            predicted_regression_normalized_np = (
                predicted_regression_normalized.detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            true_regression = target_scaler.inverse_transform(
                true_regression_normalized_np
            )
            predicted_regression = target_scaler.inverse_transform(
                predicted_regression_normalized_np
            )

            batch_size_current = x.size(0)
            metadata_rows = dataset.metadata.iloc[
                offset : offset + batch_size_current
            ].reset_index(drop=True)

            for local_index in range(batch_size_current):
                true_class = int(true_class_id_np[local_index])
                pred_class = int(predicted_class_id_np[local_index])

                row = metadata_rows.iloc[local_index]

                prediction_row: dict[str, Any] = {
                    "sample_id": format_sample_id(row["sample_id"])
                    if "sample_id" in row.index
                    else str(offset + local_index),
                    "path": str(row["path"]) if "path" in row.index else None,
                    "filename": str(row["filename"])
                    if "filename" in row.index
                    else None,
                    "split": str(row["split"]) if "split" in row.index else split,
                    "true_class_id": true_class,
                    "predicted_class_id": pred_class,
                    "true_crack_count": int(class_id_to_crack_count[true_class]),
                    "predicted_crack_count": int(class_id_to_crack_count[pred_class]),
                }

                expected_crack_count = 0.0

                for class_id_key in sorted(class_id_to_crack_count):
                    crack_count = class_id_to_crack_count[class_id_key]
                    probability = float(probabilities_np[local_index, class_id_key])

                    prediction_row[f"prob_crack_count_{crack_count}"] = probability
                    expected_crack_count += crack_count * probability

                prediction_row["expected_crack_count"] = expected_crack_count

                for target_index, target_name in enumerate(regression_columns):
                    true_value = float(true_regression[local_index, target_index])
                    predicted_value = float(
                        predicted_regression[local_index, target_index]
                    )
                    error = predicted_value - true_value

                    prediction_row[f"true_{target_name}"] = true_value
                    prediction_row[f"pred_{target_name}"] = predicted_value
                    prediction_row[f"error_{target_name}"] = error
                    prediction_row[f"abs_error_{target_name}"] = abs(error)
                    prediction_row[f"squared_error_{target_name}"] = error**2

                rows.append(prediction_row)

            offset += batch_size_current

    return pd.DataFrame(rows)


def summarize_multitask_prediction_table(
    predictions: pd.DataFrame,
    regression_columns: list[str],
) -> dict[str, Any]:
    """Build summary metrics from per-sample prediction table."""
    if len(predictions) == 0:
        raise ValueError("Predictions table is empty.")

    true_class = predictions["true_class_id"].to_numpy()
    predicted_class = predictions["predicted_class_id"].to_numpy()

    summary: dict[str, Any] = {
        "num_samples": int(len(predictions)),
        "classification": {
            "accuracy": float(accuracy_score(true_class, predicted_class)),
            "macro_f1": float(
                f1_score(
                    true_class,
                    predicted_class,
                    average="macro",
                    zero_division=0,
                )
            ),
        },
        "regression": {
            "per_target": {},
        },
    }

    all_absolute_errors = []
    all_squared_errors = []

    for target_name in regression_columns:
        abs_error_column = f"abs_error_{target_name}"
        squared_error_column = f"squared_error_{target_name}"

        if abs_error_column not in predictions.columns:
            raise ValueError(f"Missing column: {abs_error_column}")

        if squared_error_column not in predictions.columns:
            raise ValueError(f"Missing column: {squared_error_column}")

        abs_errors = predictions[abs_error_column].to_numpy(dtype=np.float64)
        squared_errors = predictions[squared_error_column].to_numpy(dtype=np.float64)

        all_absolute_errors.append(abs_errors)
        all_squared_errors.append(squared_errors)

        summary["regression"]["per_target"][target_name] = {
            "mae": float(abs_errors.mean()),
            "rmse": float(np.sqrt(squared_errors.mean())),
            "max_abs_error": float(abs_errors.max()),
        }

    all_absolute_errors_array = np.concatenate(all_absolute_errors)
    all_squared_errors_array = np.concatenate(all_squared_errors)

    summary["regression"]["mae_mean"] = float(all_absolute_errors_array.mean())
    summary["regression"]["rmse_mean"] = float(np.sqrt(all_squared_errors_array.mean()))

    return summary


def save_predictions_csv(
    predictions: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Save predictions table to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    predictions.to_csv(output_path, index=False)


def save_summary_json(
    summary: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Save summary JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(to_jsonable(summary), file, indent=2, ensure_ascii=False)
        file.write("\n")


def plot_regression_parity(
    predictions: pd.DataFrame,
    target_name: str,
    output_path: str | Path,
) -> None:
    """Save true-vs-predicted parity plot for one target."""
    true_column = f"true_{target_name}"
    predicted_column = f"pred_{target_name}"

    if true_column not in predictions.columns:
        raise ValueError(f"Missing column: {true_column}")

    if predicted_column not in predictions.columns:
        raise ValueError(f"Missing column: {predicted_column}")

    true_values = predictions[true_column].to_numpy(dtype=np.float64)
    predicted_values = predictions[predicted_column].to_numpy(dtype=np.float64)

    min_value = float(min(true_values.min(), predicted_values.min()))
    max_value = float(max(true_values.max(), predicted_values.max()))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.scatter(true_values, predicted_values)
    plt.plot([min_value, max_value], [min_value, max_value])
    plt.xlabel(f"True {target_name}")
    plt.ylabel(f"Predicted {target_name}")
    plt.title(f"True vs predicted: {target_name}")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_regression_parity_plots(
    predictions: pd.DataFrame,
    regression_columns: list[str],
    output_dir: str | Path,
) -> dict[str, str]:
    """Save parity plots for all regression targets."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_paths = {}

    for target_name in regression_columns:
        output_path = output_dir / f"parity_{target_name}.png"

        plot_regression_parity(
            predictions=predictions,
            target_name=target_name,
            output_path=output_path,
        )

        plot_paths[target_name] = str(output_path)

    return plot_paths
