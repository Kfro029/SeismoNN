from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from seismonn.data.multitask_dataset import RegressionTargetScaler
from seismonn.inference.predictor import load_torch_checkpoint
from seismonn.models.cnn_multitask import SeismoCNNMultiTask
from seismonn.training.utils import get_device


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


class SeismoMultiTaskPredictor:
    """Predict fracture count and regression parameters from one .npy sample."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        class_id_to_crack_count: dict[int, int],
        target_scaler: RegressionTargetScaler,
        regression_columns: list[str],
        input_shape: tuple[int, int, int] | None = None,
        normalize: bool = True,
        model_name: str = "unknown",
        checkpoint_path: str | Path | None = None,
    ) -> None:
        self.model = model
        self.device = device
        self.class_id_to_crack_count = class_id_to_crack_count
        self.target_scaler = target_scaler
        self.regression_columns = regression_columns
        self.input_shape = input_shape
        self.normalize = normalize
        self.model_name = model_name
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None

        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device_name: str = "auto",
    ) -> SeismoMultiTaskPredictor:
        device = get_device(device_name)

        checkpoint = load_torch_checkpoint(
            checkpoint_path=checkpoint_path,
            device=device,
        )

        model_config = checkpoint.get("model_config", {})

        if not isinstance(model_config, dict):
            raise ValueError("Checkpoint field 'model_config' must be a dictionary.")

        model_name = str(model_config.get("name", checkpoint.get("model_name", "")))

        if model_name != "cnn_multitask":
            raise ValueError(
                f"Expected multi-task checkpoint with model name 'cnn_multitask', "
                f"got {model_name!r}."
            )

        model = SeismoCNNMultiTask(
            in_channels=int(model_config.get("in_channels", 2)),
            num_classes=int(model_config.get("num_classes", 3)),
            num_regression_targets=int(model_config.get("num_regression_targets", 4)),
            dropout=float(model_config.get("dropout", 0.2)),
        )

        if "model_state_dict" not in checkpoint:
            raise ValueError("Checkpoint does not contain 'model_state_dict'.")

        model.load_state_dict(checkpoint["model_state_dict"])

        class_id_to_crack_count = normalize_class_mapping(
            checkpoint.get("class_id_to_crack_count")
        )

        target_scaler_data = checkpoint.get("target_scaler")

        if not isinstance(target_scaler_data, dict):
            raise ValueError(
                "Multi-task checkpoint must contain dictionary field 'target_scaler'."
            )

        target_scaler = RegressionTargetScaler.from_dict(target_scaler_data)

        regression_columns = checkpoint.get("regression_columns")

        if regression_columns is None:
            regression_columns = target_scaler.target_columns

        regression_columns = [str(column) for column in regression_columns]

        raw_input_shape = checkpoint.get("input_shape")
        input_shape = None

        if raw_input_shape is not None:
            input_shape = tuple(int(dim) for dim in raw_input_shape)

        data_config = checkpoint.get("data_config", {})
        normalize = True

        if isinstance(data_config, dict):
            normalize = bool(data_config.get("normalize", True))

        return cls(
            model=model,
            device=device,
            class_id_to_crack_count=class_id_to_crack_count,
            target_scaler=target_scaler,
            regression_columns=regression_columns,
            input_shape=input_shape,
            normalize=normalize,
            model_name=model_name,
            checkpoint_path=checkpoint_path,
        )

    def load_sample(self, input_path: str | Path) -> torch.Tensor:
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file does not exist: {input_path}")

        x = np.load(input_path, mmap_mode="r")
        x = np.asarray(x, dtype=np.float32)

        if x.ndim != 3:
            raise ValueError(f"Expected input shape (C, T, R), got {x.shape}")

        if self.input_shape is not None and tuple(x.shape) != self.input_shape:
            raise ValueError(
                f"Expected input shape {self.input_shape}, got {tuple(x.shape)}. "
                "For now inference expects the same shape as training."
            )

        x = np.array(x, dtype=np.float32, copy=True)

        if self.normalize:
            mean = float(x.mean())
            std = float(x.std()) + 1e-8
            x = (x - mean) / std

        return torch.from_numpy(x).unsqueeze(0)

    def predict_file(self, input_path: str | Path) -> dict[str, Any]:
        input_path = Path(input_path)
        x = self.load_sample(input_path).to(self.device)

        with torch.no_grad():
            outputs = self.model(x)

            logits = outputs["logits"]
            regression_normalized = outputs["regression"]

            probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
            regression_normalized_np = (
                regression_normalized[0].detach().cpu().numpy().astype(np.float32)
            )

        predicted_class_id = int(np.argmax(probabilities))
        predicted_crack_count = self.class_id_to_crack_count[predicted_class_id]

        class_probabilities = {
            str(self.class_id_to_crack_count[class_id]): float(probabilities[class_id])
            for class_id in sorted(self.class_id_to_crack_count)
        }

        expected_crack_count = float(
            sum(
                self.class_id_to_crack_count[class_id] * float(probabilities[class_id])
                for class_id in sorted(self.class_id_to_crack_count)
            )
        )

        regression_original = self.target_scaler.inverse_transform(
            regression_normalized_np
        )

        regression_result = {
            column: float(value)
            for column, value in zip(self.regression_columns, regression_original)
        }

        result: dict[str, Any] = {
            "model_name": self.model_name,
            "input_path": str(input_path),
            "predicted_class_id": predicted_class_id,
            "predicted_crack_count": predicted_crack_count,
            "expected_crack_count": expected_crack_count,
            "class_probabilities": class_probabilities,
            "regression": regression_result,
        }

        if self.checkpoint_path is not None:
            result["checkpoint_path"] = str(self.checkpoint_path)

        return result


def save_multitask_prediction_json(
    prediction: dict[str, Any],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(prediction, file, indent=2, ensure_ascii=False)
        file.write("\n")