from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from seismonn.models.cnn import SeismoCNN
from seismonn.training.utils import get_device


def load_torch_checkpoint(
    checkpoint_path: str | Path, device: torch.device
) -> dict[str, Any]:
    """Load torch checkpoint.

    weights_only=False is used for compatibility with checkpoints that contain
    model metadata in addition to tensors.
    """
    checkpoint_path = Path(checkpoint_path)

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"Expected checkpoint to be a dictionary, got {type(checkpoint)}"
        )

    return checkpoint


class SeismoPredictor:
    """Predict fracture count from a single seismic .npy sample."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        class_id_to_crack_count: dict[int, int],
        input_shape: tuple[int, int, int] | None = None,
        normalize: bool = True,
        model_name: str = "unknown",
        checkpoint_path: str | Path | None = None,
    ) -> None:
        self.model = model
        self.device = device
        self.class_id_to_crack_count = class_id_to_crack_count
        self.input_shape = input_shape
        self.normalize = normalize
        self.model_name = model_name
        self.checkpoint_path = (
            Path(checkpoint_path) if checkpoint_path is not None else None
        )

        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device_name: str = "auto",
    ) -> SeismoPredictor:
        device = get_device(device_name)
        checkpoint = load_torch_checkpoint(checkpoint_path, device)

        model_config = checkpoint.get("model_config", {})
        if not isinstance(model_config, dict):
            raise ValueError("Checkpoint field 'model_config' must be a dictionary.")

        model = SeismoCNN(
            in_channels=int(model_config.get("in_channels", 2)),
            num_classes=int(model_config.get("num_classes", 3)),
            dropout=float(model_config.get("dropout", 0.0)),
        )

        if "model_state_dict" not in checkpoint:
            raise ValueError("Checkpoint does not contain 'model_state_dict'.")

        model.load_state_dict(checkpoint["model_state_dict"])

        raw_mapping = checkpoint.get("class_id_to_crack_count", {0: 3, 1: 4, 2: 5})
        class_id_to_crack_count = {
            int(class_id): int(crack_count)
            for class_id, crack_count in raw_mapping.items()
        }

        raw_input_shape = checkpoint.get("input_shape")
        input_shape = None
        if raw_input_shape is not None:
            input_shape = tuple(int(dim) for dim in raw_input_shape)

        data_config = checkpoint.get("data_config", {})
        normalize = True
        if isinstance(data_config, dict):
            normalize = bool(data_config.get("normalize", True))

        model_name = str(
            checkpoint.get("model_name", model_config.get("name", "cnn_baseline"))
        )

        return cls(
            model=model,
            device=device,
            class_id_to_crack_count=class_id_to_crack_count,
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

        tensor = torch.from_numpy(x).unsqueeze(0)
        return tensor

    def predict_file(self, input_path: str | Path) -> dict[str, Any]:
        input_path = Path(input_path)
        x = self.load_sample(input_path).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

        predicted_class_id = int(np.argmax(probabilities))
        predicted_crack_count = self.class_id_to_crack_count[predicted_class_id]

        class_probabilities = {
            str(self.class_id_to_crack_count[class_id]): float(probabilities[class_id])
            for class_id in sorted(self.class_id_to_crack_count)
        }

        result: dict[str, Any] = {
            "model_name": self.model_name,
            "input_path": str(input_path),
            "predicted_class_id": predicted_class_id,
            "predicted_crack_count": predicted_crack_count,
            "class_probabilities": class_probabilities,
        }

        if self.checkpoint_path is not None:
            result["checkpoint_path"] = str(self.checkpoint_path)

        return result


def save_prediction_json(prediction: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(prediction, file, indent=2, ensure_ascii=False)
