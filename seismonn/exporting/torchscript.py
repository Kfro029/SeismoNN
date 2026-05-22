from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from seismonn.inference.predictor import load_torch_checkpoint
from seismonn.models.cnn_multitask import SeismoCNNMultiTask
from seismonn.models.factory import create_model
from seismonn.training.utils import get_device, to_jsonable


class ClassificationTorchScriptWrapper(nn.Module):
    """TorchScript wrapper for classification models.

    The wrapped model returns logits with shape [B, num_classes].
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MultiTaskTorchScriptWrapper(nn.Module):
    """TorchScript wrapper for multi-task models.

    The wrapped model returns a tuple:
    - logits: [B, num_classes]
    - regression: [B, num_regression_targets]

    Tuple output is more TorchScript-friendly than a Python dict.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.model(x)
        return outputs["logits"], outputs["regression"]


def get_checkpoint_model_name(checkpoint: dict[str, Any]) -> str:
    """Extract model name from checkpoint."""
    model_config = checkpoint.get("model_config", {})

    if isinstance(model_config, dict) and "name" in model_config:
        return str(model_config["name"])

    return str(checkpoint.get("model_name", ""))


def create_model_from_checkpoint(checkpoint: dict[str, Any]) -> nn.Module:
    """Create PyTorch model from checkpoint metadata."""
    model_config = checkpoint.get("model_config")

    if not isinstance(model_config, dict):
        raise ValueError("Checkpoint must contain dictionary field 'model_config'.")

    model_name = str(model_config.get("name", checkpoint.get("model_name", "")))

    if model_name == "cnn_multitask":
        return SeismoCNNMultiTask(
            in_channels=int(model_config.get("in_channels", 2)),
            num_classes=int(model_config.get("num_classes", 3)),
            num_regression_targets=int(model_config.get("num_regression_targets", 4)),
            dropout=float(model_config.get("dropout", 0.2)),
        )

    return create_model(model_config)


def create_torchscript_wrapper(
    model: nn.Module,
    model_name: str,
) -> tuple[nn.Module, str]:
    """Create TorchScript wrapper and output format description."""
    if model_name == "cnn_multitask":
        return MultiTaskTorchScriptWrapper(model), "tuple(logits, regression)"

    return ClassificationTorchScriptWrapper(model), "logits"


def resolve_input_shape(
    checkpoint: dict[str, Any],
    input_shape: tuple[int, int, int] | None = None,
) -> tuple[int, int, int]:
    """Resolve input shape for tracing."""
    if input_shape is not None:
        return tuple(int(dim) for dim in input_shape)

    checkpoint_input_shape = checkpoint.get("input_shape")

    if checkpoint_input_shape is None:
        raise ValueError(
            "Input shape is not provided and checkpoint does not contain "
            "'input_shape'. Pass --input-shape C T R explicitly."
        )

    if len(checkpoint_input_shape) != 3:
        raise ValueError(
            f"Expected input_shape with 3 dimensions [C, T, R], "
            f"got {checkpoint_input_shape}."
        )

    return tuple(int(dim) for dim in checkpoint_input_shape)


def save_json(data: dict[str, Any], output_path: str | Path) -> None:
    """Save JSON metadata."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(to_jsonable(data), file, indent=2, ensure_ascii=False)
        file.write("\n")


def export_torchscript_checkpoint(
    checkpoint_path: str | Path,
    output_path: str | Path,
    metadata_output_path: str | Path | None = None,
    device_name: str = "cpu",
    input_shape: tuple[int, int, int] | None = None,
) -> dict[str, Any]:
    """Export PyTorch checkpoint to TorchScript.

    Args:
        checkpoint_path: Path to PyTorch checkpoint.
        output_path: Path to save TorchScript model.
        metadata_output_path: Optional JSON metadata path.
        device_name: Device used for export.
        input_shape: Optional input shape [C, T, R]. If omitted, checkpoint
            field 'input_shape' is used.

    Returns:
        Export metadata dictionary.
    """
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)

    if metadata_output_path is None:
        metadata_output_path = output_path.with_suffix(".metadata.json")

    metadata_output_path = Path(metadata_output_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")

    device = get_device(device_name)

    checkpoint = load_torch_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint does not contain 'model_state_dict'.")

    model_name = get_checkpoint_model_name(checkpoint)

    model = create_model_from_checkpoint(checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    resolved_input_shape = resolve_input_shape(
        checkpoint=checkpoint,
        input_shape=input_shape,
    )

    wrapper, output_format = create_torchscript_wrapper(
        model=model,
        model_name=model_name,
    )
    wrapper.to(device)
    wrapper.eval()

    example_input = torch.zeros(
        (1, *resolved_input_shape),
        dtype=torch.float32,
        device=device,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        traced_model = torch.jit.trace(
            wrapper,
            example_input,
            strict=False,
        )

    traced_model.save(str(output_path))

    metadata = {
        "checkpoint_path": str(checkpoint_path),
        "torchscript_path": str(output_path),
        "metadata_path": str(metadata_output_path),
        "model_name": model_name,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "input_shape": list(resolved_input_shape),
        "output_format": output_format,
        "class_id_to_crack_count": checkpoint.get("class_id_to_crack_count"),
        "regression_columns": checkpoint.get("regression_columns"),
        "target_scaler": checkpoint.get("target_scaler"),
    }

    save_json(metadata, metadata_output_path)

    return metadata