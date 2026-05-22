from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from seismonn.inference.multitask_predictor import SeismoMultiTaskPredictor
from seismonn.inference.predictor import SeismoPredictor, load_torch_checkpoint
from seismonn.training.utils import get_device


CLASSIFICATION_PREDICTOR_TYPE = "classification"
MULTITASK_PREDICTOR_TYPE = "multitask"
AUTO_PREDICTOR_TYPE = "auto"


@dataclass
class LoadedPredictor:
    """Container for loaded predictor and its metadata."""

    predictor: Any
    predictor_type: str
    model_name: str


def get_checkpoint_model_name(checkpoint: dict[str, Any]) -> str:
    """Extract model name from checkpoint metadata."""
    model_config = checkpoint.get("model_config", {})

    if isinstance(model_config, dict) and "name" in model_config:
        return str(model_config["name"])

    return str(checkpoint.get("model_name", ""))


def infer_predictor_type_from_checkpoint(
    checkpoint_path: str | Path,
    device_name: str = "auto",
) -> tuple[str, str]:
    """Infer predictor type from checkpoint.

    Returns:
        Tuple of (predictor_type, model_name).
    """
    device = get_device(device_name)
    checkpoint = load_torch_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    model_name = get_checkpoint_model_name(checkpoint).lower()

    if model_name == "cnn_multitask":
        return MULTITASK_PREDICTOR_TYPE, model_name

    return CLASSIFICATION_PREDICTOR_TYPE, model_name


def create_predictor(
    checkpoint_path: str | Path,
    device_name: str = "auto",
    predictor_type: str = AUTO_PREDICTOR_TYPE,
) -> LoadedPredictor:
    """Create classification or multi-task predictor from checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    predictor_type = predictor_type.lower()

    if predictor_type == AUTO_PREDICTOR_TYPE:
        predictor_type, model_name = infer_predictor_type_from_checkpoint(
            checkpoint_path=checkpoint_path,
            device_name=device_name,
        )
    else:
        _, model_name = infer_predictor_type_from_checkpoint(
            checkpoint_path=checkpoint_path,
            device_name=device_name,
        )

    if predictor_type == CLASSIFICATION_PREDICTOR_TYPE:
        predictor = SeismoPredictor.from_checkpoint(
            checkpoint_path=checkpoint_path,
            device_name=device_name,
        )

        return LoadedPredictor(
            predictor=predictor,
            predictor_type=CLASSIFICATION_PREDICTOR_TYPE,
            model_name=model_name,
        )

    if predictor_type == MULTITASK_PREDICTOR_TYPE:
        predictor = SeismoMultiTaskPredictor.from_checkpoint(
            checkpoint_path=checkpoint_path,
            device_name=device_name,
        )

        return LoadedPredictor(
            predictor=predictor,
            predictor_type=MULTITASK_PREDICTOR_TYPE,
            model_name=model_name,
        )

    raise ValueError(
        f"Unsupported predictor_type={predictor_type!r}. "
        f"Expected one of: {AUTO_PREDICTOR_TYPE!r}, "
        f"{CLASSIFICATION_PREDICTOR_TYPE!r}, {MULTITASK_PREDICTOR_TYPE!r}."
    )
