from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from seismonn.training.utils import resolve_path


DEFAULT_TRANSFER_PREFIXES = (
    "temporal_encoder.",
    "encoder.",
)


def load_torch_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    """Load torch checkpoint with metadata."""
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
        raise ValueError(f"Expected checkpoint to be a dict, got {type(checkpoint)}.")

    return checkpoint


def select_compatible_encoder_weights(
    pretrained_state_dict: dict[str, torch.Tensor],
    target_state_dict: dict[str, torch.Tensor],
    prefixes: tuple[str, ...] = DEFAULT_TRANSFER_PREFIXES,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Select compatible encoder weights from a pretraining checkpoint."""
    selected: dict[str, torch.Tensor] = {}

    skipped_not_encoder: list[str] = []
    skipped_missing_in_target: list[str] = []
    skipped_shape_mismatch: list[dict[str, Any]] = []

    for key, value in pretrained_state_dict.items():
        if not any(key.startswith(prefix) for prefix in prefixes):
            skipped_not_encoder.append(key)
            continue

        if key not in target_state_dict:
            skipped_missing_in_target.append(key)
            continue

        target_value = target_state_dict[key]

        if tuple(value.shape) != tuple(target_value.shape):
            skipped_shape_mismatch.append(
                {
                    "key": key,
                    "pretrained_shape": list(value.shape),
                    "target_shape": list(target_value.shape),
                }
            )
            continue

        selected[key] = value

    summary = {
        "selected_key_count": len(selected),
        "selected_keys": sorted(selected),
        "skipped_not_encoder_count": len(skipped_not_encoder),
        "skipped_missing_in_target_count": len(skipped_missing_in_target),
        "skipped_shape_mismatch_count": len(skipped_shape_mismatch),
        "skipped_shape_mismatch": skipped_shape_mismatch,
        "prefixes": list(prefixes),
    }

    return selected, summary


def load_pretrained_encoder_weights(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
    prefixes: tuple[str, ...] = DEFAULT_TRANSFER_PREFIXES,
    min_loaded_keys: int = 1,
) -> dict[str, Any]:
    """Load encoder weights from a self-supervised checkpoint into classifier."""
    checkpoint = load_torch_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint does not contain 'model_state_dict'.")

    pretrained_state_dict = checkpoint["model_state_dict"]

    if not isinstance(pretrained_state_dict, dict):
        raise ValueError("Checkpoint field 'model_state_dict' must be a dict.")

    target_state_dict = model.state_dict()

    selected_state_dict, summary = select_compatible_encoder_weights(
        pretrained_state_dict=pretrained_state_dict,
        target_state_dict=target_state_dict,
        prefixes=prefixes,
    )

    loaded_key_count = len(selected_state_dict)

    if loaded_key_count < min_loaded_keys:
        raise ValueError(
            f"Only {loaded_key_count} compatible keys were found, "
            f"but min_loaded_keys={min_loaded_keys}. "
            "Check that supervised Transformer config matches pretraining config."
        )

    incompatible_keys = model.load_state_dict(selected_state_dict, strict=False)

    summary.update(
        {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_model_name": checkpoint.get("model_name"),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "loaded_key_count": loaded_key_count,
            "missing_keys_after_partial_load": list(incompatible_keys.missing_keys),
            "unexpected_keys_after_partial_load": list(
                incompatible_keys.unexpected_keys
            ),
        }
    )

    return summary


def maybe_load_pretrained_encoder(
    model: nn.Module,
    pretrained_config: dict[str, Any] | None,
    project_root: Path,
    device: torch.device,
) -> dict[str, Any] | None:
    """Load pretrained encoder if enabled in config."""
    if not pretrained_config or not bool(pretrained_config.get("enabled", False)):
        return None

    if "checkpoint_path" not in pretrained_config:
        raise ValueError(
            "pretrained.enabled=true, but pretrained.checkpoint_path is not provided."
        )

    checkpoint_path = resolve_path(
        project_root=project_root,
        path=pretrained_config["checkpoint_path"],
    )

    raw_prefixes = pretrained_config.get("prefixes", DEFAULT_TRANSFER_PREFIXES)
    prefixes = tuple(str(prefix) for prefix in raw_prefixes)

    min_loaded_keys = int(pretrained_config.get("min_loaded_keys", 1))

    return load_pretrained_encoder_weights(
        model=model,
        checkpoint_path=checkpoint_path,
        device=device,
        prefixes=prefixes,
        min_loaded_keys=min_loaded_keys,
    )
