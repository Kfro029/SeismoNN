from __future__ import annotations

import json
from typing import Any

import fire

from seismonn.exporting.torchscript import export_torchscript_checkpoint
from seismonn.training.utils import to_jsonable


def parse_input_shape(input_shape: Any | None) -> tuple[int, int, int] | None:
    """Parse input shape from Fire CLI value."""
    if input_shape is None:
        return None

    if isinstance(input_shape, (list, tuple)):
        if len(input_shape) != 3:
            raise ValueError(f"Expected 3 input dimensions, got {input_shape!r}.")
        return tuple(int(value) for value in input_shape)

    normalized = str(input_shape).strip().strip("()[]")
    normalized = normalized.replace("x", ",").replace(" ", "")
    parts = [part for part in normalized.split(",") if part]

    if len(parts) != 3:
        raise ValueError(f"Expected 3 input dimensions, got {input_shape!r}.")

    return tuple(int(part) for part in parts)


def main(
    checkpoint: str,
    output: str,
    metadata_output: str | None = None,
    device: str = "cpu",
    input_shape: Any | None = None,
) -> None:
    """Export SeismoNN checkpoint to TorchScript."""
    metadata = export_torchscript_checkpoint(
        checkpoint_path=checkpoint,
        output_path=output,
        metadata_output_path=metadata_output,
        device_name=device,
        input_shape=parse_input_shape(input_shape),
    )

    print(json.dumps(to_jsonable(metadata), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    fire.Fire(main)
