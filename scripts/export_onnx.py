from __future__ import annotations

import json
from pathlib import Path

import fire

from seismonn.exporting.onnx import export_onnx_checkpoint


def export_onnx(
    checkpoint: str,
    output: str,
    metadata_output: str | None = None,
    device: str = "cpu",
    input_shape: tuple[int, int, int] | None = None,
    opset_version: int = 17,
    dynamic_batch: bool = True,
    validate: bool = True,
    smoke_test: bool = True,
) -> None:
    """Export SeismoNN checkpoint to ONNX."""
    metadata = export_onnx_checkpoint(
        checkpoint_path=Path(checkpoint),
        output_path=Path(output),
        metadata_output_path=metadata_output,
        device_name=device,
        input_shape=input_shape,
        opset_version=opset_version,
        dynamic_batch=dynamic_batch,
        validate=validate,
        smoke_test=smoke_test,
    )

    print(json.dumps(metadata, indent=2, ensure_ascii=False))


def main() -> None:
    fire.Fire(export_onnx)


if __name__ == "__main__":
    main()
