from __future__ import annotations

import json
from pathlib import Path

import fire

from seismonn.exporting.tensorrt import export_tensorrt_engine


def export_tensorrt(
    onnx: str,
    engine: str,
    metadata_output: str | None = None,
    input_name: str = "features",
    input_shape: tuple[int, int, int] | None = None,
    min_batch_size: int = 1,
    opt_batch_size: int = 1,
    max_batch_size: int = 1,
    fp16: bool = False,
    int8: bool = False,
    verbose: bool = False,
    trtexec_path: str = "trtexec",
    dry_run: bool = False,
) -> None:
    """Export ONNX model to TensorRT engine."""
    metadata = export_tensorrt_engine(
        onnx_path=Path(onnx),
        engine_path=Path(engine),
        metadata_output_path=metadata_output,
        input_name=input_name,
        input_shape=input_shape,
        min_batch_size=min_batch_size,
        opt_batch_size=opt_batch_size,
        max_batch_size=max_batch_size,
        fp16=fp16,
        int8=int8,
        verbose=verbose,
        trtexec_path=trtexec_path,
        dry_run=dry_run,
    )

    print(json.dumps(metadata, indent=2, ensure_ascii=False))


def main() -> None:
    fire.Fire(export_tensorrt)


if __name__ == "__main__":
    main()
