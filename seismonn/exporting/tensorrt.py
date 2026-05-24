from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from seismonn.training.utils import to_jsonable


def save_json(data: dict[str, Any], output_path: str | Path) -> None:
    """Save JSON metadata."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(to_jsonable(data), file, indent=2, ensure_ascii=False)
        file.write("\n")


def resolve_trtexec_path(trtexec_path: str = "trtexec") -> str:
    """Resolve trtexec executable path."""
    resolved_path = shutil.which(trtexec_path)

    if resolved_path is None:
        raise RuntimeError(
            f"TensorRT executable {trtexec_path!r} was not found. "
            "Install NVIDIA TensorRT and make sure trtexec is available in PATH."
        )

    return resolved_path


def build_shape_profile_args(
    input_name: str,
    input_shape: tuple[int, int, int],
    min_batch_size: int = 1,
    opt_batch_size: int = 1,
    max_batch_size: int = 1,
) -> list[str]:
    """Build TensorRT optimization profile args for dynamic batch ONNX models."""
    if len(input_shape) != 3:
        raise ValueError(f"Expected input_shape [C, T, R], got {input_shape}")

    if min_batch_size <= 0 or opt_batch_size <= 0 or max_batch_size <= 0:
        raise ValueError("Batch sizes must be positive.")

    if not min_batch_size <= opt_batch_size <= max_batch_size:
        raise ValueError(
            "Expected min_batch_size <= opt_batch_size <= max_batch_size. "
            f"Got {min_batch_size}, {opt_batch_size}, {max_batch_size}."
        )

    channels, time_steps, receivers = input_shape

    min_shape = f"{input_name}:{min_batch_size}x{channels}x{time_steps}x{receivers}"
    opt_shape = f"{input_name}:{opt_batch_size}x{channels}x{time_steps}x{receivers}"
    max_shape = f"{input_name}:{max_batch_size}x{channels}x{time_steps}x{receivers}"

    return [
        f"--minShapes={min_shape}",
        f"--optShapes={opt_shape}",
        f"--maxShapes={max_shape}",
    ]


def build_trtexec_command(
    onnx_path: str | Path,
    engine_path: str | Path,
    input_name: str = "features",
    input_shape: tuple[int, int, int] | None = None,
    min_batch_size: int = 1,
    opt_batch_size: int = 1,
    max_batch_size: int = 1,
    fp16: bool = False,
    int8: bool = False,
    verbose: bool = False,
    trtexec_path: str = "trtexec",
) -> list[str]:
    """Build trtexec command for ONNX -> TensorRT engine conversion."""
    onnx_path = Path(onnx_path)
    engine_path = Path(engine_path)

    command = [
        trtexec_path,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
    ]

    if input_shape is not None:
        command.extend(
            build_shape_profile_args(
                input_name=input_name,
                input_shape=input_shape,
                min_batch_size=min_batch_size,
                opt_batch_size=opt_batch_size,
                max_batch_size=max_batch_size,
            )
        )

    if fp16:
        command.append("--fp16")

    if int8:
        command.append("--int8")

    if verbose:
        command.append("--verbose")

    return command


def export_tensorrt_engine(
    onnx_path: str | Path,
    engine_path: str | Path,
    metadata_output_path: str | Path | None = None,
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
) -> dict[str, Any]:
    """Export ONNX model to TensorRT engine using trtexec."""
    onnx_path = Path(onnx_path)
    engine_path = Path(engine_path)

    if metadata_output_path is None:
        metadata_output_path = engine_path.with_suffix(".metadata.json")

    metadata_output_path = Path(metadata_output_path)

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file does not exist: {onnx_path}")

    resolved_trtexec_path = trtexec_path

    if not dry_run:
        resolved_trtexec_path = resolve_trtexec_path(trtexec_path)

    command = build_trtexec_command(
        onnx_path=onnx_path,
        engine_path=engine_path,
        input_name=input_name,
        input_shape=input_shape,
        min_batch_size=min_batch_size,
        opt_batch_size=opt_batch_size,
        max_batch_size=max_batch_size,
        fp16=fp16,
        int8=int8,
        verbose=verbose,
        trtexec_path=resolved_trtexec_path,
    )

    engine_path.parent.mkdir(parents=True, exist_ok=True)

    if not dry_run:
        subprocess.run(command, check=True)

        if not engine_path.exists():
            raise FileNotFoundError(
                f"trtexec finished, but TensorRT engine was not created: {engine_path}"
            )

    metadata = {
        "onnx_path": str(onnx_path),
        "engine_path": str(engine_path),
        "metadata_path": str(metadata_output_path),
        "input_name": input_name,
        "input_shape": list(input_shape) if input_shape is not None else None,
        "min_batch_size": min_batch_size,
        "opt_batch_size": opt_batch_size,
        "max_batch_size": max_batch_size,
        "fp16": fp16,
        "int8": int8,
        "verbose": verbose,
        "dry_run": dry_run,
        "trtexec_path": resolved_trtexec_path,
        "command": command,
    }

    save_json(metadata, metadata_output_path)

    return metadata
