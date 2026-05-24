from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import fire

from seismonn.data.download import ensure_data_available
from seismonn.data.validation import validate_metadata
from seismonn.exporting.onnx import export_onnx_checkpoint
from seismonn.exporting.tensorrt import export_tensorrt_engine
from seismonn.inference.factory import create_predictor
from seismonn.serving.mlflow_model import save_mlflow_pyfunc_model
from seismonn.training.utils import to_jsonable


def print_json(data: dict[str, Any]) -> None:
    """Print JSON to stdout."""
    print(json.dumps(to_jsonable(data), indent=2, ensure_ascii=False))


def save_json(data: dict[str, Any], output_path: str | Path) -> None:
    """Save JSON to file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(to_jsonable(data), file, indent=2, ensure_ascii=False)
        file.write("\n")


def run_command(command: list[str]) -> None:
    """Run command and raise on failure."""
    subprocess.run(command, check=True)


def parse_input_shape(input_shape: str | None) -> tuple[int, int, int] | None:
    """Parse input shape from CLI string.

    Supports:
    - "2,1723,501"
    - "2x1723x501"
    """
    if input_shape is None:
        return None

    normalized = str(input_shape).replace("x", ",")
    parts = [part.strip() for part in normalized.split(",") if part.strip()]

    if len(parts) != 3:
        raise ValueError(
            f"Expected input_shape with 3 dimensions, got {input_shape!r}."
        )

    return tuple(int(part) for part in parts)


def train(overrides: str = "") -> None:
    """Run official Hydra + PyTorch Lightning training pipeline.

    Example:
        uv run seismonn train --overrides "trainer.max_epochs=1 tracking.enabled=false"
    """
    command = [
        sys.executable,
        "scripts/train_lightning.py",
    ]

    if overrides:
        command.extend(shlex.split(overrides))

    run_command(command)


def download_data(
    metadata_path: str = "data/metadata.csv",
    data_dir: str = "2nd_selection",
    repo_root: str = ".",
    use_dvc: bool = True,
    dvc_remote: str = "data_storage",
    huggingface_repo_id: str = "FAKIrik/Seismo_datasets",
    allow_huggingface_fallback: bool = True,
) -> None:
    """Ensure data and metadata are available."""
    ensure_data_available(
        metadata_path=metadata_path,
        data_dir=data_dir,
        repo_root=repo_root,
        use_dvc=use_dvc,
        dvc_remote=dvc_remote,
        huggingface_repo_id=huggingface_repo_id,
        allow_huggingface_fallback=allow_huggingface_fallback,
    )

    print_json(
        {
            "status": "ok",
            "metadata_path": metadata_path,
            "data_dir": data_dir,
        }
    )


def validate_data(
    metadata: str = "data/metadata.csv",
    data_root: str = ".",
    expected_shape: str = "2,1723,501",
    expected_dtype: str = "float32",
    expected_splits: str = "train,val",
    validate_files: bool = False,
    output: str | None = None,
) -> None:
    """Validate metadata.csv and optionally referenced .npy files."""
    expected_shape_tuple = parse_input_shape(expected_shape)
    expected_split_tuple = tuple(
        split.strip() for split in expected_splits.split(",") if split.strip()
    )

    report = validate_metadata(
        metadata_path=metadata,
        data_root=data_root,
        expected_shape=expected_shape_tuple,
        expected_dtype=expected_dtype,
        expected_splits=expected_split_tuple,
        validate_files=validate_files,
    )

    print_json(report)

    if output is not None:
        save_json(report, output)

    if not report["is_valid"]:
        raise SystemExit(1)


def predict(
    checkpoint: str,
    input_path: str,
    output: str | None = None,
    device: str = "auto",
    predictor_type: str = "auto",
) -> None:
    """Run classification or multi-task prediction for one .npy sample."""
    loaded_predictor = create_predictor(
        checkpoint_path=checkpoint,
        device_name=device,
        predictor_type=predictor_type,
    )

    prediction = loaded_predictor.predictor.predict_file(input_path)

    print_json(prediction)

    if output is not None:
        save_json(prediction, output)


def export_onnx(
    checkpoint: str,
    output: str,
    metadata_output: str | None = None,
    device: str = "cpu",
    input_shape: str | None = None,
    opset_version: int = 17,
    dynamic_batch: bool = True,
    validate: bool = True,
    smoke_test: bool = True,
) -> None:
    """Export checkpoint to ONNX."""
    metadata = export_onnx_checkpoint(
        checkpoint_path=checkpoint,
        output_path=output,
        metadata_output_path=metadata_output,
        device_name=device,
        input_shape=parse_input_shape(input_shape),
        opset_version=opset_version,
        dynamic_batch=dynamic_batch,
        validate=validate,
        smoke_test=smoke_test,
    )

    print_json(metadata)


def export_tensorrt(
    onnx: str,
    engine: str,
    metadata_output: str | None = None,
    input_name: str = "features",
    input_shape: str | None = "2,1723,501",
    min_batch_size: int = 1,
    opt_batch_size: int = 1,
    max_batch_size: int = 1,
    fp16: bool = False,
    int8: bool = False,
    verbose: bool = False,
    trtexec_path: str = "trtexec",
    dry_run: bool = False,
) -> None:
    """Export ONNX model to TensorRT engine using trtexec."""
    metadata = export_tensorrt_engine(
        onnx_path=onnx,
        engine_path=engine,
        metadata_output_path=metadata_output,
        input_name=input_name,
        input_shape=parse_input_shape(input_shape),
        min_batch_size=min_batch_size,
        opt_batch_size=opt_batch_size,
        max_batch_size=max_batch_size,
        fp16=fp16,
        int8=int8,
        verbose=verbose,
        trtexec_path=trtexec_path,
        dry_run=dry_run,
    )

    print_json(metadata)


def save_mlflow_model(
    checkpoint: str,
    output: str,
    device: str = "cpu",
    predictor_type: str = "auto",
    overwrite: bool = True,
) -> None:
    """Package checkpoint as MLflow PyFunc model."""
    metadata = save_mlflow_pyfunc_model(
        checkpoint_path=checkpoint,
        output_path=output,
        device_name=device,
        predictor_type=predictor_type,
        overwrite=overwrite,
    )

    print_json(metadata)


def main() -> None:
    fire.Fire(
        {
            "train": train,
            "download-data": download_data,
            "validate-data": validate_data,
            "predict": predict,
            "export-onnx": export_onnx,
            "export-tensorrt": export_tensorrt,
            "save-mlflow-model": save_mlflow_model,
        }
    )


if __name__ == "__main__":
    main()
