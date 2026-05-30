import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import fire
from omegaconf import DictConfig

from seismonn.config import load_hydra_config, parse_overrides
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


def parse_input_shape(input_shape: Any | None) -> tuple[int, int, int] | None:
    """Parse input shape from Hydra/CLI value."""
    if input_shape is None:
        return None

    if not isinstance(input_shape, str):
        try:
            parts = tuple(int(part) for part in input_shape)
        except TypeError:
            parts = ()
        if parts:
            if len(parts) != 3:
                raise ValueError(
                    f"Expected input_shape with 3 dimensions, got {input_shape!r}."
                )
            return parts

    normalized = str(input_shape).strip().strip("()[]")
    normalized = normalized.replace("x", ",").replace(" ", "")
    parts = [part for part in normalized.split(",") if part]

    if len(parts) != 3:
        raise ValueError(
            f"Expected input_shape with 3 dimensions, got {input_shape!r}."
        )

    return tuple(int(part) for part in parts)


def parse_splits(value: Any) -> tuple[str, ...]:
    """Parse expected split names from Hydra/CLI value."""
    if isinstance(value, str):
        normalized = value.strip().strip("[]()")
        normalized = normalized.replace("'", "").replace('"', "")
        return tuple(split.strip() for split in normalized.split(",") if split.strip())

    try:
        return tuple(str(split) for split in value)
    except TypeError as error:
        raise TypeError(f"Unsupported split specification: {value!r}") from error


def get_config(overrides: str = "", config_name: str = "config") -> DictConfig:
    """Load Hydra config from project configs directory."""
    return load_hydra_config(
        overrides=parse_overrides(overrides),
        config_name=config_name,
    )


def train(overrides: str = "") -> None:
    """Run Hydra + PyTorch Lightning training pipeline.

    Example:
        uv run seismonn train --overrides "trainer.max_epochs=1 tracking.enabled=false"
    """
    command = [
        sys.executable,
        "scripts/train_lightning.py",
    ]

    command.extend(parse_overrides(overrides))

    run_command(command)


def download_data(overrides: str = "", config_name: str = "config") -> None:
    """Ensure data and metadata are available using Hydra config."""
    config = get_config(overrides=overrides, config_name=config_name)

    ensure_data_available(
        metadata_path=str(config.data.metadata_path),
        data_dir=str(config.data.data_dir),
        repo_root=str(config.data.repo_root),
        use_dvc=bool(config.data.use_dvc),
        dvc_remote=str(config.data.dvc_remote),
        huggingface_repo_id=str(config.data.huggingface_repo_id),
        allow_huggingface_fallback=bool(config.data.allow_huggingface_fallback),
    )

    print_json(
        {
            "status": "ok",
            "metadata_path": str(config.data.metadata_path),
            "data_dir": str(config.data.data_dir),
            "repo_root": str(config.data.repo_root),
        }
    )


def validate_data(overrides: str = "", config_name: str = "config") -> None:
    """Validate metadata.csv and optionally referenced .npy files."""
    config = get_config(overrides=overrides, config_name=config_name)

    report = validate_metadata(
        metadata_path=str(config.validation.metadata_path),
        data_root=str(config.validation.data_root),
        expected_shape=parse_input_shape(config.validation.expected_shape),
        expected_dtype=str(config.validation.expected_dtype),
        expected_splits=parse_splits(config.validation.expected_splits),
        validate_files=bool(config.validation.validate_files),
    )

    print_json(report)

    if config.validation.output is not None:
        save_json(report, str(config.validation.output))

    if not report["is_valid"]:
        raise SystemExit(1)


def predict(overrides: str = "", config_name: str = "config") -> None:
    """Run classification or multi-task prediction from Hydra config."""
    config = get_config(overrides=overrides, config_name=config_name)

    if config.inference.input_path is None:
        raise ValueError(
            "inference.input_path is required. "
            'Example: --overrides "inference.input_path=2nd_selection/sample.npy"'
        )

    if bool(config.inference.get("ensure_data", False)):
        ensure_data_available(
            metadata_path=str(config.data.metadata_path),
            data_dir=str(config.data.data_dir),
            repo_root=str(config.data.repo_root),
            use_dvc=bool(config.data.use_dvc),
            dvc_remote=str(config.data.dvc_remote),
            huggingface_repo_id=str(config.data.huggingface_repo_id),
            allow_huggingface_fallback=bool(config.data.allow_huggingface_fallback),
        )

    loaded_predictor = create_predictor(
        checkpoint_path=str(config.inference.checkpoint),
        device_name=str(config.inference.device),
        predictor_type=str(config.inference.predictor_type),
    )

    prediction = loaded_predictor.predictor.predict_file(
        str(config.inference.input_path)
    )

    print_json(prediction)

    if config.inference.output is not None:
        save_json(prediction, str(config.inference.output))


def export_onnx(overrides: str = "", config_name: str = "config") -> None:
    """Export checkpoint to ONNX using Hydra config."""
    config = get_config(overrides=overrides, config_name=config_name)
    onnx_config = config.export.onnx

    metadata = export_onnx_checkpoint(
        checkpoint_path=str(onnx_config.checkpoint),
        output_path=str(onnx_config.output),
        metadata_output_path=(
            None
            if onnx_config.metadata_output is None
            else str(onnx_config.metadata_output)
        ),
        device_name=str(onnx_config.device),
        input_shape=parse_input_shape(onnx_config.input_shape),
        opset_version=int(onnx_config.opset_version),
        dynamic_batch=bool(onnx_config.dynamic_batch),
        validate=bool(onnx_config.validate),
        smoke_test=bool(onnx_config.smoke_test),
    )

    print_json(metadata)


def export_tensorrt(overrides: str = "", config_name: str = "config") -> None:
    """Export ONNX model to TensorRT engine using Hydra config."""
    config = get_config(overrides=overrides, config_name=config_name)
    tensorrt_config = config.export.tensorrt

    metadata = export_tensorrt_engine(
        onnx_path=str(tensorrt_config.onnx),
        engine_path=str(tensorrt_config.engine),
        metadata_output_path=(
            None
            if tensorrt_config.metadata_output is None
            else str(tensorrt_config.metadata_output)
        ),
        input_name=str(tensorrt_config.input_name),
        input_shape=parse_input_shape(tensorrt_config.input_shape),
        min_batch_size=int(tensorrt_config.min_batch_size),
        opt_batch_size=int(tensorrt_config.opt_batch_size),
        max_batch_size=int(tensorrt_config.max_batch_size),
        fp16=bool(tensorrt_config.fp16),
        int8=bool(tensorrt_config.int8),
        verbose=bool(tensorrt_config.verbose),
        trtexec_path=str(tensorrt_config.trtexec_path),
        dry_run=bool(tensorrt_config.dry_run),
    )

    print_json(metadata)


def save_mlflow_model(overrides: str = "", config_name: str = "config") -> None:
    """Package checkpoint as MLflow PyFunc model using Hydra config."""
    config = get_config(overrides=overrides, config_name=config_name)
    mlflow_model_config = config.serving.mlflow_model

    metadata = save_mlflow_pyfunc_model(
        checkpoint_path=str(mlflow_model_config.checkpoint),
        output_path=str(mlflow_model_config.output),
        device_name=str(mlflow_model_config.device),
        predictor_type=str(mlflow_model_config.predictor_type),
        overwrite=bool(mlflow_model_config.overwrite),
    )

    print_json(metadata)


def serve_mlflow_model(overrides: str = "", config_name: str = "config") -> None:
    """Serve packaged MLflow model using Hydra config."""
    config = get_config(overrides=overrides, config_name=config_name)
    server_config = config.serving.mlflow_server

    command = [
        "mlflow",
        "models",
        "serve",
        "-m",
        str(server_config.model_uri),
        "--host",
        str(server_config.host),
        "--port",
        str(server_config.port),
    ]

    if bool(server_config.no_conda):
        command.append("--no-conda")

    run_command([sys.executable, "-m", *command])


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
            "serve-mlflow-model": serve_mlflow_model,
        }
    )


if __name__ == "__main__":
    main()
