from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow


def is_mlflow_enabled(tracking_config: dict[str, Any] | None) -> bool:
    """Check whether MLflow tracking is enabled."""
    if tracking_config is None:
        return False

    return bool(tracking_config.get("enabled", False))


def resolve_tracking_uri(project_root: Path, tracking_uri: str | Path) -> str:
    """Resolve MLflow tracking URI.

    Local relative paths are converted to file:// URIs.
    Remote URIs such as http://... are left unchanged.
    """
    tracking_uri_str = str(tracking_uri)

    if tracking_uri_str.startswith(("http://", "https://", "file://")):
        return tracking_uri_str

    tracking_path = Path(tracking_uri_str)

    if not tracking_path.is_absolute():
        tracking_path = project_root / tracking_path

    return tracking_path.resolve().as_uri()


def _param_value_to_string(value: Any) -> str:
    """Convert config value to a stable MLflow parameter string."""
    if isinstance(value, str | int | float | bool) or value is None:
        return str(value)

    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def flatten_config(
    config: dict[str, Any],
    prefix: str = "",
) -> dict[str, str]:
    """Flatten nested config dictionary for MLflow params."""
    flattened: dict[str, str] = {}

    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)

        if isinstance(value, dict):
            flattened.update(flatten_config(value, prefix=full_key))
        else:
            flattened[full_key] = _param_value_to_string(value)

    return flattened


@contextmanager
def start_mlflow_run(
    project_root: Path,
    tracking_config: dict[str, Any] | None,
) -> Iterator[Any | None]:
    """Start MLflow run if tracking is enabled.

    If tracking is disabled, yields None.
    """
    if not is_mlflow_enabled(tracking_config):
        yield None
        return

    assert tracking_config is not None

    tracking_uri = resolve_tracking_uri(
        project_root=project_root,
        tracking_uri=tracking_config.get("tracking_uri", "mlruns"),
    )

    experiment_name = str(tracking_config.get("experiment_name", "seismonn"))
    run_name = str(tracking_config.get("run_name", "run"))

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("project", "SeismoNN")
        mlflow.set_tag("stage", "training")
        yield run


def log_mlflow_params(config: dict[str, Any]) -> None:
    """Log flattened config parameters to active MLflow run."""
    params = flatten_config(config)

    # MLflow parameter keys have length limits; our config keys are short,
    # but we still keep this function as a single controlled logging point.
    mlflow.log_params(params)


def log_mlflow_metrics(
    metrics: dict[str, float],
    step: int | None = None,
) -> None:
    """Log scalar metrics to active MLflow run."""
    mlflow.log_metrics(
        {key: float(value) for key, value in metrics.items()},
        step=step,
    )


def log_mlflow_artifacts(output_dir: str | Path) -> None:
    """Log all training artifacts from output_dir."""
    output_dir = Path(output_dir)

    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory does not exist: {output_dir}")

    mlflow.log_artifacts(str(output_dir))
