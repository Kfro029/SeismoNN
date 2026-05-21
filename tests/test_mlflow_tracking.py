from pathlib import Path

import mlflow

from seismonn.tracking.mlflow import (
    flatten_config,
    is_mlflow_enabled,
    log_mlflow_metrics,
    log_mlflow_params,
    resolve_tracking_uri,
    start_mlflow_run,
)


def test_is_mlflow_enabled():
    assert is_mlflow_enabled({"enabled": True}) is True
    assert is_mlflow_enabled({"enabled": False}) is False
    assert is_mlflow_enabled({}) is False
    assert is_mlflow_enabled(None) is False


def test_flatten_config():
    config = {
        "seed": 42,
        "model": {
            "name": "cnn_baseline",
            "num_classes": 3,
        },
        "data": {
            "input_shape": [2, 1723, 501],
        },
    }

    flattened = flatten_config(config)

    assert flattened["seed"] == "42"
    assert flattened["model.name"] == "cnn_baseline"
    assert flattened["model.num_classes"] == "3"
    assert flattened["data.input_shape"] == "[2, 1723, 501]"


def test_resolve_tracking_uri_for_local_path(tmp_path):
    tracking_uri = resolve_tracking_uri(
        project_root=tmp_path,
        tracking_uri="mlruns",
    )

    assert tracking_uri.startswith("file://")
    assert tracking_uri.endswith("/mlruns")


def test_start_mlflow_run_logs_params_and_metrics(tmp_path):
    tracking_config = {
        "enabled": True,
        "tracking_uri": str(tmp_path / "mlruns"),
        "experiment_name": "test_experiment",
        "run_name": "test_run",
    }

    with start_mlflow_run(
        project_root=Path(tmp_path), tracking_config=tracking_config
    ) as run:
        assert run is not None

        log_mlflow_params(
            {
                "seed": 42,
                "model": {
                    "name": "cnn_baseline",
                },
            }
        )
        log_mlflow_metrics({"accuracy": 0.5}, step=1)

        run_id = run.info.run_id

    client = mlflow.tracking.MlflowClient()
    loaded_run = client.get_run(run_id)

    assert loaded_run.data.params["seed"] == "42"
    assert loaded_run.data.params["model.name"] == "cnn_baseline"
    assert loaded_run.data.metrics["accuracy"] == 0.5
