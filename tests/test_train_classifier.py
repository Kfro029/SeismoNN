import pytest
import torch

from seismonn.training.train_classifier import (
    build_optimizer,
    get_optimized_metric,
    load_config,
)


def test_load_config_reads_yaml(tmp_path):
    config_path = tmp_path / "config.yaml"

    config_path.write_text(
        """
seed: 42
device: cpu

data:
  metadata_path: data/metadata.csv
  batch_size: 16

model:
  name: cnn_baseline
  num_classes: 3

optimizer:
  name: adam
  lr: 0.001
  weight_decay: 0.0001

training:
  num_epochs: 3
  metric_to_optimize: val_macro_f1
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config["seed"] == 42
    assert config["device"] == "cpu"
    assert config["data"]["batch_size"] == 16
    assert config["model"]["name"] == "cnn_baseline"
    assert config["optimizer"]["lr"] == 0.001
    assert config["training"]["metric_to_optimize"] == "val_macro_f1"


def test_load_config_rejects_non_dict_yaml(tmp_path):
    config_path = tmp_path / "config.yaml"

    config_path.write_text(
        """
- item_1
- item_2
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Config must be a dictionary"):
        load_config(config_path)


def test_build_optimizer_creates_adam():
    model = torch.nn.Linear(4, 3)

    optimizer = build_optimizer(
        model=model,
        optimizer_config={
            "name": "adam",
            "lr": 0.001,
            "weight_decay": 0.0001,
        },
    )

    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.001)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.0001)


def test_build_optimizer_creates_adamw():
    model = torch.nn.Linear(4, 3)

    optimizer = build_optimizer(
        model=model,
        optimizer_config={
            "name": "adamw",
            "lr": 0.0003,
            "weight_decay": 0.01,
        },
    )

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0003)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.01)


def test_build_optimizer_rejects_unknown_optimizer():
    model = torch.nn.Linear(4, 3)

    with pytest.raises(ValueError, match="Unsupported optimizer"):
        build_optimizer(
            model=model,
            optimizer_config={
                "name": "sgd_unknown",
                "lr": 0.001,
            },
        )


def test_get_optimized_metric_returns_val_accuracy():
    metric = get_optimized_metric(
        metric_name="val_accuracy",
        train_metrics={
            "accuracy": 0.1,
        },
        val_metrics={
            "accuracy": 0.7,
            "macro_f1": 0.6,
        },
    )

    assert metric == pytest.approx(0.7)


def test_get_optimized_metric_returns_val_macro_f1():
    metric = get_optimized_metric(
        metric_name="val_macro_f1",
        train_metrics={
            "accuracy": 0.1,
        },
        val_metrics={
            "accuracy": 0.7,
            "macro_f1": 0.6,
        },
    )

    assert metric == pytest.approx(0.6)


def test_get_optimized_metric_returns_train_accuracy():
    metric = get_optimized_metric(
        metric_name="train_accuracy",
        train_metrics={
            "accuracy": 0.8,
        },
        val_metrics={
            "accuracy": 0.7,
            "macro_f1": 0.6,
        },
    )

    assert metric == pytest.approx(0.8)


def test_get_optimized_metric_rejects_unknown_metric():
    with pytest.raises(ValueError, match="Unsupported metric_to_optimize"):
        get_optimized_metric(
            metric_name="unknown_metric",
            train_metrics={
                "accuracy": 0.8,
            },
            val_metrics={
                "accuracy": 0.7,
                "macro_f1": 0.6,
            },
        )
