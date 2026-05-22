import numpy as np
import pandas as pd
import torch

from seismonn.evaluation.multitask_checkpoint import (
    build_class_names,
    evaluate_multitask_checkpoint,
    load_multitask_model_from_checkpoint,
    save_multitask_evaluation_report,
)
from seismonn.models.cnn_multitask import SeismoCNNMultiTask


def create_multitask_checkpoint(checkpoint_path):
    model = SeismoCNNMultiTask(
        in_channels=2,
        num_classes=3,
        num_regression_targets=4,
    )

    torch.save(
        {
            "model_name": "cnn_multitask",
            "model_state_dict": model.state_dict(),
            "model_config": {
                "name": "cnn_multitask",
                "in_channels": 2,
                "num_classes": 3,
                "num_regression_targets": 4,
                "dropout": 0.2,
            },
            "data_config": {
                "normalize": True,
            },
            "target_scaler": {
                "target_columns": [
                    "mean_length",
                    "length_spread",
                    "mean_angle_deg",
                    "angle_spread_deg",
                ],
                "mean": [30.0, 2.0, 0.0, 4.0],
                "std": [5.0, 1.0, 10.0, 2.0],
            },
            "regression_columns": [
                "mean_length",
                "length_spread",
                "mean_angle_deg",
                "angle_spread_deg",
            ],
            "epoch": 1,
            "metric_to_optimize": "val_classification_macro_f1",
            "best_metric": 0.1,
            "input_shape": [2, 16, 8],
            "class_id_to_crack_count": {
                0: 3,
                1: 4,
                2: 5,
            },
        },
        checkpoint_path,
    )


def create_metadata_and_samples(tmp_path):
    rows = []

    class_ids = [0, 1, 2, 0, 1, 2]
    crack_counts = [3, 4, 5, 3, 4, 5]

    for idx in range(6):
        sample = np.random.randn(2, 16, 8).astype("float32")
        sample_path = tmp_path / f"sample_{idx}.npy"
        np.save(sample_path, sample)

        rows.append(
            {
                "sample_id": f"{idx:06d}",
                "path": sample_path.name,
                "filename": sample_path.name,
                "split": "val",
                "crack_count": crack_counts[idx],
                "class_id": class_ids[idx],
                "mean_length": 30.0 + idx,
                "length_spread": 2.0,
                "mean_angle_deg": -10.0 + idx,
                "angle_spread_deg": 4.0,
            }
        )

    metadata = pd.DataFrame(rows)
    metadata_path = tmp_path / "metadata.csv"
    metadata.to_csv(metadata_path, index=False)

    return metadata_path


def test_build_class_names():
    class_names = build_class_names(
        {
            0: 3,
            1: 4,
            2: 5,
        }
    )

    assert class_names == ["3", "4", "5"]


def test_load_multitask_model_from_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    create_multitask_checkpoint(checkpoint_path)

    model, checkpoint, target_scaler = load_multitask_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=torch.device("cpu"),
    )

    assert isinstance(model, SeismoCNNMultiTask)
    assert checkpoint["model_name"] == "cnn_multitask"
    assert target_scaler.target_columns == [
        "mean_length",
        "length_spread",
        "mean_angle_deg",
        "angle_spread_deg",
    ]


def test_evaluate_multitask_checkpoint_returns_report(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    create_multitask_checkpoint(checkpoint_path)

    metadata_path = create_metadata_and_samples(tmp_path)

    report = evaluate_multitask_checkpoint(
        checkpoint_path=checkpoint_path,
        metadata_path=metadata_path,
        split="val",
        data_root=tmp_path,
        batch_size=2,
        num_workers=0,
        device_name="cpu",
    )

    assert report["checkpoint"]["path"] == str(checkpoint_path)
    assert report["checkpoint"]["model_name"] == "cnn_multitask"
    assert report["dataset"]["num_samples"] == 6
    assert report["dataset"]["split"] == "val"
    assert report["device"] == "cpu"

    metrics = report["metrics"]

    assert "classification_accuracy" in metrics
    assert "classification_macro_f1" in metrics
    assert "regression_mae_mean" in metrics
    assert "regression_rmse_mean" in metrics
    assert "per_target" in metrics

    # Aliases for model comparison.
    assert "accuracy" in metrics
    assert "macro_f1" in metrics

    assert "classification_report" in report
    assert "3" in report["classification_report"]
    assert "4" in report["classification_report"]
    assert "5" in report["classification_report"]


def test_save_multitask_evaluation_report(tmp_path):
    output_path = tmp_path / "report.json"

    report = {
        "metrics": {
            "classification_accuracy": 0.5,
            "regression_mae_mean": 1.0,
        }
    }

    save_multitask_evaluation_report(report, output_path)

    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8").strip()
