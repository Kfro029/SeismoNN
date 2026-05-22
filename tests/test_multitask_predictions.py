import numpy as np
import pandas as pd
import torch

from seismonn.evaluation.multitask_predictions import (
    collect_multitask_predictions,
    save_predictions_csv,
    save_regression_parity_plots,
    summarize_multitask_prediction_table,
)
from seismonn.models.cnn_multitask import SeismoCNNMultiTask


def create_checkpoint(checkpoint_path):
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


def test_collect_multitask_predictions_returns_dataframe(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    create_checkpoint(checkpoint_path)

    metadata_path = create_metadata_and_samples(tmp_path)

    predictions = collect_multitask_predictions(
        checkpoint_path=checkpoint_path,
        metadata_path=metadata_path,
        split="val",
        data_root=tmp_path,
        batch_size=2,
        num_workers=0,
        device_name="cpu",
    )

    assert len(predictions) == 6

    expected_columns = {
        "sample_id",
        "path",
        "true_class_id",
        "predicted_class_id",
        "true_crack_count",
        "predicted_crack_count",
        "expected_crack_count",
        "true_mean_length",
        "pred_mean_length",
        "abs_error_mean_length",
        "true_mean_angle_deg",
        "pred_mean_angle_deg",
        "abs_error_mean_angle_deg",
    }

    assert expected_columns.issubset(set(predictions.columns))


def test_summarize_multitask_prediction_table():
    predictions = pd.DataFrame(
        {
            "true_class_id": [0, 1, 2],
            "predicted_class_id": [0, 1, 1],
            "abs_error_mean_length": [1.0, 2.0, 3.0],
            "squared_error_mean_length": [1.0, 4.0, 9.0],
            "abs_error_mean_angle_deg": [2.0, 4.0, 6.0],
            "squared_error_mean_angle_deg": [4.0, 16.0, 36.0],
        }
    )

    summary = summarize_multitask_prediction_table(
        predictions=predictions,
        regression_columns=["mean_length", "mean_angle_deg"],
    )

    assert summary["num_samples"] == 3
    assert summary["classification"]["accuracy"] == 2 / 3
    assert summary["regression"]["mae_mean"] > 0
    assert "mean_length" in summary["regression"]["per_target"]


def test_save_predictions_and_parity_plots(tmp_path):
    predictions = pd.DataFrame(
        {
            "true_mean_length": [10.0, 20.0, 30.0],
            "pred_mean_length": [11.0, 19.0, 31.0],
        }
    )

    csv_path = tmp_path / "predictions.csv"
    plots_dir = tmp_path / "plots"

    save_predictions_csv(predictions, csv_path)
    plot_paths = save_regression_parity_plots(
        predictions=predictions,
        regression_columns=["mean_length"],
        output_dir=plots_dir,
    )

    assert csv_path.exists()
    assert "mean_length" in plot_paths
    assert (plots_dir / "parity_mean_length.png").exists()
