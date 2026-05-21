import numpy as np
import pandas as pd
import torch

from seismonn.evaluation.checkpoint import (
    build_class_names,
    evaluate_checkpoint,
    normalize_class_mapping,
    save_evaluation_report,
)
from seismonn.models.cnn import SeismoCNN


def test_normalize_class_mapping_defaults():
    mapping = normalize_class_mapping(None)

    assert mapping == {
        0: 3,
        1: 4,
        2: 5,
    }


def test_normalize_class_mapping_converts_keys_and_values_to_int():
    mapping = normalize_class_mapping(
        {
            "0": "3",
            "1": "4",
            "2": "5",
        }
    )

    assert mapping == {
        0: 3,
        1: 4,
        2: 5,
    }


def test_build_class_names():
    class_names = build_class_names(
        {
            0: 3,
            1: 4,
            2: 5,
        }
    )

    assert class_names == ["3", "4", "5"]


def test_evaluate_checkpoint_returns_report(tmp_path):
    sample_paths = []

    for idx in range(6):
        sample = np.random.randn(2, 16, 8).astype("float32")
        sample_path = tmp_path / f"sample_{idx}.npy"
        np.save(sample_path, sample)
        sample_paths.append(sample_path)

    metadata = pd.DataFrame(
        [
            {
                "sample_id": f"{idx:06d}",
                "path": sample_paths[idx].name,
                "filename": sample_paths[idx].name,
                "split": "val",
                "crack_count": [3, 4, 5, 3, 4, 5][idx],
                "class_id": [0, 1, 2, 0, 1, 2][idx],
            }
            for idx in range(6)
        ]
    )

    metadata_path = tmp_path / "metadata.csv"
    metadata.to_csv(metadata_path, index=False)

    model = SeismoCNN(in_channels=2, num_classes=3)

    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model_name": "cnn_baseline",
            "model_state_dict": model.state_dict(),
            "model_config": {
                "name": "cnn_baseline",
                "in_channels": 2,
                "num_classes": 3,
                "dropout": 0.2,
            },
            "data_config": {
                "normalize": True,
            },
            "epoch": 1,
            "metric_to_optimize": "val_macro_f1",
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

    report = evaluate_checkpoint(
        checkpoint_path=checkpoint_path,
        metadata_path=metadata_path,
        split="val",
        data_root=tmp_path,
        batch_size=2,
        num_workers=0,
        device_name="cpu",
    )

    assert report["checkpoint"]["path"] == str(checkpoint_path)
    assert report["checkpoint"]["model_name"] == "cnn_baseline"
    assert report["dataset"]["num_samples"] == 6
    assert report["dataset"]["split"] == "val"
    assert report["device"] == "cpu"

    assert set(report["metrics"]) == {
        "loss",
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "confusion_matrix",
    }

    assert "classification_report" in report
    assert "3" in report["classification_report"]
    assert "4" in report["classification_report"]
    assert "5" in report["classification_report"]


def test_save_evaluation_report(tmp_path):
    output_path = tmp_path / "report.json"

    report = {
        "metrics": {
            "accuracy": 0.5,
        }
    }

    save_evaluation_report(report, output_path)

    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8").strip()