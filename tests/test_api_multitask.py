from __future__ import annotations

from io import BytesIO

import numpy as np
import torch
from fastapi.testclient import TestClient

from seismonn.api.main import create_app
from seismonn.models.cnn_multitask import SeismoCNNMultiTask


def create_multitask_test_checkpoint(checkpoint_path):
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


def test_api_predicts_multitask_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    create_multitask_test_checkpoint(checkpoint_path)

    app = create_app(
        checkpoint_path=checkpoint_path,
        device_name="cpu",
        predictor_type="auto",
    )

    with TestClient(app) as client:
        health_response = client.get("/health")

        assert health_response.status_code == 200

        health = health_response.json()

        assert health["status"] == "ok"
        assert health["model_loaded"] is True
        assert health["predictor_type"] == "multitask"
        assert health["model_name"] == "cnn_multitask"
        assert health["startup_error"] is None

        sample = np.random.randn(2, 16, 8).astype("float32")

        buffer = BytesIO()
        np.save(buffer, sample)
        buffer.seek(0)

        predict_response = client.post(
            "/predict",
            files={
                "file": (
                    "sample.npy",
                    buffer,
                    "application/octet-stream",
                )
            },
        )

        assert predict_response.status_code == 200

        prediction = predict_response.json()

        assert prediction["model_name"] == "cnn_multitask"
        assert prediction["input_filename"] == "sample.npy"
        assert prediction["predicted_class_id"] in {0, 1, 2}
        assert prediction["predicted_crack_count"] in {3, 4, 5}
        assert 3.0 <= prediction["expected_crack_count"] <= 5.0

        assert set(prediction["class_probabilities"]) == {"3", "4", "5"}

        assert set(prediction["regression"]) == {
            "mean_length",
            "length_spread",
            "mean_angle_deg",
            "angle_spread_deg",
        }
