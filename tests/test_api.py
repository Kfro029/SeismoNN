from __future__ import annotations

from io import BytesIO

import numpy as np
import torch
from fastapi.testclient import TestClient

from seismonn.api.main import create_app
from seismonn.models.cnn import SeismoCNN


def create_test_checkpoint(checkpoint_path):
    model = SeismoCNN(in_channels=2, num_classes=3)

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
            "input_shape": [2, 16, 8],
            "class_id_to_crack_count": {
                0: 3,
                1: 4,
                2: 5,
            },
        },
        checkpoint_path,
    )


def test_api_health_and_predict(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    create_test_checkpoint(checkpoint_path)

    app = create_app(checkpoint_path=checkpoint_path, device_name="cpu")

    with TestClient(app) as client:
        health_response = client.get("/health")

        assert health_response.status_code == 200

        health = health_response.json()

        assert health["status"] == "ok"
        assert health["model_loaded"] is True
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

        assert prediction["model_name"] == "cnn_baseline"
        assert prediction["input_filename"] == "sample.npy"
        assert prediction["predicted_class_id"] in {0, 1, 2}
        assert prediction["predicted_crack_count"] in {3, 4, 5}

        probabilities = prediction["class_probabilities"]

        assert set(probabilities) == {"3", "4", "5"}
        assert all(0.0 <= value <= 1.0 for value in probabilities.values())
        assert abs(sum(probabilities.values()) - 1.0) < 1e-6


def test_api_rejects_non_npy_file(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    create_test_checkpoint(checkpoint_path)

    app = create_app(checkpoint_path=checkpoint_path, device_name="cpu")

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            files={
                "file": (
                    "sample.txt",
                    b"not a numpy file",
                    "text/plain",
                )
            },
        )

        assert response.status_code == 400
