import numpy as np
import torch

from seismonn.inference.predictor import SeismoPredictor
from seismonn.models.cnn import SeismoCNN


def test_seismo_predictor_predicts_file(tmp_path):
    sample = np.random.randn(2, 16, 8).astype("float32")
    sample_path = tmp_path / "sample.npy"
    np.save(sample_path, sample)

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
            "input_shape": [2, 16, 8],
            "class_id_to_crack_count": {
                0: 3,
                1: 4,
                2: 5,
            },
        },
        checkpoint_path,
    )

    predictor = SeismoPredictor.from_checkpoint(checkpoint_path, device_name="cpu")
    prediction = predictor.predict_file(sample_path)

    assert prediction["model_name"] == "cnn_baseline"
    assert prediction["input_path"] == str(sample_path)
    assert prediction["predicted_class_id"] in {0, 1, 2}
    assert prediction["predicted_crack_count"] in {3, 4, 5}

    probabilities = prediction["class_probabilities"]

    assert set(probabilities) == {"3", "4", "5"}
    assert all(0.0 <= value <= 1.0 for value in probabilities.values())
    assert abs(sum(probabilities.values()) - 1.0) < 1e-6
