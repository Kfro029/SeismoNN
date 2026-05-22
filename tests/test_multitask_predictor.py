import numpy as np
import torch

from seismonn.inference.multitask_predictor import SeismoMultiTaskPredictor
from seismonn.models.cnn_multitask import SeismoCNNMultiTask


def test_multitask_predictor_predicts_classification_and_regression(tmp_path):
    sample = np.random.randn(2, 16, 8).astype("float32")
    sample_path = tmp_path / "sample.npy"
    np.save(sample_path, sample)

    model = SeismoCNNMultiTask(
        in_channels=2,
        num_classes=3,
        num_regression_targets=4,
    )

    checkpoint_path = tmp_path / "checkpoint.pt"

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

    predictor = SeismoMultiTaskPredictor.from_checkpoint(
        checkpoint_path=checkpoint_path,
        device_name="cpu",
    )

    prediction = predictor.predict_file(sample_path)

    assert prediction["model_name"] == "cnn_multitask"
    assert prediction["input_path"] == str(sample_path)
    assert prediction["predicted_class_id"] in {0, 1, 2}
    assert prediction["predicted_crack_count"] in {3, 4, 5}
    assert 3.0 <= prediction["expected_crack_count"] <= 5.0

    probabilities = prediction["class_probabilities"]

    assert set(probabilities) == {"3", "4", "5"}
    assert all(0.0 <= value <= 1.0 for value in probabilities.values())
    assert abs(sum(probabilities.values()) - 1.0) < 1e-6

    regression = prediction["regression"]

    assert set(regression) == {
        "mean_length",
        "length_spread",
        "mean_angle_deg",
        "angle_spread_deg",
    }

    assert all(isinstance(value, float) for value in regression.values())
