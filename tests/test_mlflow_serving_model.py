import numpy as np
import pandas as pd
import torch
import mlflow.pyfunc

from seismonn.models.cnn import SeismoCNN
from seismonn.models.cnn_multitask import SeismoCNNMultiTask
from seismonn.serving.mlflow_model import (
    flatten_prediction,
    save_mlflow_pyfunc_model,
)
from seismonn.serving.mlflow_model import _extract_input_paths


def test_flatten_prediction():
    prediction = {
        "predicted_crack_count": 4,
        "class_probabilities": {
            "3": 0.1,
            "4": 0.8,
            "5": 0.1,
        },
        "regression": {
            "mean_length": 30.0,
        },
    }

    flattened = flatten_prediction(prediction)

    assert flattened["predicted_crack_count"] == 4
    assert flattened["class_probabilities_3"] == 0.1
    assert flattened["class_probabilities_4"] == 0.8
    assert flattened["class_probabilities_5"] == 0.1
    assert flattened["regression_mean_length"] == 30.0


def test_save_and_load_classification_mlflow_pyfunc_model(tmp_path):
    sample = np.random.randn(2, 16, 8).astype("float32")
    sample_path = tmp_path / "sample.npy"
    np.save(sample_path, sample)

    model = SeismoCNN(in_channels=2, num_classes=3)
    checkpoint_path = tmp_path / "classification.pt"

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

    mlflow_model_path = tmp_path / "mlflow_classification_model"

    save_mlflow_pyfunc_model(
        checkpoint_path=checkpoint_path,
        output_path=mlflow_model_path,
        device_name="cpu",
        predictor_type="auto",
    )

    loaded_model = mlflow.pyfunc.load_model(str(mlflow_model_path))

    output = loaded_model.predict(
        pd.DataFrame(
            [
                {
                    "input_path": str(sample_path),
                }
            ]
        )
    )

    assert len(output) == 1
    assert output.loc[0, "model_name"] == "cnn_baseline"
    assert output.loc[0, "predictor_type"] == "classification"
    assert output.loc[0, "predicted_crack_count"] in {3, 4, 5}
    assert "class_probabilities_3" in output.columns
    assert "class_probabilities_4" in output.columns
    assert "class_probabilities_5" in output.columns


def test_save_and_load_multitask_mlflow_pyfunc_model(tmp_path):
    sample = np.random.randn(2, 16, 8).astype("float32")
    sample_path = tmp_path / "sample.npy"
    np.save(sample_path, sample)

    model = SeismoCNNMultiTask(
        in_channels=2,
        num_classes=3,
        num_regression_targets=4,
    )
    checkpoint_path = tmp_path / "multitask.pt"

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

    mlflow_model_path = tmp_path / "mlflow_multitask_model"

    save_mlflow_pyfunc_model(
        checkpoint_path=checkpoint_path,
        output_path=mlflow_model_path,
        device_name="cpu",
        predictor_type="auto",
    )

    loaded_model = mlflow.pyfunc.load_model(str(mlflow_model_path))

    output = loaded_model.predict(
        pd.DataFrame(
            [
                {
                    "input_path": str(sample_path),
                }
            ]
        )
    )

    assert len(output) == 1
    assert output.loc[0, "model_name"] == "cnn_multitask"
    assert output.loc[0, "predictor_type"] == "multitask"
    assert output.loc[0, "predicted_crack_count"] in {3, 4, 5}
    assert "regression_mean_length" in output.columns
    assert "regression_length_spread" in output.columns
    assert "regression_mean_angle_deg" in output.columns
    assert "regression_angle_spread_deg" in output.columns


def test_extract_input_paths_from_numpy_array():
    model_input = np.array([["sample_0.npy"], ["sample_1.npy"]])

    input_paths = _extract_input_paths(model_input)

    assert input_paths == ["sample_0.npy", "sample_1.npy"]
