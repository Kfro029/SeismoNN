import torch

from seismonn.inference.factory import (
    CLASSIFICATION_PREDICTOR_TYPE,
    MULTITASK_PREDICTOR_TYPE,
    create_predictor,
    infer_predictor_type_from_checkpoint,
)
from seismonn.models.cnn import SeismoCNN
from seismonn.models.cnn_multitask import SeismoCNNMultiTask


def test_infer_predictor_type_for_classification_checkpoint(tmp_path):
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
            "input_shape": [2, 16, 8],
            "class_id_to_crack_count": {
                0: 3,
                1: 4,
                2: 5,
            },
        },
        checkpoint_path,
    )

    predictor_type, model_name = infer_predictor_type_from_checkpoint(
        checkpoint_path=checkpoint_path,
        device_name="cpu",
    )

    assert predictor_type == CLASSIFICATION_PREDICTOR_TYPE
    assert model_name == "cnn_baseline"


def test_infer_predictor_type_for_multitask_checkpoint(tmp_path):
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

    predictor_type, model_name = infer_predictor_type_from_checkpoint(
        checkpoint_path=checkpoint_path,
        device_name="cpu",
    )

    assert predictor_type == MULTITASK_PREDICTOR_TYPE
    assert model_name == "cnn_multitask"


def test_create_predictor_auto_for_multitask_checkpoint(tmp_path):
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

    loaded_predictor = create_predictor(
        checkpoint_path=checkpoint_path,
        device_name="cpu",
        predictor_type="auto",
    )

    assert loaded_predictor.predictor_type == MULTITASK_PREDICTOR_TYPE
    assert loaded_predictor.model_name == "cnn_multitask"
