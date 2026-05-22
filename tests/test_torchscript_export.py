import torch

from seismonn.exporting.torchscript import export_torchscript_checkpoint
from seismonn.models.cnn import SeismoCNN
from seismonn.models.cnn_multitask import SeismoCNNMultiTask


def test_export_classification_checkpoint_to_torchscript(tmp_path):
    model = SeismoCNN(in_channels=2, num_classes=3)

    checkpoint_path = tmp_path / "classification.pt"
    torchscript_path = tmp_path / "classification_torchscript.pt"
    metadata_path = tmp_path / "classification_torchscript.metadata.json"

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
            "epoch": 1,
        },
        checkpoint_path,
    )

    metadata = export_torchscript_checkpoint(
        checkpoint_path=checkpoint_path,
        output_path=torchscript_path,
        metadata_output_path=metadata_path,
        device_name="cpu",
    )

    assert torchscript_path.exists()
    assert metadata_path.exists()
    assert metadata["model_name"] == "cnn_baseline"
    assert metadata["output_format"] == "logits"

    scripted_model = torch.jit.load(str(torchscript_path))
    scripted_model.eval()

    x = torch.randn(1, 2, 16, 8)

    with torch.no_grad():
        logits = scripted_model(x)

    assert logits.shape == torch.Size([1, 3])


def test_export_multitask_checkpoint_to_torchscript(tmp_path):
    model = SeismoCNNMultiTask(
        in_channels=2,
        num_classes=3,
        num_regression_targets=4,
    )

    checkpoint_path = tmp_path / "multitask.pt"
    torchscript_path = tmp_path / "multitask_torchscript.pt"
    metadata_path = tmp_path / "multitask_torchscript.metadata.json"

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
            "input_shape": [2, 16, 8],
            "class_id_to_crack_count": {
                0: 3,
                1: 4,
                2: 5,
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
        },
        checkpoint_path,
    )

    metadata = export_torchscript_checkpoint(
        checkpoint_path=checkpoint_path,
        output_path=torchscript_path,
        metadata_output_path=metadata_path,
        device_name="cpu",
    )

    assert torchscript_path.exists()
    assert metadata_path.exists()
    assert metadata["model_name"] == "cnn_multitask"
    assert metadata["output_format"] == "tuple(logits, regression)"

    scripted_model = torch.jit.load(str(torchscript_path))
    scripted_model.eval()

    x = torch.randn(1, 2, 16, 8)

    with torch.no_grad():
        logits, regression = scripted_model(x)

    assert logits.shape == torch.Size([1, 3])
    assert regression.shape == torch.Size([1, 4])