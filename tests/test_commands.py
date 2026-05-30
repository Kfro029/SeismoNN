import numpy as np
import torch

from seismonn.commands import parse_input_shape, predict
from seismonn.models.cnn import SeismoCNN


def test_parse_input_shape_accepts_comma_separated_shape():
    assert parse_input_shape("2,1723,501") == (2, 1723, 501)


def test_parse_input_shape_accepts_x_separated_shape():
    assert parse_input_shape("2x1723x501") == (2, 1723, 501)


def test_predict_command_writes_output_json(tmp_path):
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

    output_path = tmp_path / "prediction.json"

    predict(
        overrides=[
            f"inference.checkpoint={checkpoint_path.as_posix()}",
            f"inference.input_path={sample_path.as_posix()}",
            f"inference.output={output_path.as_posix()}",
            "inference.device=cpu",
            "inference.predictor_type=auto",
        ]
    )

    assert output_path.exists()
    assert "predicted_crack_count" in output_path.read_text(encoding="utf-8")
