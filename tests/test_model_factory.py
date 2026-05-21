import pytest

from seismonn.models.cnn import SeismoCNN
from seismonn.models.factory import create_model
from seismonn.models.transformer import TraceTransformerClassifier


def test_create_cnn_model():
    model = create_model(
        {
            "name": "cnn_baseline",
            "in_channels": 2,
            "num_classes": 3,
            "dropout": 0.2,
        }
    )

    assert isinstance(model, SeismoCNN)


def test_create_transformer_model():
    model = create_model(
        {
            "name": "trace_transformer",
            "in_channels": 2,
            "num_classes": 3,
            "d_model": 32,
            "nhead": 4,
            "num_layers": 1,
            "dim_feedforward": 64,
            "dropout": 0.1,
            "temporal_hidden_channels": 16,
            "max_receivers": 64,
        }
    )

    assert isinstance(model, TraceTransformerClassifier)


def test_create_model_rejects_unknown_model_name():
    with pytest.raises(ValueError, match="Unsupported model name"):
        create_model({"name": "unknown_model"})
