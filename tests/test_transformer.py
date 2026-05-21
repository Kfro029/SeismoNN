import pytest
import torch

from seismonn.models.transformer import (
    SinusoidalPositionalEncoding,
    TraceTransformerClassifier,
)


def test_sinusoidal_positional_encoding_preserves_shape():
    positional_encoding = SinusoidalPositionalEncoding(d_model=16, max_len=32)

    x = torch.zeros(2, 10, 16)
    y = positional_encoding(x)

    assert y.shape == torch.Size([2, 10, 16])


def test_trace_transformer_forward_shape():
    model = TraceTransformerClassifier(
        in_channels=2,
        num_classes=3,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        dropout=0.1,
        temporal_hidden_channels=16,
        max_receivers=64,
    )
    model.eval()

    x = torch.randn(2, 2, 128, 32)

    with torch.no_grad():
        logits = model(x)

    assert logits.shape == torch.Size([2, 3])
    assert logits.dtype == torch.float32


def test_trace_transformer_rejects_wrong_number_of_channels():
    model = TraceTransformerClassifier(
        in_channels=2,
        num_classes=3,
        d_model=32,
        nhead=4,
        num_layers=1,
        max_receivers=64,
    )

    x = torch.randn(2, 3, 128, 32)

    with pytest.raises(ValueError, match="Expected 2 input channels"):
        model(x)


def test_trace_transformer_rejects_too_many_receivers():
    model = TraceTransformerClassifier(
        in_channels=2,
        num_classes=3,
        d_model=32,
        nhead=4,
        num_layers=1,
        max_receivers=16,
    )

    x = torch.randn(2, 2, 128, 32)

    with pytest.raises(ValueError, match="exceeds max_receivers"):
        model(x)
