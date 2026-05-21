from __future__ import annotations

from typing import Any

from torch import nn

from seismonn.models.cnn import SeismoCNN
from seismonn.models.transformer import TraceTransformerClassifier


def create_model(model_config: dict[str, Any]) -> nn.Module:
    """Create model from config."""
    model_name = str(model_config.get("name", "cnn_baseline")).lower()

    if model_name == "cnn_baseline":
        return SeismoCNN(
            in_channels=int(model_config.get("in_channels", 2)),
            num_classes=int(model_config.get("num_classes", 3)),
            dropout=float(model_config.get("dropout", 0.2)),
        )

    if model_name in {"trace_transformer", "transformer_encoder"}:
        return TraceTransformerClassifier(
            in_channels=int(model_config.get("in_channels", 2)),
            num_classes=int(model_config.get("num_classes", 3)),
            d_model=int(model_config.get("d_model", 128)),
            nhead=int(model_config.get("nhead", 4)),
            num_layers=int(model_config.get("num_layers", 2)),
            dim_feedforward=int(model_config.get("dim_feedforward", 256)),
            dropout=float(model_config.get("dropout", 0.1)),
            temporal_hidden_channels=int(
                model_config.get("temporal_hidden_channels", 32)
            ),
            max_receivers=int(model_config.get("max_receivers", 1024)),
        )

    raise ValueError(f"Unsupported model name: {model_name}")
