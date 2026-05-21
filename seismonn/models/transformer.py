from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for receiver tokens."""

    def __init__(
        self,
        d_model: int,
        max_len: int = 2048,
    ) -> None:
        super().__init__()

        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)

        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding.

        Args:
            x: Tensor with shape [batch_size, seq_len, d_model].
        """
        seq_len = x.size(1)

        if seq_len > self.pe.size(1):
            raise ValueError(
                f"Sequence length {seq_len} exceeds max positional encoding "
                f"length {self.pe.size(1)}."
            )

        return x + self.pe[:, :seq_len, :]


class TraceTransformerClassifier(nn.Module):
    """Transformer encoder classifier for seismic receiver traces.

    Input shape:
        [batch_size, 2, time_steps, receivers]

    Internal representation:
        each receiver trace is converted into one token.

    Output shape:
        [batch_size, num_classes]
    """

    def __init__(
        self,
        in_channels: int = 2,
        num_classes: int = 3,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        temporal_hidden_channels: int = 32,
        max_receivers: int = 1024,
    ) -> None:
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError(
                f"d_model must be divisible by nhead. "
                f"Got d_model={d_model}, nhead={nhead}."
            )

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.d_model = d_model
        self.max_receivers = max_receivers

        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=temporal_hidden_channels,
                kernel_size=7,
                stride=2,
                padding=3,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(
                in_channels=temporal_hidden_channels,
                out_channels=d_model,
                kernel_size=7,
                stride=2,
                padding=3,
            ),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )

        self.positional_encoding = SinusoidalPositionalEncoding(
            d_model=d_model,
            max_len=max_receivers,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor with shape [B, C, T, R].

        Returns:
            Logits with shape [B, num_classes].
        """
        if x.ndim != 4:
            raise ValueError(f"Expected input shape [B, C, T, R], got {x.shape}")

        batch_size, channels, _time_steps, receivers = x.shape

        if channels != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {channels}."
            )

        if receivers > self.max_receivers:
            raise ValueError(
                f"Number of receivers {receivers} exceeds max_receivers={self.max_receivers}."
            )

        # [B, C, T, R] -> [B, R, C, T]
        x = x.permute(0, 3, 1, 2).contiguous()

        # Process each receiver trace independently:
        # [B, R, C, T] -> [B * R, C, T]
        x = x.view(batch_size * receivers, channels, x.size(-1))

        # [B * R, C, T] -> [B * R, d_model, 1]
        x = self.temporal_encoder(x)

        # [B * R, d_model, 1] -> [B, R, d_model]
        x = x.squeeze(-1).view(batch_size, receivers, self.d_model)

        x = self.positional_encoding(x)
        x = self.encoder(x)

        # Mean pooling over receiver tokens.
        x = x.mean(dim=1)

        logits = self.classifier(x)
        return logits
