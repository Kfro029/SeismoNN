from __future__ import annotations

import torch
from torch import nn

from seismonn.models.transformer import SinusoidalPositionalEncoding


class MaskedTraceReconstructionTransformer(nn.Module):
    """Transformer model for self-supervised masked trace reconstruction.

    Input:
        x with shape [batch_size, channels, time_steps, receivers]

    Mask:
        boolean tensor with shape [batch_size, receivers]
        True means that receiver trace was masked.

    Output:
        reconstructed tensor with shape [batch_size, channels, time_steps, receivers]
    """

    def __init__(
        self,
        in_channels: int = 2,
        time_steps: int = 1723,
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

        if time_steps <= 0:
            raise ValueError(f"time_steps must be positive, got {time_steps}")

        self.in_channels = in_channels
        self.time_steps = time_steps
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

        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_token, mean=0.0, std=0.02)

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

        self.reconstruction_head = nn.Linear(
            d_model,
            in_channels * time_steps,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor with shape [B, C, T, R].
            mask: Optional boolean tensor with shape [B, R].

        Returns:
            Reconstructed tensor with shape [B, C, T, R].
        """
        if x.ndim != 4:
            raise ValueError(f"Expected input shape [B, C, T, R], got {x.shape}")

        batch_size, channels, time_steps, receivers = x.shape

        if channels != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {channels}."
            )

        if time_steps != self.time_steps:
            raise ValueError(
                f"Expected time_steps={self.time_steps}, got {time_steps}."
            )

        if receivers > self.max_receivers:
            raise ValueError(
                f"Number of receivers {receivers} exceeds max_receivers={self.max_receivers}."
            )

        if mask is not None:
            if mask.shape != (batch_size, receivers):
                raise ValueError(
                    f"Expected mask shape {(batch_size, receivers)}, got {mask.shape}."
                )

            mask = mask.to(dtype=torch.bool, device=x.device)

        # [B, C, T, R] -> [B, R, C, T]
        x = x.permute(0, 3, 1, 2).contiguous()

        # [B, R, C, T] -> [B * R, C, T]
        x = x.view(batch_size * receivers, channels, time_steps)

        # [B * R, C, T] -> [B * R, d_model, 1]
        x = self.temporal_encoder(x)

        # [B * R, d_model, 1] -> [B, R, d_model]
        x = x.squeeze(-1).view(batch_size, receivers, self.d_model)

        if mask is not None:
            mask_token = self.mask_token.expand(batch_size, receivers, self.d_model)
            x = torch.where(mask.unsqueeze(-1), mask_token, x)

        x = self.positional_encoding(x)
        x = self.encoder(x)

        # [B, R, d_model] -> [B, R, C * T]
        reconstructed = self.reconstruction_head(x)

        # [B, R, C * T] -> [B, R, C, T] -> [B, C, T, R]
        reconstructed = reconstructed.view(
            batch_size,
            receivers,
            self.in_channels,
            self.time_steps,
        )
        reconstructed = reconstructed.permute(0, 2, 3, 1).contiguous()

        return reconstructed