from __future__ import annotations

import torch
from torch import nn


class SeismoCNNMultiTask(nn.Module):
    """CNN baseline for joint classification and regression.

    Input:
        [B, 2, T, R]

    Output:
        {
            "logits": [B, num_classes],
            "regression": [B, num_regression_targets]
        }
    """

    def __init__(
        self,
        in_channels: int = 2,
        num_classes: int = 3,
        num_regression_targets: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.classification_head = nn.Linear(128, num_classes)
        self.regression_head = nn.Linear(128, num_regression_targets)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.features(x)
        x = self.pool(x)
        x = self.shared(x)

        return {
            "logits": self.classification_head(x),
            "regression": self.regression_head(x),
        }
