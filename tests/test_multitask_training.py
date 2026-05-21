import numpy as np
import torch
from torch import nn

from seismonn.data.multitask_dataset import RegressionTargetScaler
from seismonn.training.multitask import (
    compute_multitask_loss,
    compute_regression_metrics,
)


def test_compute_multitask_loss():
    outputs = {
        "logits": torch.randn(4, 3),
        "regression": torch.randn(4, 4),
    }

    targets = {
        "class_id": torch.tensor([0, 1, 2, 1], dtype=torch.long),
        "regression": torch.randn(4, 4),
    }

    total_loss, parts = compute_multitask_loss(
        outputs=outputs,
        targets=targets,
        classification_criterion=nn.CrossEntropyLoss(),
        regression_criterion=nn.MSELoss(),
        regression_loss_weight=0.5,
    )

    assert total_loss.item() > 0
    assert parts["classification_loss"] > 0
    assert parts["regression_loss"] > 0
    assert parts["total_loss"] > 0


def test_compute_regression_metrics():
    scaler = RegressionTargetScaler(
        target_columns=[
            "mean_length",
            "length_spread",
            "mean_angle_deg",
            "angle_spread_deg",
        ],
        mean=[10.0, 2.0, 0.0, 1.0],
        std=[2.0, 1.0, 10.0, 1.0],
    )

    y_true = np.zeros((2, 4), dtype=np.float32)
    y_pred = np.ones((2, 4), dtype=np.float32)

    metrics = compute_regression_metrics(
        y_true_normalized=y_true,
        y_pred_normalized=y_pred,
        target_scaler=scaler,
    )

    assert metrics["regression_mae_mean"] > 0
    assert metrics["regression_rmse_mean"] > 0
    assert "mean_length" in metrics["per_target"]
