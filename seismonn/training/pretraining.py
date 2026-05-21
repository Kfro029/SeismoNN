from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader


def create_receiver_mask(
    batch_size: int,
    receivers: int,
    mask_ratio: float,
    device: torch.device,
) -> torch.Tensor:
    """Create boolean receiver mask.

    Shape:
        [batch_size, receivers]

    True means that receiver trace should be masked.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    if receivers <= 0:
        raise ValueError(f"receivers must be positive, got {receivers}")

    if not 0.0 < mask_ratio < 1.0:
        raise ValueError(f"mask_ratio must be in (0, 1), got {mask_ratio}")

    mask = torch.rand(batch_size, receivers, device=device) < mask_ratio

    # Ensure at least one masked receiver per sample.
    empty_rows = ~mask.any(dim=1)

    if empty_rows.any():
        random_indices = torch.randint(
            low=0,
            high=receivers,
            size=(int(empty_rows.sum().item()),),
            device=device,
        )
        mask[empty_rows, random_indices] = True

    return mask


def apply_receiver_mask(
    x: torch.Tensor,
    mask: torch.Tensor,
    fill_value: float = 0.0,
) -> torch.Tensor:
    """Mask receiver traces in input tensor.

    Args:
        x: Tensor with shape [B, C, T, R].
        mask: Boolean tensor with shape [B, R].
    """
    if x.ndim != 4:
        raise ValueError(f"Expected x shape [B, C, T, R], got {x.shape}")

    batch_size, _channels, _time_steps, receivers = x.shape

    if mask.shape != (batch_size, receivers):
        raise ValueError(
            f"Expected mask shape {(batch_size, receivers)}, got {mask.shape}"
        )

    mask = mask.to(dtype=torch.bool, device=x.device)

    x_masked = x.clone()
    x_masked = x_masked.masked_fill(mask[:, None, None, :], fill_value)

    return x_masked


def masked_trace_mse_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute MSE only on masked receiver traces."""
    if reconstruction.shape != target.shape:
        raise ValueError(
            f"reconstruction and target shapes must match. "
            f"Got {reconstruction.shape} and {target.shape}."
        )

    if reconstruction.ndim != 4:
        raise ValueError(
            f"Expected reconstruction shape [B, C, T, R], got {reconstruction.shape}"
        )

    batch_size, _channels, _time_steps, receivers = reconstruction.shape

    if mask.shape != (batch_size, receivers):
        raise ValueError(
            f"Expected mask shape {(batch_size, receivers)}, got {mask.shape}"
        )

    mask = mask.to(dtype=torch.bool, device=reconstruction.device)
    mask_4d = mask[:, None, None, :].expand_as(reconstruction)

    selected_errors = (reconstruction - target).pow(2).masked_select(mask_4d)

    if selected_errors.numel() == 0:
        raise ValueError("Mask does not contain any selected receiver traces.")

    return selected_errors.mean()


def pretrain_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mask_ratio: float,
) -> dict[str, float]:
    """Run one self-supervised pre-training epoch."""
    model.train()

    total_loss = 0.0
    total_samples = 0

    for x, _target in dataloader:
        x = x.to(device)

        mask = create_receiver_mask(
            batch_size=x.size(0),
            receivers=x.size(-1),
            mask_ratio=mask_ratio,
            device=device,
        )
        x_masked = apply_receiver_mask(x, mask)

        optimizer.zero_grad(set_to_none=True)

        reconstruction = model(x_masked, mask=mask)
        loss = masked_trace_mse_loss(
            reconstruction=reconstruction,
            target=x,
            mask=mask,
        )

        loss.backward()
        optimizer.step()

        batch_size = x.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Cannot pretrain on an empty dataloader.")

    return {
        "masked_mse": total_loss / total_samples,
    }


def evaluate_pretraining(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    mask_ratio: float,
) -> dict[str, float]:
    """Evaluate masked trace reconstruction."""
    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for x, _target in dataloader:
            x = x.to(device)

            mask = create_receiver_mask(
                batch_size=x.size(0),
                receivers=x.size(-1),
                mask_ratio=mask_ratio,
                device=device,
            )
            x_masked = apply_receiver_mask(x, mask)

            reconstruction = model(x_masked, mask=mask)
            loss = masked_trace_mse_loss(
                reconstruction=reconstruction,
                target=x,
                mask=mask,
            )

            batch_size = x.size(0)
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Cannot evaluate on an empty dataloader.")

    return {
        "masked_mse": total_loss / total_samples,
    }


def get_pretraining_metric(
    metric_name: str,
    train_metrics: dict[str, Any],
    val_metrics: dict[str, Any],
) -> float:
    """Get metric for checkpoint selection.

    For pretraining lower metric is better.
    """
    if metric_name == "val_masked_mse":
        return float(val_metrics["masked_mse"])

    if metric_name == "train_masked_mse":
        return float(train_metrics["masked_mse"])

    raise ValueError(f"Unsupported pretraining metric: {metric_name}")
