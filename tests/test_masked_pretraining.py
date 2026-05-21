import pytest
import torch

from seismonn.models.masked_transformer import MaskedTraceReconstructionTransformer
from seismonn.training.pretraining import (
    apply_receiver_mask,
    create_receiver_mask,
    masked_trace_mse_loss,
)


def test_create_receiver_mask_shape_and_at_least_one_masked_receiver():
    device = torch.device("cpu")

    mask = create_receiver_mask(
        batch_size=4,
        receivers=8,
        mask_ratio=0.15,
        device=device,
    )

    assert mask.shape == torch.Size([4, 8])
    assert mask.dtype == torch.bool
    assert mask.any(dim=1).all()


def test_apply_receiver_mask_zeroes_masked_receivers():
    x = torch.ones(2, 2, 4, 3)
    mask = torch.tensor(
        [
            [True, False, False],
            [False, True, False],
        ]
    )

    x_masked = apply_receiver_mask(x, mask)

    assert torch.all(x_masked[0, :, :, 0] == 0.0)
    assert torch.all(x_masked[0, :, :, 1] == 1.0)
    assert torch.all(x_masked[1, :, :, 1] == 0.0)
    assert torch.all(x_masked[1, :, :, 2] == 1.0)


def test_masked_trace_mse_loss_uses_only_masked_receivers():
    target = torch.zeros(1, 1, 2, 3)
    reconstruction = torch.zeros_like(target)

    reconstruction[:, :, :, 0] = 2.0
    reconstruction[:, :, :, 1] = 10.0

    mask = torch.tensor([[True, False, False]])

    loss = masked_trace_mse_loss(
        reconstruction=reconstruction,
        target=target,
        mask=mask,
    )

    assert loss.item() == pytest.approx(4.0)


def test_masked_trace_reconstruction_transformer_forward_shape():
    model = MaskedTraceReconstructionTransformer(
        in_channels=2,
        time_steps=32,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        dropout=0.1,
        temporal_hidden_channels=16,
        max_receivers=16,
    )
    model.eval()

    x = torch.randn(2, 2, 32, 8)
    mask = torch.zeros(2, 8, dtype=torch.bool)
    mask[:, 0] = True

    with torch.no_grad():
        reconstruction = model(x, mask=mask)

    assert reconstruction.shape == x.shape
    assert reconstruction.dtype == torch.float32


def test_masked_trace_reconstruction_transformer_rejects_wrong_time_steps():
    model = MaskedTraceReconstructionTransformer(
        in_channels=2,
        time_steps=32,
        d_model=32,
        nhead=4,
        num_layers=1,
        max_receivers=16,
    )

    x = torch.randn(2, 2, 64, 8)

    with pytest.raises(ValueError, match="Expected time_steps=32"):
        model(x)
