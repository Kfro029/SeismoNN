import torch

from seismonn.models.cnn_multitask import SeismoCNNMultiTask


def test_cnn_multitask_forward_shape():
    model = SeismoCNNMultiTask(
        in_channels=2,
        num_classes=3,
        num_regression_targets=4,
    )
    model.eval()

    x = torch.randn(2, 2, 64, 32)

    with torch.no_grad():
        outputs = model(x)

    assert set(outputs) == {"logits", "regression"}
    assert outputs["logits"].shape == torch.Size([2, 3])
    assert outputs["regression"].shape == torch.Size([2, 4])
