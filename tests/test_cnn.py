import torch

from seismonn.models.cnn import SeismoCNN


def test_seismo_cnn_forward_shape():
    model = SeismoCNN(in_channels=2, num_classes=3)
    model.eval()

    x = torch.randn(4, 2, 64, 32)

    with torch.no_grad():
        logits = model(x)

    assert logits.shape == torch.Size([4, 3])
    assert logits.dtype == torch.float32


def test_seismo_cnn_accepts_real_input_shape():
    model = SeismoCNN(in_channels=2, num_classes=3)
    model.eval()

    x = torch.randn(1, 2, 1723, 501)

    with torch.no_grad():
        logits = model(x)

    assert logits.shape == torch.Size([1, 3])
