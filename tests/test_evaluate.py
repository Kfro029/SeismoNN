import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from seismonn.models.cnn import SeismoCNN
from seismonn.training.evaluate import evaluate_classifier


def test_evaluate_classifier_returns_metrics():
    model = SeismoCNN(in_channels=2, num_classes=3)
    model.eval()

    x = torch.randn(6, 2, 16, 8)
    y = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)

    dataloader = DataLoader(TensorDataset(x, y), batch_size=2)
    criterion = nn.CrossEntropyLoss()

    metrics = evaluate_classifier(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        device=torch.device("cpu"),
        labels=[0, 1, 2],
    )

    assert set(metrics) == {
        "loss",
        "accuracy",
        "macro_f1",
        "confusion_matrix",
        "y_true",
        "y_pred",
    }
    assert isinstance(metrics["loss"], float)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert metrics["confusion_matrix"].shape == (3, 3)
    assert metrics["y_true"].shape == (6,)
    assert metrics["y_pred"].shape == (6,)
