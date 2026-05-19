import numpy as np
import pandas as pd
import torch

from seismonn.data.dataset import SeismoDataset


def test_seismo_dataset_loads_sample(tmp_path):
    sample = np.random.randn(2, 16, 8).astype("float32")
    sample_path = tmp_path / "sample.npy"
    np.save(sample_path, sample)

    metadata = pd.DataFrame(
        [
            {
                "sample_id": "000000",
                "path": "sample.npy",
                "filename": "sample.npy",
                "split": "train",
                "crack_count": 4,
                "class_id": 1,
            }
        ]
    )

    metadata_path = tmp_path / "metadata.csv"
    metadata.to_csv(metadata_path, index=False)

    dataset = SeismoDataset(
        metadata_path=metadata_path,
        split="train",
        data_root=tmp_path,
        normalize=False,
    )

    x, y = dataset[0]

    assert len(dataset) == 1
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.shape == torch.Size([2, 16, 8])
    assert x.dtype == torch.float32
    assert y.dtype == torch.long
    assert y.item() == 1


def test_seismo_dataset_filters_by_split(tmp_path):
    sample_train = np.random.randn(2, 16, 8).astype("float32")
    sample_val = np.random.randn(2, 16, 8).astype("float32")

    np.save(tmp_path / "train_sample.npy", sample_train)
    np.save(tmp_path / "val_sample.npy", sample_val)

    metadata = pd.DataFrame(
        [
            {
                "sample_id": "000000",
                "path": "train_sample.npy",
                "filename": "train_sample.npy",
                "split": "train",
                "crack_count": 3,
                "class_id": 0,
            },
            {
                "sample_id": "000001",
                "path": "val_sample.npy",
                "filename": "val_sample.npy",
                "split": "val",
                "crack_count": 5,
                "class_id": 2,
            },
        ]
    )

    metadata_path = tmp_path / "metadata.csv"
    metadata.to_csv(metadata_path, index=False)

    train_dataset = SeismoDataset(
        metadata_path=metadata_path,
        split="train",
        data_root=tmp_path,
        normalize=False,
    )

    val_dataset = SeismoDataset(
        metadata_path=metadata_path,
        split="val",
        data_root=tmp_path,
        normalize=False,
    )

    _, train_y = train_dataset[0]
    _, val_y = val_dataset[0]

    assert len(train_dataset) == 1
    assert len(val_dataset) == 1
    assert train_y.item() == 0
    assert val_y.item() == 2


def test_seismo_dataset_normalizes_sample(tmp_path):
    sample = np.random.randn(2, 16, 8).astype("float32") * 10 + 100
    np.save(tmp_path / "sample.npy", sample)

    metadata = pd.DataFrame(
        [
            {
                "sample_id": "000000",
                "path": "sample.npy",
                "filename": "sample.npy",
                "split": "train",
                "crack_count": 4,
                "class_id": 1,
            }
        ]
    )

    metadata_path = tmp_path / "metadata.csv"
    metadata.to_csv(metadata_path, index=False)

    dataset = SeismoDataset(
        metadata_path=metadata_path,
        split="train",
        data_root=tmp_path,
        normalize=True,
    )

    x, _ = dataset[0]

    assert abs(float(x.mean())) < 1e-5
    assert abs(float(x.std()) - 1.0) < 1e-2
