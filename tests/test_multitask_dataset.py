import numpy as np
import pandas as pd
import torch

from seismonn.data.multitask_dataset import (
    RegressionTargetScaler,
    SeismoMultiTaskDataset,
)


def test_regression_target_scaler_transforms_and_inverse_transforms():
    metadata = pd.DataFrame(
        {
            "mean_length": [10.0, 20.0, 30.0],
            "length_spread": [1.0, 2.0, 3.0],
            "mean_angle_deg": [-10.0, 0.0, 10.0],
            "angle_spread_deg": [2.0, 4.0, 6.0],
        }
    )

    columns = [
        "mean_length",
        "length_spread",
        "mean_angle_deg",
        "angle_spread_deg",
    ]

    scaler = RegressionTargetScaler.fit(metadata, columns)

    values = np.array([20.0, 2.0, 0.0, 4.0], dtype=np.float32)
    transformed = scaler.transform(values)
    restored = scaler.inverse_transform(transformed)

    assert np.allclose(restored, values)


def test_seismo_multitask_dataset_loads_sample(tmp_path):
    sample = np.random.randn(2, 16, 8).astype("float32")
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
                "mean_length": 30.0,
                "length_spread": 2.0,
                "mean_angle_deg": 14.0,
                "angle_spread_deg": 4.0,
            }
        ]
    )

    metadata_path = tmp_path / "metadata.csv"
    metadata.to_csv(metadata_path, index=False)

    dataset = SeismoMultiTaskDataset(
        metadata_path=metadata_path,
        split="train",
        data_root=tmp_path,
        normalize_input=False,
    )

    x, target = dataset[0]

    assert x.shape == torch.Size([2, 16, 8])
    assert x.dtype == torch.float32
    assert target["class_id"].item() == 1
    assert target["regression"].shape == torch.Size([4])
    assert target["regression"].dtype == torch.float32
