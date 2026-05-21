import numpy as np
import pandas as pd

from seismonn.data.inspection import inspect_sample, select_metadata_row


def test_select_metadata_row_by_index():
    metadata = pd.DataFrame(
        [
            {
                "sample_id": "000000",
                "path": "sample_0.npy",
                "crack_count": 3,
                "class_id": 0,
            },
            {
                "sample_id": "000001",
                "path": "sample_1.npy",
                "crack_count": 4,
                "class_id": 1,
            },
        ]
    )

    row = select_metadata_row(metadata, index=1)

    assert row["sample_id"] == "000001"
    assert row["crack_count"] == 4


def test_select_metadata_row_by_sample_id_with_zero_padding():
    metadata = pd.DataFrame(
        [
            {
                "sample_id": 0,
                "path": "sample_0.npy",
                "crack_count": 3,
                "class_id": 0,
            }
        ]
    )

    row = select_metadata_row(metadata, sample_id="000000")

    assert row["path"] == "sample_0.npy"


def test_inspect_sample_returns_metadata_and_array_stats(tmp_path):
    sample = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ],
        dtype=np.float32,
    )

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
                "cluster_center_x": 0.0,
                "cluster_center_y": -150.0,
                "cluster_half_size_x": 250.0,
                "cluster_half_size_y": 150.0,
                "mean_length": 30.0,
                "length_spread": 2.0,
                "mean_angle_deg": 14.0,
                "angle_spread_deg": 14.0,
            }
        ]
    )

    metadata_path = tmp_path / "metadata.csv"
    metadata.to_csv(metadata_path, index=False)

    result = inspect_sample(
        metadata_path=metadata_path,
        data_root=tmp_path,
        index=0,
    )

    assert result["sample_id"] == "000000"
    assert result["path"] == "sample.npy"
    assert result["filename"] == "sample.npy"
    assert result["split"] == "train"

    assert result["target"] == {
        "class_id": 1,
        "crack_count": 4,
    }

    assert result["fracture_parameters"]["mean_length"] == 30.0
    assert result["fracture_parameters"]["mean_angle_deg"] == 14.0

    assert result["array"]["shape"] == [2, 2, 2]
    assert result["array"]["dtype"] == "float32"
    assert result["array"]["num_elements"] == 8
    assert result["array"]["min"] == 1.0
    assert result["array"]["max"] == 8.0
    assert result["array"]["mean"] == 4.5