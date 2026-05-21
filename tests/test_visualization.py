import numpy as np
import pandas as pd

from seismonn.data.visualization import (
    crop_sample_for_plot,
    load_sample_array_from_metadata,
    visualize_sample,
)


def create_metadata(tmp_path):
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

    return metadata_path, sample_path


def test_load_sample_array_from_metadata(tmp_path):
    metadata_path, sample_path = create_metadata(tmp_path)

    array, row, full_path = load_sample_array_from_metadata(
        metadata_path=metadata_path,
        data_root=tmp_path,
        index=0,
    )

    assert array.shape == (2, 16, 8)
    assert array.dtype == np.float32
    assert row["crack_count"] == 4
    assert full_path == sample_path


def test_crop_sample_for_plot():
    array = np.random.randn(2, 16, 8).astype("float32")

    cropped = crop_sample_for_plot(
        array=array,
        max_time_steps=10,
        max_receivers=4,
    )

    assert cropped.shape == (2, 10, 4)


def test_visualize_sample_creates_files(tmp_path):
    metadata_path, _sample_path = create_metadata(tmp_path)

    output_dir = tmp_path / "visualization"

    result = visualize_sample(
        metadata_path=metadata_path,
        data_root=tmp_path,
        output_dir=output_dir,
        index=0,
        receiver_index=2,
        max_time_steps=12,
        max_receivers=6,
    )

    expected_keys = {
        "output_dir",
        "sample_info",
        "vx_heatmap",
        "vy_heatmap",
        "vx_receiver_trace",
        "vy_receiver_trace",
    }

    assert set(result) == expected_keys

    for path in result.values():
        assert path

    assert (output_dir / "sample_info.json").exists()
    assert (output_dir / "vx_heatmap.png").exists()
    assert (output_dir / "vy_heatmap.png").exists()
    assert (output_dir / "vx_receiver_trace.png").exists()
    assert (output_dir / "vy_receiver_trace.png").exists()
