import numpy as np
import pandas as pd

from seismonn.data.validation import (
    parse_shape_value,
    validate_metadata,
    validate_metadata_dataframe,
    validate_metadata_files,
)


def create_valid_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_id": "000000",
                "path": "sample_0.npy",
                "filename": "sample_0.npy",
                "split": "train",
                "crack_count": 3,
                "class_id": 0,
                "shape": "2x16x8",
                "dtype": "float32",
            },
            {
                "sample_id": "000001",
                "path": "sample_1.npy",
                "filename": "sample_1.npy",
                "split": "val",
                "crack_count": 4,
                "class_id": 1,
                "shape": "2x16x8",
                "dtype": "float32",
            },
            {
                "sample_id": "000002",
                "path": "sample_2.npy",
                "filename": "sample_2.npy",
                "split": "train",
                "crack_count": 5,
                "class_id": 2,
                "shape": "2x16x8",
                "dtype": "float32",
            },
        ]
    )


def test_parse_shape_value():
    assert parse_shape_value("2x1723x501") == (2, 1723, 501)
    assert parse_shape_value("2,1723,501") == (2, 1723, 501)
    assert parse_shape_value("(2, 1723, 501)") == (2, 1723, 501)
    assert parse_shape_value([2, 1723, 501]) == (2, 1723, 501)


def test_validate_metadata_dataframe_accepts_valid_metadata():
    metadata = create_valid_metadata()

    report = validate_metadata_dataframe(
        metadata=metadata,
        expected_shape=(2, 16, 8),
        expected_dtype="float32",
        expected_splits=("train", "val"),
    )

    assert report["is_valid"] is True
    assert report["errors"] == []
    assert report["summary"]["num_rows"] == 3


def test_validate_metadata_dataframe_detects_class_mismatch():
    metadata = create_valid_metadata()
    metadata.loc[0, "class_id"] = 2

    report = validate_metadata_dataframe(
        metadata=metadata,
        expected_shape=(2, 16, 8),
        expected_dtype="float32",
        expected_splits=("train", "val"),
    )

    assert report["is_valid"] is False
    assert any("should map to" in error for error in report["errors"])


def test_validate_metadata_dataframe_detects_duplicate_paths():
    metadata = create_valid_metadata()
    metadata.loc[1, "path"] = "sample_0.npy"

    report = validate_metadata_dataframe(
        metadata=metadata,
        expected_shape=(2, 16, 8),
        expected_dtype="float32",
        expected_splits=("train", "val"),
    )

    assert report["is_valid"] is False
    assert any("duplicate" in error for error in report["errors"])


def test_validate_metadata_files_accepts_existing_files(tmp_path):
    metadata = create_valid_metadata()

    for filename in metadata["path"]:
        sample = np.random.randn(2, 16, 8).astype("float32")
        np.save(tmp_path / filename, sample)

    report = validate_metadata_files(
        metadata=metadata,
        data_root=tmp_path,
        expected_shape=(2, 16, 8),
        expected_dtype="float32",
    )

    assert report["is_valid"] is True
    assert report["summary"]["checked_files"] == 3


def test_validate_metadata_files_detects_wrong_shape(tmp_path):
    metadata = create_valid_metadata()

    for filename in metadata["path"]:
        sample = np.random.randn(2, 16, 8).astype("float32")
        np.save(tmp_path / filename, sample)

    wrong_sample = np.random.randn(2, 10, 8).astype("float32")
    np.save(tmp_path / "sample_1.npy", wrong_sample)

    report = validate_metadata_files(
        metadata=metadata,
        data_root=tmp_path,
        expected_shape=(2, 16, 8),
        expected_dtype="float32",
    )

    assert report["is_valid"] is False
    assert any("has shape" in error for error in report["errors"])


def test_validate_metadata_combines_metadata_and_file_checks(tmp_path):
    metadata = create_valid_metadata()

    for filename in metadata["path"]:
        sample = np.random.randn(2, 16, 8).astype("float32")
        np.save(tmp_path / filename, sample)

    metadata_path = tmp_path / "metadata.csv"
    metadata.to_csv(metadata_path, index=False)

    report = validate_metadata(
        metadata_path=metadata_path,
        data_root=tmp_path,
        expected_shape=(2, 16, 8),
        expected_dtype="float32",
        expected_splits=("train", "val"),
        validate_files=True,
    )

    assert report["is_valid"] is True
    assert report["metadata_validation"]["is_valid"] is True
    assert report["file_validation"]["is_valid"] is True
