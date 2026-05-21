from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "sample_id",
    "path",
    "filename",
    "split",
    "crack_count",
    "class_id",
]

DEFAULT_CLASS_ID_TO_CRACK_COUNT = {
    0: 3,
    1: 4,
    2: 5,
}


def parse_shape_value(value: Any) -> tuple[int, ...]:
    """Parse shape value from metadata.

    Supports values like:
    - "2x1723x501"
    - "2,1723,501"
    - "(2, 1723, 501)"
    - [2, 1723, 501]
    """
    if isinstance(value, (list, tuple)):
        return tuple(int(item) for item in value)

    value_str = str(value).strip()
    value_str = value_str.strip("()[]")
    value_str = value_str.replace(" ", "")

    if "x" in value_str:
        parts = value_str.split("x")
    else:
        parts = value_str.split(",")

    if not parts or any(part == "" for part in parts):
        raise ValueError(f"Cannot parse shape value: {value!r}")

    return tuple(int(part) for part in parts)


def _value_counts_to_dict(series: pd.Series) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in series.value_counts().sort_index().to_dict().items()
    }


def _group_counts_to_dict(
    metadata: pd.DataFrame,
    columns: list[str],
) -> dict[str, int]:
    counts = metadata.groupby(columns).size().sort_index().to_dict()

    return {
        ":".join(str(part) for part in key): int(value) for key, value in counts.items()
    }


def build_metadata_summary(metadata: pd.DataFrame) -> dict[str, Any]:
    """Build compact metadata summary."""
    summary: dict[str, Any] = {
        "num_rows": int(len(metadata)),
        "columns": list(metadata.columns),
    }

    if "split" in metadata.columns:
        summary["split_counts"] = _value_counts_to_dict(metadata["split"])

    if "crack_count" in metadata.columns:
        summary["crack_count_counts"] = _value_counts_to_dict(metadata["crack_count"])

    if {"split", "crack_count"}.issubset(metadata.columns):
        summary["split_crack_count_counts"] = _group_counts_to_dict(
            metadata=metadata,
            columns=["split", "crack_count"],
        )

    if "shape" in metadata.columns:
        summary["shape_counts"] = _value_counts_to_dict(metadata["shape"])

    if "dtype" in metadata.columns:
        summary["dtype_counts"] = _value_counts_to_dict(metadata["dtype"])

    return summary


def validate_metadata_dataframe(
    metadata: pd.DataFrame,
    expected_shape: tuple[int, ...] | None = None,
    expected_dtype: str | None = "float32",
    expected_splits: tuple[str, ...] = ("train", "val"),
    class_id_to_crack_count: dict[int, int] | None = None,
) -> dict[str, Any]:
    """Validate metadata.csv without reading .npy files."""
    errors: list[str] = []
    warnings: list[str] = []

    if class_id_to_crack_count is None:
        class_id_to_crack_count = DEFAULT_CLASS_ID_TO_CRACK_COUNT

    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in metadata.columns
    ]

    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")

    if len(metadata) == 0:
        errors.append("Metadata is empty.")

    for column in ["sample_id", "path", "filename"]:
        if column in metadata.columns:
            duplicate_count = int(metadata[column].duplicated().sum())

            if duplicate_count > 0:
                errors.append(
                    f"Column {column!r} contains {duplicate_count} duplicate values."
                )

    for column in REQUIRED_COLUMNS:
        if column in metadata.columns:
            missing_count = int(metadata[column].isna().sum())

            if missing_count > 0:
                errors.append(
                    f"Column {column!r} contains {missing_count} missing values."
                )

    if "split" in metadata.columns:
        actual_splits = set(metadata["split"].astype(str))
        expected_split_set = set(expected_splits)

        unexpected_splits = sorted(actual_splits - expected_split_set)
        missing_splits = sorted(expected_split_set - actual_splits)

        if unexpected_splits:
            errors.append(f"Unexpected split values: {unexpected_splits}")

        if missing_splits:
            errors.append(f"Expected splits are missing: {missing_splits}")

    if "class_id" in metadata.columns:
        expected_class_ids = set(class_id_to_crack_count)
        actual_class_ids = set(metadata["class_id"].astype(int))

        unexpected_class_ids = sorted(actual_class_ids - expected_class_ids)
        missing_class_ids = sorted(expected_class_ids - actual_class_ids)

        if unexpected_class_ids:
            errors.append(f"Unexpected class_id values: {unexpected_class_ids}")

        if missing_class_ids:
            warnings.append(
                f"Some expected class_id values are absent: {missing_class_ids}"
            )

    if "crack_count" in metadata.columns:
        expected_crack_counts = set(class_id_to_crack_count.values())
        actual_crack_counts = set(metadata["crack_count"].astype(int))

        unexpected_crack_counts = sorted(actual_crack_counts - expected_crack_counts)
        missing_crack_counts = sorted(expected_crack_counts - actual_crack_counts)

        if unexpected_crack_counts:
            errors.append(f"Unexpected crack_count values: {unexpected_crack_counts}")

        if missing_crack_counts:
            warnings.append(
                f"Some expected crack_count values are absent: {missing_crack_counts}"
            )

    if {"class_id", "crack_count"}.issubset(metadata.columns):
        for row_index, row in metadata.iterrows():
            class_id = int(row["class_id"])
            crack_count = int(row["crack_count"])

            expected_crack_count = class_id_to_crack_count.get(class_id)

            if expected_crack_count is None:
                continue

            if crack_count != expected_crack_count:
                errors.append(
                    f"Row {row_index}: class_id={class_id} should map to "
                    f"crack_count={expected_crack_count}, got {crack_count}."
                )

    if expected_shape is not None and "shape" in metadata.columns:
        for row_index, shape_value in metadata["shape"].items():
            try:
                actual_shape = parse_shape_value(shape_value)
            except ValueError as exc:
                errors.append(f"Row {row_index}: {exc}")
                continue

            if actual_shape != expected_shape:
                errors.append(
                    f"Row {row_index}: expected shape={expected_shape}, "
                    f"got shape={actual_shape}."
                )

    if expected_dtype is not None and "dtype" in metadata.columns:
        actual_dtypes = set(metadata["dtype"].astype(str))
        unexpected_dtypes = sorted(actual_dtypes - {expected_dtype})

        if unexpected_dtypes:
            errors.append(
                f"Unexpected dtype values: {unexpected_dtypes}. "
                f"Expected only {expected_dtype!r}."
            )

    summary = build_metadata_summary(metadata)

    return {
        "is_valid": len(errors) == 0,
        "summary": summary,
        "errors": errors,
        "warnings": warnings,
    }


def validate_metadata_files(
    metadata: pd.DataFrame,
    data_root: str | Path = ".",
    expected_shape: tuple[int, ...] | None = None,
    expected_dtype: str | None = "float32",
) -> dict[str, Any]:
    """Validate that files from metadata exist and have expected shape/dtype."""
    data_root = Path(data_root)

    errors: list[str] = []
    warnings: list[str] = []

    if "path" not in metadata.columns:
        return {
            "is_valid": False,
            "summary": {
                "checked_files": 0,
            },
            "errors": ["Metadata does not contain required column 'path'."],
            "warnings": warnings,
        }

    checked_files = 0

    for row_index, row in metadata.iterrows():
        relative_path = Path(str(row["path"]))
        full_path = (
            relative_path if relative_path.is_absolute() else data_root / relative_path
        )

        if not full_path.exists():
            errors.append(f"Row {row_index}: file does not exist: {full_path}")
            continue

        try:
            array = np.load(full_path, mmap_mode="r")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Row {row_index}: failed to load {full_path}: {exc}")
            continue

        checked_files += 1

        if expected_shape is not None and tuple(array.shape) != expected_shape:
            errors.append(
                f"Row {row_index}: file {full_path} has shape={tuple(array.shape)}, "
                f"expected {expected_shape}."
            )

        if expected_dtype is not None and str(array.dtype) != expected_dtype:
            errors.append(
                f"Row {row_index}: file {full_path} has dtype={array.dtype}, "
                f"expected {expected_dtype}."
            )

    return {
        "is_valid": len(errors) == 0,
        "summary": {
            "checked_files": checked_files,
            "total_rows": int(len(metadata)),
        },
        "errors": errors,
        "warnings": warnings,
    }


def validate_metadata(
    metadata_path: str | Path,
    data_root: str | Path = ".",
    expected_shape: tuple[int, ...] | None = None,
    expected_dtype: str | None = "float32",
    expected_splits: tuple[str, ...] = ("train", "val"),
    validate_files: bool = False,
) -> dict[str, Any]:
    """Validate metadata.csv and optionally referenced .npy files."""
    metadata_path = Path(metadata_path)

    if not metadata_path.exists():
        return {
            "is_valid": False,
            "metadata_path": str(metadata_path),
            "errors": [f"Metadata file does not exist: {metadata_path}"],
            "warnings": [],
            "metadata_validation": None,
            "file_validation": None,
        }

    metadata = pd.read_csv(metadata_path)

    metadata_validation = validate_metadata_dataframe(
        metadata=metadata,
        expected_shape=expected_shape,
        expected_dtype=expected_dtype,
        expected_splits=expected_splits,
    )

    file_validation = None

    if validate_files:
        file_validation = validate_metadata_files(
            metadata=metadata,
            data_root=data_root,
            expected_shape=expected_shape,
            expected_dtype=expected_dtype,
        )

    errors = list(metadata_validation["errors"])
    warnings = list(metadata_validation["warnings"])

    if file_validation is not None:
        errors.extend(file_validation["errors"])
        warnings.extend(file_validation["warnings"])

    return {
        "is_valid": len(errors) == 0,
        "metadata_path": str(metadata_path),
        "data_root": str(data_root),
        "expected_shape": list(expected_shape) if expected_shape is not None else None,
        "expected_dtype": expected_dtype,
        "expected_splits": list(expected_splits),
        "metadata_validation": metadata_validation,
        "file_validation": file_validation,
        "errors": errors,
        "warnings": warnings,
    }
