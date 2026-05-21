from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FRACTURE_PARAMETER_COLUMNS = [
    "cluster_center_x",
    "cluster_center_y",
    "cluster_half_size_x",
    "cluster_half_size_y",
    "mean_length",
    "length_spread",
    "mean_angle_deg",
    "angle_spread_deg",
]


def _to_python_scalar(value: Any) -> Any:
    """Convert pandas/numpy scalar values to JSON-friendly Python values."""
    if pd.isna(value):
        return None

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    return value


def _format_sample_id(value: Any) -> str:
    """Format sample_id in a stable way.

    metadata.csv may read "000000" as integer 0, so we restore zero-padding
    for integer-like sample ids.
    """
    value = _to_python_scalar(value)

    if isinstance(value, int):
        return f"{value:06d}"

    if isinstance(value, float) and value.is_integer():
        return f"{int(value):06d}"

    value_str = str(value)

    if value_str.isdigit():
        return value_str.zfill(6)

    return value_str


def select_metadata_row(
    metadata: pd.DataFrame,
    index: int | None = 0,
    sample_id: str | None = None,
    sample_path: str | None = None,
) -> pd.Series:
    """Select one row from metadata by index, sample_id or path."""
    if sample_id is not None and sample_path is not None:
        raise ValueError("Use only one selector: sample_id or sample_path.")

    if sample_id is not None:
        if "sample_id" not in metadata.columns:
            raise ValueError("metadata.csv does not contain column 'sample_id'.")

        requested_sample_id = str(sample_id).zfill(6)
        normalized_ids = metadata["sample_id"].map(_format_sample_id)

        matched = metadata[normalized_ids == requested_sample_id]

        if len(matched) == 0:
            raise ValueError(f"sample_id={sample_id!r} was not found in metadata.")

        if len(matched) > 1:
            raise ValueError(f"sample_id={sample_id!r} is not unique in metadata.")

        return matched.iloc[0]

    if sample_path is not None:
        if "path" not in metadata.columns:
            raise ValueError("metadata.csv does not contain column 'path'.")

        requested_path = str(sample_path)
        requested_name = Path(requested_path).name

        path_values = metadata["path"].astype(str)
        filename_values = path_values.map(lambda value: Path(value).name)

        matched = metadata[
            (path_values == requested_path) | (filename_values == requested_name)
        ]

        if len(matched) == 0:
            raise ValueError(f"sample_path={sample_path!r} was not found in metadata.")

        if len(matched) > 1:
            raise ValueError(f"sample_path={sample_path!r} is not unique in metadata.")

        return matched.iloc[0]

    if index is None:
        index = 0

    if index < 0 or index >= len(metadata):
        raise IndexError(f"index={index} is out of range for metadata with {len(metadata)} rows.")

    return metadata.iloc[index]


def inspect_sample(
    metadata_path: str | Path,
    data_root: str | Path = ".",
    index: int | None = 0,
    sample_id: str | None = None,
    sample_path: str | None = None,
) -> dict[str, Any]:
    """Inspect one seismic sample from metadata.

    Returns JSON-friendly information:
    - metadata fields
    - target
    - fracture parameters
    - array shape/dtype/statistics
    """
    metadata_path = Path(metadata_path)
    data_root = Path(data_root)

    metadata = pd.read_csv(metadata_path)

    if len(metadata) == 0:
        raise ValueError(f"Metadata is empty: {metadata_path}")

    row = select_metadata_row(
        metadata=metadata,
        index=index,
        sample_id=sample_id,
        sample_path=sample_path,
    )

    if "path" not in row:
        raise ValueError("metadata.csv must contain column 'path'.")

    relative_path = Path(str(row["path"]))
    full_path = relative_path if relative_path.is_absolute() else data_root / relative_path

    if not full_path.exists():
        raise FileNotFoundError(f"Sample file does not exist: {full_path}")

    array = np.load(full_path, mmap_mode="r")

    fracture_parameters = {
        column: _to_python_scalar(row[column])
        for column in FRACTURE_PARAMETER_COLUMNS
        if column in row.index
    }

    target = {
        "class_id": int(row["class_id"]) if "class_id" in row.index else None,
        "crack_count": int(row["crack_count"]) if "crack_count" in row.index else None,
    }

    array_info = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "num_elements": int(array.size),
        "memory_mib": float(array.size * array.dtype.itemsize / (1024**2)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
    }

    result = {
        "sample_id": _format_sample_id(row["sample_id"]) if "sample_id" in row.index else None,
        "path": str(relative_path),
        "absolute_path": str(full_path.resolve()),
        "filename": str(row["filename"]) if "filename" in row.index else relative_path.name,
        "split": str(row["split"]) if "split" in row.index else None,
        "target": target,
        "fracture_parameters": fracture_parameters,
        "array": array_info,
    }

    return result