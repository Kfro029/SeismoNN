from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import fire

from seismonn.data.validation import validate_metadata
from seismonn.training.utils import to_jsonable


def parse_shape(expected_shape: str | tuple[int, ...] | list[int]) -> tuple[int, ...]:
    """Parse expected shape from Fire CLI value."""
    if isinstance(expected_shape, (tuple, list)):
        return tuple(int(value) for value in expected_shape)

    normalized = str(expected_shape).strip().strip("()[]")
    normalized = normalized.replace("x", ",").replace(" ", "")
    parts = [part for part in normalized.split(",") if part]

    return tuple(int(part) for part in parts)


def parse_splits(expected_splits: str | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Parse split names from Fire CLI value."""
    if isinstance(expected_splits, (tuple, list)):
        return tuple(str(value) for value in expected_splits)

    return tuple(
        split.strip() for split in str(expected_splits).split(",") if split.strip()
    )


def save_json(data: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(to_jsonable(data), file, indent=2, ensure_ascii=False)
        file.write("\n")


def validate(
    metadata: str = "data/metadata.csv",
    data_root: str = ".",
    expected_shape: str = "2,1723,501",
    expected_dtype: str = "float32",
    expected_splits: str = "train,val",
    validate_files: bool = False,
    output: str | None = None,
) -> None:
    """Validate SeismoNN metadata.csv and optionally .npy files."""
    report = validate_metadata(
        metadata_path=metadata,
        data_root=data_root,
        expected_shape=parse_shape(expected_shape),
        expected_dtype=expected_dtype,
        expected_splits=parse_splits(expected_splits),
        validate_files=validate_files,
    )

    print(json.dumps(to_jsonable(report), indent=2, ensure_ascii=False))

    if output is not None:
        save_json(report, output)

    if not report["is_valid"]:
        sys.exit(1)


def main() -> None:
    fire.Fire(validate)


if __name__ == "__main__":
    main()
