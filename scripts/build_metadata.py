from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import fire


PREFIXES = ("receivers_fractures_", "fractures_")
CLASS_ID = {3: 0, 4: 1, 5: 2}

PARAM_NAMES = [
    "cluster_center_x",
    "cluster_center_y",
    "cluster_half_size_x",
    "cluster_half_size_y",
    "mean_length",
    "length_spread",
    "mean_angle_deg",
    "angle_spread_deg",
]


def parse_sample_filename(filename: str) -> dict[str, Any]:
    """Parse filename like:
    receivers_fractures_4_0.0_-150.0_250.0_150.0_30.0_2.0_14.0_14.0.npy

    Format:
    fractures_{crack_count}_{cluster_center_x}_{cluster_center_y}
              _{cluster_half_size_x}_{cluster_half_size_y}
              _{mean_length}_{length_spread}
              _{mean_angle_deg}_{angle_spread_deg}
    """
    path = Path(filename)
    stem = path.stem

    matched_prefix = None
    for prefix in PREFIXES:
        if stem.startswith(prefix):
            matched_prefix = prefix
            break

    if matched_prefix is None:
        raise ValueError(f"Unexpected filename format: {filename}")

    raw_tokens = stem[len(matched_prefix) :].split("_")

    if len(raw_tokens) != 1 + len(PARAM_NAMES):
        raise ValueError(
            f"Unexpected number of numeric tokens in filename={filename}. "
            f"Expected {1 + len(PARAM_NAMES)}, got {len(raw_tokens)}."
        )

    try:
        crack_count = int(raw_tokens[0])
    except ValueError as exc:
        raise ValueError(f"Cannot parse crack_count from filename: {filename}") from exc

    if crack_count not in CLASS_ID:
        raise ValueError(
            f"Unexpected crack_count={crack_count} in {filename}. "
            f"Expected one of {sorted(CLASS_ID)}."
        )

    try:
        params = [float(token) for token in raw_tokens[1:]]
    except ValueError as exc:
        raise ValueError(
            f"Cannot parse numeric parameters from filename: {filename}"
        ) from exc

    row: dict[str, Any] = {
        "crack_count": crack_count,
        "class_id": CLASS_ID[crack_count],
    }

    row.update(dict(zip(PARAM_NAMES, params)))

    return row


def normalize_filename(raw_path: str) -> str:
    """Convert absolute/local path from JSON to a clean filename.

    Also supports old split files that may contain .txt names.
    """
    filename = Path(raw_path).name

    if filename.endswith(".txt"):
        filename = str(Path(filename).with_suffix(".npy"))

    return filename


def build_metadata(
    split_json_path: Path,
    data_dir: Path,
    output_path: Path,
    test_split_name: str,
    validate_files: bool,
) -> None:
    with split_json_path.open("r", encoding="utf-8") as file:
        split_data = json.load(file)

    rows: list[dict[str, Any]] = []
    seen_filenames: set[str] = set()

    sample_idx = 0

    for original_split, raw_paths in split_data.items():
        if original_split == "test":
            split = test_split_name
        else:
            split = original_split

        for raw_path in raw_paths:
            filename = normalize_filename(raw_path)

            if filename in seen_filenames:
                raise ValueError(f"Duplicate filename in split file: {filename}")
            seen_filenames.add(filename)

            relative_path = data_dir / filename

            parsed = parse_sample_filename(filename)

            row: dict[str, Any] = {
                "sample_id": f"{sample_idx:06d}",
                "path": relative_path.as_posix(),
                "filename": filename,
                "split": split,
                **parsed,
            }

            if validate_files:
                full_path = Path(relative_path)

                if not full_path.exists():
                    raise FileNotFoundError(
                        f"File from metadata does not exist: {full_path}"
                    )

                # mmap_mode allows us to read shape/dtype without loading the whole array into RAM.
                import numpy as np

                arr = np.load(full_path, mmap_mode="r")
                row["shape"] = "x".join(map(str, arr.shape))
                row["dtype"] = str(arr.dtype)

            rows.append(row)
            sample_idx += 1

    if not rows:
        raise ValueError(f"No rows were created from {split_json_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Keep stable column order.
    all_columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in all_columns:
                all_columns.append(key)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=all_columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved metadata to: {output_path}")
    print(f"Rows: {len(rows)}")

    split_counts: dict[str, int] = {}
    class_counts: dict[tuple[str, int], int] = {}

    for row in rows:
        split = str(row["split"])
        crack_count = int(row["crack_count"])

        split_counts[split] = split_counts.get(split, 0) + 1
        class_counts[(split, crack_count)] = (
            class_counts.get((split, crack_count), 0) + 1
        )

    print("Split counts:")
    for split, count in sorted(split_counts.items()):
        print(f"  {split}: {count}")

    print("Class counts:")
    for (split, crack_count), count in sorted(class_counts.items()):
        print(f"  {split}, crack_count={crack_count}: {count}")


def main(
    split_json: str = "2nd_sel.json",
    data_dir: str = "2nd_selection",
    output: str = "data/metadata.csv",
    test_split_name: str = "val",
    validate_files: bool = False,
) -> None:
    """Build metadata.csv from split JSON and .npy files."""
    build_metadata(
        split_json_path=Path(split_json),
        data_dir=Path(data_dir),
        output_path=Path(output),
        test_split_name=test_split_name,
        validate_files=validate_files,
    )


if __name__ == "__main__":
    fire.Fire(main)
