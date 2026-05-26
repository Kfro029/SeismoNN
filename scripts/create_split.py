from __future__ import annotations

import json
from pathlib import Path

import fire
import pandas as pd

from seismonn.data.splits import create_stratified_split, get_split_summary


def main(
    input: str = "data/metadata.csv",
    output: str = "data/metadata_stratified.csv",
    val_size: float = 0.2,
    test_size: float = 0.0,
    seed: int = 42,
    stratify_column: str = "class_id",
    target_column: str = "crack_count",
) -> None:
    """Create reproducible stratified splits for SeismoNN metadata."""
    metadata = pd.read_csv(input)

    updated_metadata = create_stratified_split(
        metadata=metadata,
        val_size=val_size,
        test_size=test_size,
        seed=seed,
        stratify_column=stratify_column,
    )

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    updated_metadata.to_csv(output_path, index=False)

    summary = get_split_summary(
        metadata=updated_metadata,
        split_column="split",
        target_column=target_column,
    )

    print(f"Saved stratified metadata to: {output_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    fire.Fire(main)
