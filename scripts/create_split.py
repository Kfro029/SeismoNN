from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from seismonn.data.splits import create_stratified_split, get_split_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create reproducible stratified splits for SeismoNN metadata."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/metadata.csv"),
        help="Input metadata CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/metadata_stratified.csv"),
        help="Output metadata CSV with updated split column.",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.2,
        help="Validation fraction of the full dataset.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.0,
        help="Test fraction of the full dataset. Use 0.0 for train/val split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible split.",
    )
    parser.add_argument(
        "--stratify-column",
        type=str,
        default="class_id",
        help="Column used for stratification.",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="crack_count",
        help="Target column used only for summary printing.",
    )

    args = parser.parse_args()

    metadata = pd.read_csv(args.input)

    updated_metadata = create_stratified_split(
        metadata=metadata,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
        stratify_column=args.stratify_column,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    updated_metadata.to_csv(args.output, index=False)

    summary = get_split_summary(
        metadata=updated_metadata,
        split_column="split",
        target_column=args.target_column,
    )

    print(f"Saved stratified metadata to: {args.output}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
