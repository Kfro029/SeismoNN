from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from seismonn.data.validation import validate_metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate SeismoNN metadata.csv and optionally .npy files."
    )

    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/metadata.csv"),
        help="Path to metadata.csv.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("."),
        help="Root directory for relative sample paths.",
    )
    parser.add_argument(
        "--expected-shape",
        type=int,
        nargs="+",
        default=[2, 1723, 501],
        help="Expected .npy shape, for example: --expected-shape 2 1723 501.",
    )
    parser.add_argument(
        "--expected-dtype",
        type=str,
        default="float32",
        help="Expected dtype.",
    )
    parser.add_argument(
        "--expected-splits",
        type=str,
        nargs="+",
        default=["train", "val"],
        help="Expected split names.",
    )
    parser.add_argument(
        "--validate-files",
        action="store_true",
        help="Also check that .npy files exist and have expected shape/dtype.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save validation report as JSON.",
    )

    args = parser.parse_args()

    report = validate_metadata(
        metadata_path=args.metadata,
        data_root=args.data_root,
        expected_shape=tuple(args.expected_shape),
        expected_dtype=args.expected_dtype,
        expected_splits=tuple(args.expected_splits),
        validate_files=args.validate_files,
    )

    report_json = json.dumps(report, indent=2, ensure_ascii=False)
    print(report_json)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)

        with args.output.open("w", encoding="utf-8") as file:
            file.write(report_json)
            file.write("\n")

    if not report["is_valid"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
