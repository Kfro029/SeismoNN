from __future__ import annotations

import argparse
import json
from pathlib import Path

from seismonn.data.inspection import inspect_sample


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect one seismic sample from metadata.csv."
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
        "--index",
        type=int,
        default=0,
        help="Row index in metadata.csv. Ignored if --sample-id or --sample-path is used.",
    )
    parser.add_argument(
        "--sample-id",
        type=str,
        default=None,
        help="Sample id from metadata.csv.",
    )
    parser.add_argument(
        "--sample-path",
        type=str,
        default=None,
        help="Sample path or filename from metadata.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save inspection result as JSON.",
    )

    args = parser.parse_args()

    index = (
        None
        if args.sample_id is not None or args.sample_path is not None
        else args.index
    )

    result = inspect_sample(
        metadata_path=args.metadata,
        data_root=args.data_root,
        index=index,
        sample_id=args.sample_id,
        sample_path=args.sample_path,
    )

    result_json = json.dumps(result, indent=2, ensure_ascii=False)
    print(result_json)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)

        with args.output.open("w", encoding="utf-8") as file:
            file.write(result_json)
            file.write("\n")


if __name__ == "__main__":
    main()
