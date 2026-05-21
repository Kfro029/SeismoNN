from __future__ import annotations

import argparse
import json
from pathlib import Path

from seismonn.data.visualization import visualize_sample


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create visualizations for one SeismoNN sample."
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
        "--output-dir",
        type=Path,
        default=Path("outputs/sample_visualization"),
        help="Directory for generated plots.",
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
        "--receiver-index",
        type=int,
        default=None,
        help="Receiver index for trace plot. Defaults to the middle receiver.",
    )
    parser.add_argument(
        "--max-time-steps",
        type=int,
        default=None,
        help="Optional crop for plotting only.",
    )
    parser.add_argument(
        "--max-receivers",
        type=int,
        default=None,
        help="Optional crop for plotting only.",
    )

    args = parser.parse_args()

    index = (
        None
        if args.sample_id is not None or args.sample_path is not None
        else args.index
    )

    result = visualize_sample(
        metadata_path=args.metadata,
        data_root=args.data_root,
        output_dir=args.output_dir,
        index=index,
        sample_id=args.sample_id,
        sample_path=args.sample_path,
        receiver_index=args.receiver_index,
        max_time_steps=args.max_time_steps,
        max_receivers=args.max_receivers,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
