from __future__ import annotations

import argparse
import json
from pathlib import Path

from seismonn.evaluation.checkpoint import (
    evaluate_checkpoint,
    save_evaluation_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained SeismoNN checkpoint on a metadata split."
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint.",
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
        "--split",
        type=str,
        default="val",
        help="Metadata split to evaluate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto, cpu, cuda.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable sample normalization. By default, checkpoint data_config is used.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save JSON evaluation report.",
    )

    args = parser.parse_args()

    normalize = False if args.no_normalize else None

    report = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        metadata_path=args.metadata,
        split=args.split,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        normalize=normalize,
        device_name=args.device,
    )

    report_json = json.dumps(report, indent=2, ensure_ascii=False)
    print(report_json)

    if args.output is not None:
        save_evaluation_report(report, args.output)


if __name__ == "__main__":
    main()
