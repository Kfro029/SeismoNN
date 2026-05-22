from __future__ import annotations

import argparse
import json
from pathlib import Path

from seismonn.evaluation.multitask_predictions import (
    collect_multitask_predictions,
    save_predictions_csv,
    save_regression_parity_plots,
    save_summary_json,
    summarize_multitask_prediction_table,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export per-sample predictions for a multi-task checkpoint."
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to multi-task checkpoint.",
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
        help="Metadata split to export predictions for.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference.",
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
        help="Disable sample normalization. By default checkpoint data_config is used.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Path to save per-sample predictions CSV.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Optional path to save summary JSON.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Optional directory to save true-vs-predicted parity plots.",
    )

    args = parser.parse_args()

    normalize = False if args.no_normalize else None

    predictions = collect_multitask_predictions(
        checkpoint_path=args.checkpoint,
        metadata_path=args.metadata,
        split=args.split,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        normalize=normalize,
        device_name=args.device,
    )

    regression_columns = [
        column.removeprefix("true_")
        for column in predictions.columns
        if column.startswith("true_")
        and column
        not in {
            "true_class_id",
            "true_crack_count",
        }
    ]

    summary = summarize_multitask_prediction_table(
        predictions=predictions,
        regression_columns=regression_columns,
    )

    save_predictions_csv(
        predictions=predictions,
        output_path=args.output_csv,
    )

    if args.summary_output is not None:
        save_summary_json(
            summary=summary,
            output_path=args.summary_output,
        )

    if args.plots_dir is not None:
        plot_paths = save_regression_parity_plots(
            predictions=predictions,
            regression_columns=regression_columns,
            output_dir=args.plots_dir,
        )
        summary["plots"] = plot_paths

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
