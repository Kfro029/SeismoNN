from __future__ import annotations

import json

import fire

from seismonn.evaluation.multitask_predictions import (
    collect_multitask_predictions,
    save_predictions_csv,
    save_regression_parity_plots,
    save_summary_json,
    summarize_multitask_prediction_table,
)
from seismonn.training.utils import to_jsonable


def main(
    checkpoint: str,
    output_csv: str,
    metadata: str = "data/metadata.csv",
    data_root: str = ".",
    split: str = "val",
    batch_size: int = 8,
    num_workers: int = 0,
    device: str = "auto",
    no_normalize: bool = False,
    summary_output: str | None = None,
    plots_dir: str | None = None,
) -> None:
    """Export per-sample predictions for a multi-task checkpoint."""
    normalize = False if no_normalize else None

    predictions = collect_multitask_predictions(
        checkpoint_path=checkpoint,
        metadata_path=metadata,
        split=split,
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        normalize=normalize,
        device_name=device,
    )

    regression_columns = [
        column.removeprefix("true_")
        for column in predictions.columns
        if column.startswith("true_")
        and column not in {"true_class_id", "true_crack_count"}
    ]

    summary = summarize_multitask_prediction_table(
        predictions=predictions,
        regression_columns=regression_columns,
    )

    save_predictions_csv(
        predictions=predictions,
        output_path=output_csv,
    )

    if plots_dir is not None:
        plot_paths = save_regression_parity_plots(
            predictions=predictions,
            regression_columns=regression_columns,
            output_dir=plots_dir,
        )
        summary["plots"] = plot_paths

    if summary_output is not None:
        save_summary_json(
            summary=summary,
            output_path=summary_output,
        )

    print(json.dumps(to_jsonable(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    fire.Fire(main)
