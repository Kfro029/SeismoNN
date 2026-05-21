from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


METRIC_COLUMNS = [
    "loss",
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
]


DEFAULT_DISPLAY_COLUMNS = [
    "model_label",
    "model_name",
    "split",
    "num_samples",
    "loss",
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "checkpoint_path",
]


def load_evaluation_report(report_path: str | Path) -> dict[str, Any]:
    """Load evaluation report from JSON file."""
    report_path = Path(report_path)

    if not report_path.exists():
        raise FileNotFoundError(f"Evaluation report does not exist: {report_path}")

    with report_path.open("r", encoding="utf-8") as file:
        report = json.load(file)

    if not isinstance(report, dict):
        raise ValueError(f"Expected JSON object in {report_path}, got {type(report)}")

    return report


def infer_model_label(
    report: dict[str, Any],
    report_path: str | Path,
) -> str:
    """Infer human-readable model label from report."""
    checkpoint = report.get("checkpoint", {})

    checkpoint_path = checkpoint.get("path")

    if checkpoint_path:
        checkpoint_parent = Path(str(checkpoint_path)).parent.name

        if checkpoint_parent:
            return checkpoint_parent

    model_name = checkpoint.get("model_name")

    if model_name:
        return str(model_name)

    return Path(report_path).stem


def extract_comparison_row(
    report: dict[str, Any],
    report_path: str | Path,
) -> dict[str, Any]:
    """Extract one comparison table row from evaluation report."""
    checkpoint = report.get("checkpoint", {})
    dataset = report.get("dataset", {})
    metrics = report.get("metrics", {})

    if not isinstance(checkpoint, dict):
        raise ValueError("Report field 'checkpoint' must be a dictionary.")

    if not isinstance(dataset, dict):
        raise ValueError("Report field 'dataset' must be a dictionary.")

    if not isinstance(metrics, dict):
        raise ValueError("Report field 'metrics' must be a dictionary.")

    row: dict[str, Any] = {
        "model_label": infer_model_label(report, report_path),
        "model_name": checkpoint.get("model_name"),
        "checkpoint_path": checkpoint.get("path"),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_best_metric": checkpoint.get("best_metric"),
        "split": dataset.get("split"),
        "num_samples": dataset.get("num_samples"),
        "report_path": str(report_path),
    }

    for metric_name in METRIC_COLUMNS:
        row[metric_name] = metrics.get(metric_name)

    return row


def compare_evaluation_reports(
    report_paths: list[str | Path],
    sort_by: str = "macro_f1",
    ascending: bool = False,
) -> pd.DataFrame:
    """Create comparison dataframe from multiple evaluation JSON reports."""
    if not report_paths:
        raise ValueError("report_paths must contain at least one path.")

    rows = []

    for report_path in report_paths:
        report = load_evaluation_report(report_path)
        row = extract_comparison_row(report, report_path)
        rows.append(row)

    comparison = pd.DataFrame(rows)

    if sort_by:
        if sort_by not in comparison.columns:
            raise ValueError(
                f"Cannot sort by {sort_by!r}. "
                f"Available columns: {comparison.columns.tolist()}"
            )

        comparison = comparison.sort_values(
            by=sort_by,
            ascending=ascending,
            na_position="last",
        )

    comparison = comparison.reset_index(drop=True)

    return comparison


def _format_markdown_value(value: Any, float_precision: int) -> str:
    """Format value for markdown table."""
    if value is None:
        return ""

    if pd.isna(value):
        return ""

    if isinstance(value, float):
        return f"{value:.{float_precision}f}"

    return str(value)


def format_markdown_table(
    comparison: pd.DataFrame,
    columns: list[str] | None = None,
    float_precision: int = 4,
) -> str:
    """Format comparison dataframe as markdown table without extra dependencies."""
    if columns is None:
        columns = [
            column for column in DEFAULT_DISPLAY_COLUMNS if column in comparison.columns
        ]

    if not columns:
        raise ValueError("No columns selected for markdown table.")

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"

    rows = [header, separator]

    for _, row in comparison[columns].iterrows():
        values = [
            _format_markdown_value(row[column], float_precision=float_precision)
            for column in columns
        ]
        rows.append("| " + " | ".join(values) + " |")

    return "\n".join(rows)


def save_comparison_csv(
    comparison: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Save comparison dataframe to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path, index=False)


def save_comparison_markdown(
    comparison: pd.DataFrame,
    output_path: str | Path,
    columns: list[str] | None = None,
    float_precision: int = 4,
) -> None:
    """Save comparison dataframe to markdown table."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    markdown = format_markdown_table(
        comparison=comparison,
        columns=columns,
        float_precision=float_precision,
    )

    with output_path.open("w", encoding="utf-8") as file:
        file.write(markdown)
        file.write("\n")
