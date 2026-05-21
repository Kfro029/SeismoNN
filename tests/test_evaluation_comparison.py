import json

import pandas as pd

from seismonn.evaluation.comparison import (
    compare_evaluation_reports,
    extract_comparison_row,
    format_markdown_table,
    save_comparison_csv,
    save_comparison_markdown,
)


def create_report(
    model_name: str,
    checkpoint_path: str,
    accuracy: float,
    macro_f1: float,
) -> dict:
    return {
        "checkpoint": {
            "path": checkpoint_path,
            "model_name": model_name,
            "epoch": 3,
            "best_metric": macro_f1,
        },
        "dataset": {
            "split": "val",
            "num_samples": 133,
        },
        "metrics": {
            "loss": 1.0,
            "accuracy": accuracy,
            "balanced_accuracy": accuracy,
            "macro_precision": macro_f1,
            "macro_recall": macro_f1,
            "macro_f1": macro_f1,
            "confusion_matrix": [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
        },
    }


def write_report(path, report: dict) -> None:
    path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )


def test_extract_comparison_row():
    report = create_report(
        model_name="cnn_baseline",
        checkpoint_path="outputs/cnn_baseline/best.pt",
        accuracy=0.5,
        macro_f1=0.4,
    )

    row = extract_comparison_row(
        report=report,
        report_path="outputs/cnn_baseline/evaluation_val.json",
    )

    assert row["model_label"] == "cnn_baseline"
    assert row["model_name"] == "cnn_baseline"
    assert row["checkpoint_path"] == "outputs/cnn_baseline/best.pt"
    assert row["split"] == "val"
    assert row["num_samples"] == 133
    assert row["accuracy"] == 0.5
    assert row["macro_f1"] == 0.4


def test_compare_evaluation_reports_sorts_by_macro_f1(tmp_path):
    first_path = tmp_path / "cnn.json"
    second_path = tmp_path / "transformer.json"

    write_report(
        first_path,
        create_report(
            model_name="cnn_baseline",
            checkpoint_path="outputs/cnn_baseline/best.pt",
            accuracy=0.4,
            macro_f1=0.3,
        ),
    )

    write_report(
        second_path,
        create_report(
            model_name="trace_transformer",
            checkpoint_path="outputs/trace_transformer/best.pt",
            accuracy=0.6,
            macro_f1=0.5,
        ),
    )

    comparison = compare_evaluation_reports(
        report_paths=[first_path, second_path],
        sort_by="macro_f1",
        ascending=False,
    )

    assert comparison.iloc[0]["model_label"] == "trace_transformer"
    assert comparison.iloc[1]["model_label"] == "cnn_baseline"


def test_format_markdown_table():
    comparison = pd.DataFrame(
        [
            {
                "model_label": "cnn_baseline",
                "accuracy": 0.456789,
                "macro_f1": 0.345678,
            }
        ]
    )

    markdown = format_markdown_table(
        comparison=comparison,
        columns=["model_label", "accuracy", "macro_f1"],
        float_precision=3,
    )

    assert "| model_label | accuracy | macro_f1 |" in markdown
    assert "| cnn_baseline | 0.457 | 0.346 |" in markdown


def test_save_comparison_outputs(tmp_path):
    comparison = pd.DataFrame(
        [
            {
                "model_label": "cnn_baseline",
                "accuracy": 0.5,
                "macro_f1": 0.4,
            }
        ]
    )

    csv_path = tmp_path / "comparison.csv"
    md_path = tmp_path / "comparison.md"

    save_comparison_csv(comparison, csv_path)
    save_comparison_markdown(
        comparison=comparison,
        output_path=md_path,
        columns=["model_label", "accuracy", "macro_f1"],
    )

    assert csv_path.exists()
    assert md_path.exists()

    assert "cnn_baseline" in csv_path.read_text(encoding="utf-8")
    assert "cnn_baseline" in md_path.read_text(encoding="utf-8")
