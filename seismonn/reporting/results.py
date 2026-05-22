from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_EVALUATION_REPORTS = [
    Path("outputs/cnn_baseline/evaluation_val.json"),
    Path("outputs/trace_transformer/evaluation_val.json"),
    Path("outputs/trace_transformer_finetuned/evaluation_val.json"),
    Path("outputs/cnn_multitask/evaluation_val.json"),
    Path("outputs/cnn_multitask_50ep/evaluation_val.json"),
]

DEFAULT_MULTITASK_SUMMARIES = [
    Path("outputs/cnn_multitask/predictions_summary_val.json"),
    Path("outputs/cnn_multitask_50ep/predictions_summary_val.json"),
]

DEFAULT_BENCHMARK_REPORTS = [
    Path("outputs/inference_benchmark.json"),
    Path("outputs/cnn_baseline/inference_benchmark.json"),
    Path("outputs/cnn_multitask_50ep/inference_benchmark.json"),
]


def load_json(path: str | Path) -> dict[str, Any]:
    """Load JSON file."""
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(data)}")

    return data


def format_float(value: Any, digits: int = 4) -> str:
    """Format float-like value for markdown."""
    if value is None:
        return "—"

    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def existing_paths(paths: list[Path]) -> list[Path]:
    """Return only existing paths."""
    return [path for path in paths if path.exists()]


def infer_model_label(report: dict[str, Any], report_path: str | Path) -> str:
    """Infer readable model label."""
    checkpoint = report.get("checkpoint", {})

    if isinstance(checkpoint, dict):
        checkpoint_path = checkpoint.get("path")

        if checkpoint_path:
            parent = Path(str(checkpoint_path)).parent.name

            if parent:
                return parent

        model_name = checkpoint.get("model_name")

        if model_name:
            return str(model_name)

    return Path(report_path).stem


def extract_evaluation_row(report_path: str | Path) -> dict[str, Any]:
    """Extract one row from evaluation JSON."""
    report = load_json(report_path)

    checkpoint = report.get("checkpoint", {})
    dataset = report.get("dataset", {})
    metrics = report.get("metrics", {})

    if not isinstance(checkpoint, dict):
        checkpoint = {}

    if not isinstance(dataset, dict):
        dataset = {}

    if not isinstance(metrics, dict):
        metrics = {}

    return {
        "model": infer_model_label(report, report_path),
        "model_name": checkpoint.get("model_name", "—"),
        "split": dataset.get("split", "—"),
        "num_samples": dataset.get("num_samples", "—"),
        "loss": metrics.get("loss"),
        "accuracy": metrics.get("accuracy"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "macro_precision": metrics.get("macro_precision"),
        "macro_recall": metrics.get("macro_recall"),
        "macro_f1": metrics.get("macro_f1"),
        "regression_mae_mean": metrics.get("regression_mae_mean"),
        "regression_rmse_mean": metrics.get("regression_rmse_mean"),
        "report_path": str(report_path),
    }


def markdown_table(
    rows: list[dict[str, Any]],
    columns: list[str],
    float_columns: set[str] | None = None,
) -> str:
    """Build markdown table."""
    if float_columns is None:
        float_columns = set()

    if not rows:
        return "_Нет доступных данных._"

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"

    lines = [header, separator]

    for row in rows:
        values = []

        for column in columns:
            value = row.get(column)

            if column in float_columns:
                values.append(format_float(value))
            else:
                values.append("—" if value is None else str(value))

        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def build_evaluation_section(report_paths: list[Path]) -> str:
    """Build model evaluation section."""
    existing_reports = existing_paths(report_paths)

    if not existing_reports:
        return "\n".join(
            [
                "## Сравнение моделей",
                "",
                "_Evaluation JSON отчёты пока не найдены._",
                "",
                "Чтобы создать отчёт, запусти, например:",
                "",
                "```bash",
                "make evaluate-cnn",
                "make evaluate-multitask",
                "make compare",
                "```",
            ]
        )

    rows = [extract_evaluation_row(path) for path in existing_reports]

    rows = sorted(
        rows,
        key=lambda row: float(row["macro_f1"])
        if row.get("macro_f1") is not None
        else -1.0,
        reverse=True,
    )

    table = markdown_table(
        rows=rows,
        columns=[
            "model",
            "model_name",
            "split",
            "num_samples",
            "loss",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "regression_mae_mean",
            "regression_rmse_mean",
        ],
        float_columns={
            "loss",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "regression_mae_mean",
            "regression_rmse_mean",
        },
    )

    report_list = "\n".join(f"- `{path}`" for path in existing_reports)

    return "\n".join(
        [
            "## Сравнение моделей",
            "",
            table,
            "",
            "Использованные evaluation reports:",
            "",
            report_list,
        ]
    )


def extract_multitask_summary(summary_path: str | Path) -> dict[str, Any]:
    """Extract multi-task prediction summary."""
    summary = load_json(summary_path)

    classification = summary.get("classification", {})
    regression = summary.get("regression", {})

    if not isinstance(classification, dict):
        classification = {}

    if not isinstance(regression, dict):
        regression = {}

    return {
        "summary_path": str(summary_path),
        "num_samples": summary.get("num_samples"),
        "classification_accuracy": classification.get("accuracy"),
        "classification_macro_f1": classification.get("macro_f1"),
        "regression_mae_mean": regression.get("mae_mean"),
        "regression_rmse_mean": regression.get("rmse_mean"),
        "per_target": regression.get("per_target", {}),
    }


def build_multitask_section(summary_paths: list[Path]) -> str:
    """Build multi-task regression analysis section."""
    existing_summaries = existing_paths(summary_paths)

    if not existing_summaries:
        return "\n".join(
            [
                "## Multi-task regression analysis",
                "",
                "_Per-sample multi-task prediction summaries пока не найдены._",
                "",
                "Чтобы создать их, запусти:",
                "",
                "```bash",
                "make export-multitask-predictions",
                "```",
            ]
        )

    summaries = [extract_multitask_summary(path) for path in existing_summaries]

    rows = [
        {
            "summary": Path(summary["summary_path"]).parent.name,
            "num_samples": summary["num_samples"],
            "classification_accuracy": summary["classification_accuracy"],
            "classification_macro_f1": summary["classification_macro_f1"],
            "regression_mae_mean": summary["regression_mae_mean"],
            "regression_rmse_mean": summary["regression_rmse_mean"],
        }
        for summary in summaries
    ]

    table = markdown_table(
        rows=rows,
        columns=[
            "summary",
            "num_samples",
            "classification_accuracy",
            "classification_macro_f1",
            "regression_mae_mean",
            "regression_rmse_mean",
        ],
        float_columns={
            "classification_accuracy",
            "classification_macro_f1",
            "regression_mae_mean",
            "regression_rmse_mean",
        },
    )

    per_target_lines = []

    for summary in summaries:
        per_target = summary.get("per_target", {})

        if not isinstance(per_target, dict) or not per_target:
            continue

        per_target_lines.append("")
        per_target_lines.append(f"### `{summary['summary_path']}`")
        per_target_lines.append("")

        target_rows = []

        for target_name, target_metrics in per_target.items():
            if not isinstance(target_metrics, dict):
                continue

            target_rows.append(
                {
                    "target": target_name,
                    "mae": target_metrics.get("mae"),
                    "rmse": target_metrics.get("rmse"),
                    "max_abs_error": target_metrics.get("max_abs_error"),
                }
            )

        per_target_lines.append(
            markdown_table(
                rows=target_rows,
                columns=["target", "mae", "rmse", "max_abs_error"],
                float_columns={"mae", "rmse", "max_abs_error"},
            )
        )

    return "\n".join(
        [
            "## Multi-task regression analysis",
            "",
            table,
            *per_target_lines,
        ]
    )


def extract_benchmark_summary(benchmark_path: str | Path) -> dict[str, Any]:
    """Extract benchmark summary."""
    benchmark = load_json(benchmark_path)

    model_only = benchmark.get("model_only", {})
    end_to_end = benchmark.get("end_to_end", {})

    if not isinstance(model_only, dict):
        model_only = {}

    if not isinstance(end_to_end, dict):
        end_to_end = {}

    return {
        "benchmark": Path(benchmark_path).parent.name,
        "device": benchmark.get("device"),
        "model_only_p50_ms": model_only.get("p50_ms"),
        "model_only_p95_ms": model_only.get("p95_ms"),
        "model_only_throughput": model_only.get("throughput_samples_per_sec"),
        "end_to_end_p50_ms": end_to_end.get("p50_ms"),
        "end_to_end_p95_ms": end_to_end.get("p95_ms"),
        "end_to_end_throughput": end_to_end.get("throughput_samples_per_sec"),
        "path": str(benchmark_path),
    }


def build_benchmark_section(benchmark_paths: list[Path]) -> str:
    """Build inference benchmark section."""
    existing_benchmarks = existing_paths(benchmark_paths)

    if not existing_benchmarks:
        return "\n".join(
            [
                "## Inference benchmark",
                "",
                "_Benchmark reports пока не найдены._",
                "",
                "Чтобы создать benchmark, запусти:",
                "",
                "```bash",
                "make benchmark-cnn",
                "```",
            ]
        )

    rows = [extract_benchmark_summary(path) for path in existing_benchmarks]

    table = markdown_table(
        rows=rows,
        columns=[
            "benchmark",
            "device",
            "model_only_p50_ms",
            "model_only_p95_ms",
            "model_only_throughput",
            "end_to_end_p50_ms",
            "end_to_end_p95_ms",
            "end_to_end_throughput",
        ],
        float_columns={
            "model_only_p50_ms",
            "model_only_p95_ms",
            "model_only_throughput",
            "end_to_end_p50_ms",
            "end_to_end_p95_ms",
            "end_to_end_throughput",
        },
    )

    return "\n".join(
        [
            "## Inference benchmark",
            "",
            table,
        ]
    )


def generate_results_markdown(
    evaluation_report_paths: list[Path] | None = None,
    multitask_summary_paths: list[Path] | None = None,
    benchmark_paths: list[Path] | None = None,
) -> str:
    """Generate RESULTS.md content."""
    if evaluation_report_paths is None:
        evaluation_report_paths = DEFAULT_EVALUATION_REPORTS

    if multitask_summary_paths is None:
        multitask_summary_paths = DEFAULT_MULTITASK_SUMMARIES

    if benchmark_paths is None:
        benchmark_paths = DEFAULT_BENCHMARK_REPORTS

    sections = [
        "# RESULTS.md: результаты экспериментов SeismoNN",
        "",
        "Этот файл содержит сводку фактических результатов обучения, оценки и инференса.",
        "",
        "Файл можно пересоздать командой:",
        "",
        "```bash",
        "uv run python scripts/generate_results_report.py --output RESULTS.md",
        "```",
        "",
        build_evaluation_section(evaluation_report_paths),
        "",
        build_multitask_section(multitask_summary_paths),
        "",
        build_benchmark_section(benchmark_paths),
        "",
        "## Выводы",
        "",
        "Основные выводы следует интерпретировать с учётом ограничений датасета:",
        "",
        "```text",
        "- данные синтетические;",
        "- объектов немного: 665;",
        "- validation split стратифицирован, но пока не является group split;",
        "- перенос на реальные field data не проверялся.",
        "```",
        "",
        "Фактические численные выводы зависят от запущенных экспериментов и доступных файлов в `outputs/`.",
    ]

    return "\n".join(sections) + "\n"


def save_results_markdown(
    markdown: str,
    output_path: str | Path,
) -> None:
    """Save RESULTS.md content."""
    output_path = Path(output_path)

    with output_path.open("w", encoding="utf-8") as file:
        file.write(markdown)