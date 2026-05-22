import json

from seismonn.reporting.results import generate_results_markdown


def test_generate_results_markdown_with_artifacts(tmp_path):
    evaluation_report = {
        "checkpoint": {
            "path": "outputs/cnn_baseline/best.pt",
            "model_name": "cnn_baseline",
        },
        "dataset": {
            "split": "val",
            "num_samples": 10,
        },
        "metrics": {
            "loss": 1.0,
            "accuracy": 0.5,
            "balanced_accuracy": 0.5,
            "macro_precision": 0.5,
            "macro_recall": 0.5,
            "macro_f1": 0.5,
        },
    }

    evaluation_path = tmp_path / "evaluation_val.json"
    evaluation_path.write_text(
        json.dumps(evaluation_report),
        encoding="utf-8",
    )

    multitask_summary = {
        "num_samples": 10,
        "classification": {
            "accuracy": 0.5,
            "macro_f1": 0.5,
        },
        "regression": {
            "mae_mean": 1.0,
            "rmse_mean": 2.0,
            "per_target": {
                "mean_length": {
                    "mae": 1.0,
                    "rmse": 2.0,
                    "max_abs_error": 3.0,
                }
            },
        },
    }

    multitask_summary_path = tmp_path / "predictions_summary_val.json"
    multitask_summary_path.write_text(
        json.dumps(multitask_summary),
        encoding="utf-8",
    )

    benchmark = {
        "device": "cpu",
        "model_only": {
            "p50_ms": 1.0,
            "p95_ms": 2.0,
            "throughput_samples_per_sec": 100.0,
        },
        "end_to_end": {
            "p50_ms": 3.0,
            "p95_ms": 4.0,
            "throughput_samples_per_sec": 50.0,
        },
    }

    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text(
        json.dumps(benchmark),
        encoding="utf-8",
    )

    markdown = generate_results_markdown(
        evaluation_report_paths=[evaluation_path],
        multitask_summary_paths=[multitask_summary_path],
        benchmark_paths=[benchmark_path],
    )

    assert "# RESULTS.md" in markdown
    assert "cnn_baseline" in markdown
    assert "Multi-task regression analysis" in markdown
    assert "mean_length" in markdown
    assert "Inference benchmark" in markdown


def test_generate_results_markdown_without_artifacts():
    markdown = generate_results_markdown(
        evaluation_report_paths=[],
        multitask_summary_paths=[],
        benchmark_paths=[],
    )

    assert "Evaluation JSON отчёты пока не найдены" in markdown
    assert "Benchmark reports пока не найдены" in markdown