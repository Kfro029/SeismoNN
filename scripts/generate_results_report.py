from __future__ import annotations

import argparse
from pathlib import Path

from seismonn.reporting.results import generate_results_markdown, save_results_markdown


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate RESULTS.md from SeismoNN output artifacts."
    )

    parser.add_argument(
        "--evaluation-reports",
        type=Path,
        nargs="*",
        default=None,
        help="Optional list of evaluation JSON reports.",
    )
    parser.add_argument(
        "--multitask-summaries",
        type=Path,
        nargs="*",
        default=None,
        help="Optional list of multi-task prediction summary JSON files.",
    )
    parser.add_argument(
        "--benchmarks",
        type=Path,
        nargs="*",
        default=None,
        help="Optional list of inference benchmark JSON files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("RESULTS.md"),
        help="Path to save generated markdown report.",
    )

    args = parser.parse_args()

    markdown = generate_results_markdown(
        evaluation_report_paths=args.evaluation_reports,
        multitask_summary_paths=args.multitask_summaries,
        benchmark_paths=args.benchmarks,
    )

    save_results_markdown(
        markdown=markdown,
        output_path=args.output,
    )

    print(f"Saved results report to: {args.output}")


if __name__ == "__main__":
    main()