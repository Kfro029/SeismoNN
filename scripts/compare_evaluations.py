from __future__ import annotations

import argparse
from pathlib import Path

from seismonn.evaluation.comparison import (
    compare_evaluation_reports,
    format_markdown_table,
    save_comparison_csv,
    save_comparison_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare multiple SeismoNN evaluation JSON reports."
    )

    parser.add_argument(
        "--reports",
        type=Path,
        nargs="+",
        required=True,
        help="Paths to evaluation JSON reports.",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default="macro_f1",
        help="Column used for sorting.",
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="Sort in ascending order. By default sorting is descending.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path to save comparison CSV.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Optional path to save comparison markdown table.",
    )
    parser.add_argument(
        "--float-precision",
        type=int,
        default=4,
        help="Number of digits after decimal point in markdown table.",
    )

    args = parser.parse_args()

    comparison = compare_evaluation_reports(
        report_paths=args.reports,
        sort_by=args.sort_by,
        ascending=args.ascending,
    )

    markdown = format_markdown_table(
        comparison=comparison,
        float_precision=args.float_precision,
    )

    print(markdown)

    if args.output_csv is not None:
        save_comparison_csv(comparison, args.output_csv)

    if args.output_md is not None:
        save_comparison_markdown(
            comparison=comparison,
            output_path=args.output_md,
            float_precision=args.float_precision,
        )


if __name__ == "__main__":
    main()
