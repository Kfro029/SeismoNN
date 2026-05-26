from __future__ import annotations

from pathlib import Path

import fire

from seismonn.evaluation.comparison import (
    compare_evaluation_reports,
    format_markdown_table,
    save_comparison_csv,
    save_comparison_markdown,
)


def parse_reports(reports: tuple[str, ...]) -> list[Path]:
    """Parse report paths from Fire varargs.

    Supports:
    - scripts/compare_evaluations.py report1.json report2.json
    - scripts/compare_evaluations.py "report1.json,report2.json"
    """
    if len(reports) == 1 and "," in reports[0]:
        return [Path(path.strip()) for path in reports[0].split(",") if path.strip()]

    return [Path(path) for path in reports]


def main(
    *reports: str,
    sort_by: str = "macro_f1",
    ascending: bool = False,
    output_csv: str | None = None,
    output_md: str | None = None,
    float_precision: int = 4,
) -> None:
    """Compare multiple SeismoNN evaluation JSON reports."""
    report_paths = parse_reports(reports)

    comparison = compare_evaluation_reports(
        report_paths=report_paths,
        sort_by=sort_by,
        ascending=ascending,
    )

    markdown = format_markdown_table(
        comparison=comparison,
        float_precision=float_precision,
    )

    print(markdown)

    if output_csv is not None:
        save_comparison_csv(comparison, output_csv)

    if output_md is not None:
        save_comparison_markdown(
            comparison=comparison,
            output_path=output_md,
            float_precision=float_precision,
        )


if __name__ == "__main__":
    fire.Fire(main)
