from __future__ import annotations

from pathlib import Path

import fire

from seismonn.reporting.results import generate_results_markdown, save_results_markdown


def parse_paths(paths: str | tuple[str, ...] | None) -> list[Path] | None:
    """Parse optional path list from Fire CLI value."""
    if paths is None:
        return None

    if isinstance(paths, tuple):
        return [Path(path) for path in paths]

    if "," in str(paths):
        return [Path(path.strip()) for path in str(paths).split(",") if path.strip()]

    return [Path(str(paths))]


def main(
    evaluation_reports: str | tuple[str, ...] | None = None,
    multitask_summaries: str | tuple[str, ...] | None = None,
    benchmarks: str | tuple[str, ...] | None = None,
    output: str = "RESULTS.md",
) -> None:
    """Generate RESULTS.md from SeismoNN output artifacts."""
    markdown = generate_results_markdown(
        evaluation_report_paths=parse_paths(evaluation_reports),
        multitask_summary_paths=parse_paths(multitask_summaries),
        benchmark_paths=parse_paths(benchmarks),
    )

    save_results_markdown(
        markdown=markdown,
        output_path=output,
    )

    print(f"Saved results report to: {output}")


if __name__ == "__main__":
    fire.Fire(main)
