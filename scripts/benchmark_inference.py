from __future__ import annotations

import json

import fire

from seismonn.inference.benchmark import benchmark_predictor
from seismonn.training.utils import to_jsonable


def save_json(data, output_path: str) -> None:
    from pathlib import Path

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(to_jsonable(data), file, indent=2, ensure_ascii=False)
        file.write("\n")


def main(
    checkpoint: str,
    input_path: str,
    device: str = "auto",
    warmup_runs: int = 3,
    timed_runs: int = 20,
    output: str | None = None,
) -> None:
    """Benchmark SeismoNN inference latency and throughput."""
    result = benchmark_predictor(
        checkpoint_path=checkpoint,
        input_path=input_path,
        device_name=device,
        warmup_runs=warmup_runs,
        timed_runs=timed_runs,
    )

    print(json.dumps(to_jsonable(result), indent=2, ensure_ascii=False))

    if output is not None:
        save_json(result, output)


if __name__ == "__main__":
    fire.Fire(main)
