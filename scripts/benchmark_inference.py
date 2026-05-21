from __future__ import annotations

import argparse
import json
from pathlib import Path

from seismonn.inference.benchmark import benchmark_predictor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark SeismoNN inference latency and throughput."
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to trained model checkpoint.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input .npy file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto, cpu, cuda.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=3,
        help="Number of warmup runs.",
    )
    parser.add_argument(
        "--timed-runs",
        type=int,
        default=20,
        help="Number of timed runs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save benchmark result as JSON.",
    )

    args = parser.parse_args()

    result = benchmark_predictor(
        checkpoint_path=args.checkpoint,
        input_path=args.input,
        device_name=args.device,
        warmup_runs=args.warmup_runs,
        timed_runs=args.timed_runs,
    )

    result_json = json.dumps(result, indent=2, ensure_ascii=False)
    print(result_json)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)

        with args.output.open("w", encoding="utf-8") as file:
            file.write(result_json)
            file.write("\n")


if __name__ == "__main__":
    main()
