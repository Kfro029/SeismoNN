from __future__ import annotations

import argparse
import json
from pathlib import Path

from seismonn.exporting.torchscript import export_torchscript_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export SeismoNN checkpoint to TorchScript."
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to PyTorch checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to save TorchScript model.",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=None,
        help="Optional path to save export metadata JSON.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device used for export: cpu, cuda, auto.",
    )
    parser.add_argument(
        "--input-shape",
        type=int,
        nargs=3,
        default=None,
        metavar=("C", "T", "R"),
        help="Optional input shape for tracing, for example: --input-shape 2 1723 501.",
    )

    args = parser.parse_args()

    input_shape = None

    if args.input_shape is not None:
        input_shape = tuple(args.input_shape)

    metadata = export_torchscript_checkpoint(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        metadata_output_path=args.metadata_output,
        device_name=args.device,
        input_shape=input_shape,
    )

    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()