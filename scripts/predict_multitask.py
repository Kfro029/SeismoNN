from __future__ import annotations

import argparse
import json
from pathlib import Path

from seismonn.inference.multitask_predictor import (
    SeismoMultiTaskPredictor,
    save_multitask_prediction_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict fracture count and regression parameters for one .npy sample."
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to trained multi-task checkpoint.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input .npy file with shape (2, T, R).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save prediction JSON.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto, cpu, cuda.",
    )

    args = parser.parse_args()

    predictor = SeismoMultiTaskPredictor.from_checkpoint(
        checkpoint_path=args.checkpoint,
        device_name=args.device,
    )

    prediction = predictor.predict_file(args.input)

    prediction_json = json.dumps(prediction, indent=2, ensure_ascii=False)
    print(prediction_json)

    if args.output is not None:
        save_multitask_prediction_json(prediction, args.output)


if __name__ == "__main__":
    main()
