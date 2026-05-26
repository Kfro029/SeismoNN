from __future__ import annotations

import json
from pathlib import Path

import fire

from seismonn.inference.multitask_predictor import (
    SeismoMultiTaskPredictor,
    save_multitask_prediction_json,
)
from seismonn.training.utils import to_jsonable


def predict_multitask(
    checkpoint: str,
    input_path: str,
    output: str | None = None,
    device: str = "auto",
) -> None:
    """Predict fracture count and regression parameters for one .npy sample."""
    predictor = SeismoMultiTaskPredictor.from_checkpoint(
        checkpoint_path=checkpoint,
        device_name=device,
    )

    prediction = predictor.predict_file(input_path)

    print(json.dumps(to_jsonable(prediction), indent=2, ensure_ascii=False))

    if output is not None:
        save_multitask_prediction_json(
            prediction=prediction,
            output_path=Path(output),
        )


def main() -> None:
    fire.Fire(predict_multitask)


if __name__ == "__main__":
    main()
