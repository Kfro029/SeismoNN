from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import fire

from seismonn.inference.predictor import SeismoPredictor
from seismonn.training.utils import to_jsonable


def save_json(data: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(to_jsonable(data), file, indent=2, ensure_ascii=False)
        file.write("\n")


def predict(
    checkpoint: str,
    input_path: str,
    output: str | None = None,
    device: str = "auto",
) -> None:
    """Predict fracture count for one .npy sample."""
    predictor = SeismoPredictor.from_checkpoint(
        checkpoint_path=checkpoint,
        device_name=device,
    )

    prediction = predictor.predict_file(input_path)

    print(json.dumps(to_jsonable(prediction), indent=2, ensure_ascii=False))

    if output is not None:
        save_json(prediction, output)


def main() -> None:
    fire.Fire(predict)


if __name__ == "__main__":
    main()
