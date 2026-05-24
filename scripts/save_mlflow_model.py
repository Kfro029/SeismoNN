from __future__ import annotations

import json
from pathlib import Path

import fire

from seismonn.serving.mlflow_model import save_mlflow_pyfunc_model


def save_model(
    checkpoint: str,
    output: str,
    device: str = "cpu",
    predictor_type: str = "auto",
    overwrite: bool = True,
) -> None:
    """Save SeismoNN checkpoint as MLflow PyFunc model."""
    metadata = save_mlflow_pyfunc_model(
        checkpoint_path=Path(checkpoint),
        output_path=Path(output),
        device_name=device,
        predictor_type=predictor_type,
        overwrite=overwrite,
    )

    print(json.dumps(metadata, indent=2, ensure_ascii=False))


def main() -> None:
    fire.Fire(save_model)


if __name__ == "__main__":
    main()
