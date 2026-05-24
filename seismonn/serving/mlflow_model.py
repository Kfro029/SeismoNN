from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import mlflow.pyfunc
import pandas as pd
import numpy as np

from seismonn.inference.factory import create_predictor


def flatten_prediction(
    prediction: dict[str, Any],
    prefix: str = "",
) -> dict[str, Any]:
    """Flatten nested prediction dictionary for tabular MLflow output."""
    flattened: dict[str, Any] = {}

    for key, value in prediction.items():
        output_key = f"{prefix}{key}" if prefix else str(key)

        if isinstance(value, dict):
            nested = flatten_prediction(
                prediction=value,
                prefix=f"{output_key}_",
            )
            flattened.update(nested)
        else:
            flattened[output_key] = value

    return flattened


def _extract_input_paths(model_input: Any) -> list[str]:
    """Extract input paths from MLflow Serving input.

    Supports:
    - pandas.DataFrame with column "input_path"
    - numpy array produced by MLflow "inputs" JSON format
    - list of dicts
    - list of strings
    """
    if isinstance(model_input, pd.DataFrame):
        if "input_path" not in model_input.columns:
            raise ValueError("MLflow input dataframe must contain column 'input_path'.")
        return [str(value) for value in model_input["input_path"].tolist()]

    if isinstance(model_input, np.ndarray):
        if model_input.ndim == 0:
            return [str(model_input.item())]

        if model_input.ndim == 1:
            return [str(value) for value in model_input.tolist()]

        if model_input.ndim == 2 and model_input.shape[1] == 1:
            return [str(value[0]) for value in model_input.tolist()]

        raise ValueError(
            f"Unsupported numpy input shape for MLflow Serving: {model_input.shape}. "
            "Expected shape [N] or [N, 1] with input paths."
        )

    if isinstance(model_input, list):
        input_paths = []

        for item in model_input:
            if isinstance(item, dict):
                if "input_path" not in item:
                    raise ValueError("Each input dict must contain 'input_path'.")
                input_paths.append(str(item["input_path"]))
            else:
                input_paths.append(str(item))

        return input_paths

    raise ValueError(f"Unsupported MLflow input type: {type(model_input)}")


class SeismoPyFuncModel(mlflow.pyfunc.PythonModel):
    """MLflow PyFunc wrapper for SeismoNN predictors.

    Expected model input:
        pandas.DataFrame with column "input_path".

    Example:
        pd.DataFrame([{"input_path": "2nd_selection/sample.npy"}])
    """

    def __init__(
        self,
        device_name: str = "cpu",
        predictor_type: str = "auto",
    ) -> None:
        self.device_name = device_name
        self.predictor_type = predictor_type
        self.loaded_predictor = None

    def load_context(self, context: Any) -> None:
        """Load predictor from checkpoint artifact."""
        checkpoint_path = context.artifacts["checkpoint"]

        self.loaded_predictor = create_predictor(
            checkpoint_path=checkpoint_path,
            device_name=self.device_name,
            predictor_type=self.predictor_type,
        )

    def predict(
        self,
        context: Any,
        model_input: Any,
    ) -> pd.DataFrame:
        """Run prediction for input paths from MLflow input."""
        del context

        if self.loaded_predictor is None:
            raise RuntimeError(
                "Predictor is not loaded. Did MLflow call load_context()?"
            )

        input_paths = _extract_input_paths(model_input)

        rows: list[dict[str, Any]] = []

        for input_path in input_paths:
            prediction = self.loaded_predictor.predictor.predict_file(input_path)
            flattened = flatten_prediction(prediction)
            flattened["predictor_type"] = self.loaded_predictor.predictor_type
            flattened["loaded_model_name"] = self.loaded_predictor.model_name
            rows.append(flattened)

        return pd.DataFrame(rows)


def save_mlflow_pyfunc_model(
    checkpoint_path: str | Path,
    output_path: str | Path,
    device_name: str = "cpu",
    predictor_type: str = "auto",
    overwrite: bool = True,
) -> dict[str, Any]:
    """Save SeismoNN checkpoint as MLflow PyFunc model."""
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")

    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output path already exists: {output_path}")
        shutil.rmtree(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    mlflow.pyfunc.save_model(
        path=str(output_path),
        python_model=SeismoPyFuncModel(
            device_name=device_name,
            predictor_type=predictor_type,
        ),
        artifacts={
            "checkpoint": str(checkpoint_path),
        },
        pip_requirements=[
            "mlflow",
            "torch",
            "numpy",
            "pandas",
            "scikit-learn",
            "pyyaml",
        ],
    )

    return {
        "checkpoint_path": str(checkpoint_path),
        "mlflow_model_path": str(output_path),
        "device_name": device_name,
        "predictor_type": predictor_type,
    }
