from __future__ import annotations

import os
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile

from seismonn.inference.predictor import SeismoPredictor


DEFAULT_CHECKPOINT_PATH = "outputs/cnn_baseline/best.pt"
DEFAULT_DEVICE = "auto"


def create_app(
    checkpoint_path: str | Path | None = None,
    device_name: str | None = None,
) -> FastAPI:
    """Create FastAPI application for SeismoNN inference."""

    resolved_checkpoint_path = Path(
        checkpoint_path or os.getenv("SEISMONN_CHECKPOINT", DEFAULT_CHECKPOINT_PATH)
    )
    resolved_device_name = device_name or os.getenv("SEISMONN_DEVICE", DEFAULT_DEVICE)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.predictor = None
        app.state.startup_error = None
        app.state.checkpoint_path = str(resolved_checkpoint_path)
        app.state.device_name = resolved_device_name

        try:
            app.state.predictor = SeismoPredictor.from_checkpoint(
                checkpoint_path=resolved_checkpoint_path,
                device_name=resolved_device_name,
            )
        except Exception as exc:  # noqa: BLE001
            app.state.startup_error = str(exc)

        yield

    app = FastAPI(
        title="SeismoNN API",
        description="Inference API for fracture count classification from seismic .npy samples.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        predictor_loaded = app.state.predictor is not None

        return {
            "status": "ok" if predictor_loaded else "degraded",
            "model_loaded": predictor_loaded,
            "checkpoint_path": app.state.checkpoint_path,
            "device": app.state.device_name,
            "startup_error": app.state.startup_error,
        }

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
        predictor = app.state.predictor

        if predictor is None:
            raise HTTPException(
                status_code=503,
                detail={
                    "message": "Model is not loaded.",
                    "startup_error": app.state.startup_error,
                },
            )

        if file.filename is None or not file.filename.endswith(".npy"):
            raise HTTPException(
                status_code=400,
                detail="Expected uploaded file with .npy extension.",
            )

        contents = await file.read()

        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        tmp_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp_file:
                tmp_file.write(contents)
                tmp_path = Path(tmp_file.name)

            prediction = predictor.predict_file(tmp_path)

            # Replace temporary path with user-facing filename.
            prediction["input_path"] = file.filename
            prediction["input_filename"] = file.filename

            return prediction

        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

    return app


app = create_app()
