from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from seismonn.inference.predictor import SeismoPredictor


def synchronize_device(device: torch.device) -> None:
    """Synchronize device for correct latency measurements."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def summarize_latencies(latencies_s: list[float]) -> dict[str, float | int]:
    """Summarize latency measurements.

    Args:
        latencies_s: List of latency values in seconds.

    Returns:
        Dictionary with latency statistics in milliseconds and throughput.
    """
    if not latencies_s:
        raise ValueError("latencies_s must contain at least one value.")

    latencies = np.asarray(latencies_s, dtype=np.float64)
    latencies_ms = latencies * 1000.0

    mean_latency_s = float(np.mean(latencies))

    return {
        "runs": int(len(latencies)),
        "mean_ms": float(np.mean(latencies_ms)),
        "std_ms": float(np.std(latencies_ms)),
        "min_ms": float(np.min(latencies_ms)),
        "p50_ms": float(np.percentile(latencies_ms, 50)),
        "p95_ms": float(np.percentile(latencies_ms, 95)),
        "p99_ms": float(np.percentile(latencies_ms, 99)),
        "max_ms": float(np.max(latencies_ms)),
        "throughput_samples_per_sec": float(1.0 / mean_latency_s),
    }


def benchmark_model_only(
    predictor: SeismoPredictor,
    input_path: str | Path,
    warmup_runs: int = 3,
    timed_runs: int = 20,
) -> dict[str, float | int]:
    """Benchmark only model forward pass on a preloaded tensor.

    This excludes .npy loading and preprocessing time.
    """
    if warmup_runs < 0:
        raise ValueError(f"warmup_runs must be non-negative, got {warmup_runs}")

    if timed_runs <= 0:
        raise ValueError(f"timed_runs must be positive, got {timed_runs}")

    x = predictor.load_sample(input_path).to(predictor.device)

    def run_once() -> None:
        with torch.no_grad():
            logits = predictor.model(x)
            _ = torch.softmax(logits, dim=1)

        synchronize_device(predictor.device)

    for _ in range(warmup_runs):
        run_once()

    latencies_s: list[float] = []

    for _ in range(timed_runs):
        start_time = time.perf_counter()
        run_once()
        end_time = time.perf_counter()

        latencies_s.append(end_time - start_time)

    return summarize_latencies(latencies_s)


def benchmark_end_to_end(
    predictor: SeismoPredictor,
    input_path: str | Path,
    warmup_runs: int = 3,
    timed_runs: int = 20,
) -> dict[str, float | int]:
    """Benchmark full prediction pipeline.

    This includes .npy loading, preprocessing, model forward pass,
    softmax and JSON-friendly prediction formatting.
    """
    if warmup_runs < 0:
        raise ValueError(f"warmup_runs must be non-negative, got {warmup_runs}")

    if timed_runs <= 0:
        raise ValueError(f"timed_runs must be positive, got {timed_runs}")

    def run_once() -> None:
        _ = predictor.predict_file(input_path)
        synchronize_device(predictor.device)

    for _ in range(warmup_runs):
        run_once()

    latencies_s: list[float] = []

    for _ in range(timed_runs):
        start_time = time.perf_counter()
        run_once()
        end_time = time.perf_counter()

        latencies_s.append(end_time - start_time)

    return summarize_latencies(latencies_s)


def benchmark_predictor(
    checkpoint_path: str | Path,
    input_path: str | Path,
    device_name: str = "auto",
    warmup_runs: int = 3,
    timed_runs: int = 20,
) -> dict[str, Any]:
    """Benchmark SeismoPredictor loaded from checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    input_path = Path(input_path)

    load_start_time = time.perf_counter()
    predictor = SeismoPredictor.from_checkpoint(
        checkpoint_path=checkpoint_path,
        device_name=device_name,
    )
    load_end_time = time.perf_counter()

    example_prediction = predictor.predict_file(input_path)

    return {
        "checkpoint_path": str(checkpoint_path),
        "input_path": str(input_path),
        "device": str(predictor.device),
        "warmup_runs": warmup_runs,
        "timed_runs": timed_runs,
        "model_load_time_ms": float((load_end_time - load_start_time) * 1000.0),
        "example_prediction": example_prediction,
        "model_only": benchmark_model_only(
            predictor=predictor,
            input_path=input_path,
            warmup_runs=warmup_runs,
            timed_runs=timed_runs,
        ),
        "end_to_end": benchmark_end_to_end(
            predictor=predictor,
            input_path=input_path,
            warmup_runs=warmup_runs,
            timed_runs=timed_runs,
        ),
    }
