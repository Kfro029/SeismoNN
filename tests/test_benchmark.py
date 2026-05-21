import numpy as np
import torch

from seismonn.inference.benchmark import benchmark_predictor, summarize_latencies
from seismonn.models.cnn import SeismoCNN


def test_summarize_latencies():
    summary = summarize_latencies([0.001, 0.002, 0.003])

    assert summary["runs"] == 3
    assert summary["mean_ms"] == 2.0
    assert summary["min_ms"] == 1.0
    assert summary["max_ms"] == 3.0
    assert summary["throughput_samples_per_sec"] == 500.0


def test_benchmark_predictor_returns_latency_metrics(tmp_path):
    sample = np.random.randn(2, 16, 8).astype("float32")
    sample_path = tmp_path / "sample.npy"
    np.save(sample_path, sample)

    model = SeismoCNN(in_channels=2, num_classes=3)

    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model_name": "cnn_baseline",
            "model_state_dict": model.state_dict(),
            "model_config": {
                "name": "cnn_baseline",
                "in_channels": 2,
                "num_classes": 3,
                "dropout": 0.2,
            },
            "data_config": {
                "normalize": True,
            },
            "input_shape": [2, 16, 8],
            "class_id_to_crack_count": {
                0: 3,
                1: 4,
                2: 5,
            },
        },
        checkpoint_path,
    )

    result = benchmark_predictor(
        checkpoint_path=checkpoint_path,
        input_path=sample_path,
        device_name="cpu",
        warmup_runs=1,
        timed_runs=2,
    )

    assert result["checkpoint_path"] == str(checkpoint_path)
    assert result["input_path"] == str(sample_path)
    assert result["device"] == "cpu"
    assert result["warmup_runs"] == 1
    assert result["timed_runs"] == 2

    assert "example_prediction" in result
    assert "model_only" in result
    assert "end_to_end" in result

    for section_name in ["model_only", "end_to_end"]:
        section = result[section_name]

        assert section["runs"] == 2
        assert section["mean_ms"] > 0.0
        assert section["p50_ms"] > 0.0
        assert section["p95_ms"] > 0.0
        assert section["throughput_samples_per_sec"] > 0.0
