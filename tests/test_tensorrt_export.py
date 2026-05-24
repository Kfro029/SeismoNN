from seismonn.exporting.tensorrt import (
    build_shape_profile_args,
    build_trtexec_command,
    export_tensorrt_engine,
)


def test_build_shape_profile_args():
    args = build_shape_profile_args(
        input_name="features",
        input_shape=(2, 16, 8),
        min_batch_size=1,
        opt_batch_size=2,
        max_batch_size=4,
    )

    assert args == [
        "--minShapes=features:1x2x16x8",
        "--optShapes=features:2x2x16x8",
        "--maxShapes=features:4x2x16x8",
    ]


def test_build_trtexec_command():
    command = build_trtexec_command(
        onnx_path="model.onnx",
        engine_path="model.engine",
        input_name="features",
        input_shape=(2, 16, 8),
        fp16=True,
        verbose=True,
    )

    assert command[0] == "trtexec"
    assert "--onnx=model.onnx" in command
    assert "--saveEngine=model.engine" in command
    assert "--minShapes=features:1x2x16x8" in command
    assert "--optShapes=features:1x2x16x8" in command
    assert "--maxShapes=features:1x2x16x8" in command
    assert "--fp16" in command
    assert "--verbose" in command


def test_export_tensorrt_engine_dry_run(tmp_path):
    onnx_path = tmp_path / "model.onnx"
    engine_path = tmp_path / "model.engine"
    metadata_path = tmp_path / "model.engine.metadata.json"

    onnx_path.write_bytes(b"fake onnx for dry run")

    metadata = export_tensorrt_engine(
        onnx_path=onnx_path,
        engine_path=engine_path,
        metadata_output_path=metadata_path,
        input_name="features",
        input_shape=(2, 16, 8),
        fp16=False,
        dry_run=True,
    )

    assert metadata_path.exists()
    assert metadata["dry_run"] is True
    assert metadata["onnx_path"] == str(onnx_path)
    assert metadata["engine_path"] == str(engine_path)
    assert "--onnx=" + str(onnx_path) in metadata["command"]
    assert "--saveEngine=" + str(engine_path) in metadata["command"]
