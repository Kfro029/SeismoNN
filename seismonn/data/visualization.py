from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from seismonn.data.inspection import inspect_sample, select_metadata_row  # noqa: E402


COMPONENT_NAMES = {
    0: "vx",
    1: "vy",
}


def load_sample_array_from_metadata(
    metadata_path: str | Path,
    data_root: str | Path = ".",
    index: int | None = 0,
    sample_id: str | None = None,
    sample_path: str | None = None,
) -> tuple[np.ndarray, pd.Series, Path]:
    """Load one sample array using metadata selectors."""
    metadata_path = Path(metadata_path)
    data_root = Path(data_root)

    metadata = pd.read_csv(metadata_path)

    row = select_metadata_row(
        metadata=metadata,
        index=index,
        sample_id=sample_id,
        sample_path=sample_path,
    )

    relative_path = Path(str(row["path"]))
    full_path = (
        relative_path if relative_path.is_absolute() else data_root / relative_path
    )

    if not full_path.exists():
        raise FileNotFoundError(f"Sample file does not exist: {full_path}")

    array = np.load(full_path, mmap_mode="r")
    array = np.asarray(array, dtype=np.float32)

    if array.ndim != 3:
        raise ValueError(f"Expected sample shape [C, T, R], got {array.shape}")

    if array.shape[0] != 2:
        raise ValueError(f"Expected 2 components, got shape {array.shape}")

    return array, row, full_path


def crop_sample_for_plot(
    array: np.ndarray,
    max_time_steps: int | None = None,
    max_receivers: int | None = None,
) -> np.ndarray:
    """Crop sample for faster and clearer plotting."""
    if array.ndim != 3:
        raise ValueError(f"Expected sample shape [C, T, R], got {array.shape}")

    time_slice = slice(None)
    receiver_slice = slice(None)

    if max_time_steps is not None:
        if max_time_steps <= 0:
            raise ValueError(f"max_time_steps must be positive, got {max_time_steps}")
        time_slice = slice(0, max_time_steps)

    if max_receivers is not None:
        if max_receivers <= 0:
            raise ValueError(f"max_receivers must be positive, got {max_receivers}")
        receiver_slice = slice(0, max_receivers)

    return array[:, time_slice, receiver_slice]


def plot_component_heatmap(
    array: np.ndarray,
    component_index: int,
    output_path: str | Path,
    title: str | None = None,
) -> None:
    """Save heatmap for one velocity component.

    The plotted matrix has axes:
    - x axis: receiver index
    - y axis: time step
    """
    if component_index not in COMPONENT_NAMES:
        raise ValueError(f"Unsupported component_index={component_index}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    component = array[component_index]

    plt.figure(figsize=(10, 6))
    plt.imshow(component, aspect="auto", origin="lower")
    plt.xlabel("Receiver index")
    plt.ylabel("Time step")
    plt.title(title or f"{COMPONENT_NAMES[component_index]} component")
    plt.colorbar(label="Velocity component value")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_receiver_trace(
    array: np.ndarray,
    component_index: int,
    receiver_index: int,
    output_path: str | Path,
    title: str | None = None,
) -> None:
    """Save one receiver trace for one component."""
    if component_index not in COMPONENT_NAMES:
        raise ValueError(f"Unsupported component_index={component_index}")

    if receiver_index < 0 or receiver_index >= array.shape[2]:
        raise ValueError(
            f"receiver_index={receiver_index} is out of range for "
            f"{array.shape[2]} receivers."
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    trace = array[component_index, :, receiver_index]
    time_steps = np.arange(array.shape[1])

    plt.figure(figsize=(10, 4))
    plt.plot(time_steps, trace)
    plt.xlabel("Time step")
    plt.ylabel("Velocity component value")
    plt.title(
        title or f"{COMPONENT_NAMES[component_index]} trace, receiver={receiver_index}"
    )
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_json(data: dict[str, Any], output_path: str | Path) -> None:
    """Save JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        file.write("\n")


def visualize_sample(
    metadata_path: str | Path,
    data_root: str | Path = ".",
    output_dir: str | Path = "outputs/sample_visualization",
    index: int | None = 0,
    sample_id: str | None = None,
    sample_path: str | None = None,
    receiver_index: int | None = None,
    max_time_steps: int | None = None,
    max_receivers: int | None = None,
) -> dict[str, Any]:
    """Create visualizations for one sample.

    Returns paths to generated files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    array, _row, full_path = load_sample_array_from_metadata(
        metadata_path=metadata_path,
        data_root=data_root,
        index=index,
        sample_id=sample_id,
        sample_path=sample_path,
    )

    if receiver_index is None:
        receiver_index = array.shape[2] // 2

    if receiver_index < 0 or receiver_index >= array.shape[2]:
        raise ValueError(
            f"receiver_index={receiver_index} is out of range for "
            f"{array.shape[2]} receivers."
        )

    plotted_array = crop_sample_for_plot(
        array=array,
        max_time_steps=max_time_steps,
        max_receivers=max_receivers,
    )

    plotted_receiver_index = min(receiver_index, plotted_array.shape[2] - 1)

    vx_heatmap_path = output_dir / "vx_heatmap.png"
    vy_heatmap_path = output_dir / "vy_heatmap.png"
    vx_trace_path = output_dir / "vx_receiver_trace.png"
    vy_trace_path = output_dir / "vy_receiver_trace.png"
    sample_info_path = output_dir / "sample_info.json"

    plot_component_heatmap(
        array=plotted_array,
        component_index=0,
        output_path=vx_heatmap_path,
        title="vx component heatmap",
    )
    plot_component_heatmap(
        array=plotted_array,
        component_index=1,
        output_path=vy_heatmap_path,
        title="vy component heatmap",
    )
    plot_receiver_trace(
        array=plotted_array,
        component_index=0,
        receiver_index=plotted_receiver_index,
        output_path=vx_trace_path,
        title=f"vx trace, receiver={plotted_receiver_index}",
    )
    plot_receiver_trace(
        array=plotted_array,
        component_index=1,
        receiver_index=plotted_receiver_index,
        output_path=vy_trace_path,
        title=f"vy trace, receiver={plotted_receiver_index}",
    )

    sample_info = inspect_sample(
        metadata_path=metadata_path,
        data_root=data_root,
        index=index,
        sample_id=sample_id,
        sample_path=sample_path,
    )
    sample_info["visualization"] = {
        "source_file": str(full_path),
        "output_dir": str(output_dir),
        "receiver_index": receiver_index,
        "plotted_receiver_index": plotted_receiver_index,
        "max_time_steps": max_time_steps,
        "max_receivers": max_receivers,
        "files": {
            "vx_heatmap": str(vx_heatmap_path),
            "vy_heatmap": str(vy_heatmap_path),
            "vx_receiver_trace": str(vx_trace_path),
            "vy_receiver_trace": str(vy_trace_path),
        },
    }

    save_json(sample_info, sample_info_path)

    return {
        "output_dir": str(output_dir),
        "sample_info": str(sample_info_path),
        "vx_heatmap": str(vx_heatmap_path),
        "vy_heatmap": str(vy_heatmap_path),
        "vx_receiver_trace": str(vx_trace_path),
        "vy_receiver_trace": str(vy_trace_path),
    }
