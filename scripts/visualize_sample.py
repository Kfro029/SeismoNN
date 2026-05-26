from __future__ import annotations

import json

import fire

from seismonn.data.visualization import visualize_sample
from seismonn.training.utils import to_jsonable


def visualize(
    metadata: str = "data/metadata.csv",
    data_root: str = ".",
    output_dir: str = "outputs/sample_visualization",
    index: int = 0,
    sample_id: str | None = None,
    sample_path: str | None = None,
    receiver_index: int | None = None,
    max_time_steps: int | None = None,
    max_receivers: int | None = None,
) -> None:
    """Create visualizations for one SeismoNN sample."""
    selected_index = None if sample_id is not None or sample_path is not None else index

    result = visualize_sample(
        metadata_path=metadata,
        data_root=data_root,
        output_dir=output_dir,
        index=selected_index,
        sample_id=sample_id,
        sample_path=sample_path,
        receiver_index=receiver_index,
        max_time_steps=max_time_steps,
        max_receivers=max_receivers,
    )

    print(json.dumps(to_jsonable(result), indent=2, ensure_ascii=False))


def main() -> None:
    fire.Fire(visualize)


if __name__ == "__main__":
    main()
