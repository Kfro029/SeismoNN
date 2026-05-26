from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import fire

from seismonn.data.inspection import inspect_sample
from seismonn.training.utils import to_jsonable


def save_json(data: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(to_jsonable(data), file, indent=2, ensure_ascii=False)
        file.write("\n")


def inspect(
    metadata: str = "data/metadata.csv",
    data_root: str = ".",
    index: int = 0,
    sample_id: str | None = None,
    sample_path: str | None = None,
    output: str | None = None,
) -> None:
    """Inspect one seismic sample from metadata.csv."""
    selected_index = None if sample_id is not None or sample_path is not None else index

    result = inspect_sample(
        metadata_path=metadata,
        data_root=data_root,
        index=selected_index,
        sample_id=sample_id,
        sample_path=sample_path,
    )

    print(json.dumps(to_jsonable(result), indent=2, ensure_ascii=False))

    if output is not None:
        save_json(result, output)


def main() -> None:
    fire.Fire(inspect)


if __name__ == "__main__":
    main()
