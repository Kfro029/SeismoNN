from __future__ import annotations

import json

import fire

from seismonn.evaluation.checkpoint import (
    evaluate_checkpoint,
    save_evaluation_report,
)
from seismonn.training.utils import to_jsonable


def main(
    checkpoint: str,
    metadata: str = "data/metadata.csv",
    data_root: str = ".",
    split: str = "val",
    batch_size: int = 16,
    num_workers: int = 0,
    device: str = "auto",
    no_normalize: bool = False,
    output: str | None = None,
) -> None:
    """Evaluate a trained SeismoNN classification checkpoint."""
    normalize = False if no_normalize else None

    report = evaluate_checkpoint(
        checkpoint_path=checkpoint,
        metadata_path=metadata,
        split=split,
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        normalize=normalize,
        device_name=device,
    )

    print(json.dumps(to_jsonable(report), indent=2, ensure_ascii=False))

    if output is not None:
        save_evaluation_report(report, output)


if __name__ == "__main__":
    fire.Fire(main)
