from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


def create_stratified_split(
    metadata: pd.DataFrame,
    val_size: float = 0.2,
    test_size: float = 0.0,
    seed: int = 42,
    stratify_column: str = "class_id",
    split_column: str = "split",
    strategy_name: str = "stratified_random",
) -> pd.DataFrame:
    """Create reproducible stratified train/val or train/val/test split."""
    if stratify_column not in metadata.columns:
        raise ValueError(f"Column {stratify_column!r} not found in metadata.")

    if not 0.0 <= val_size < 1.0:
        raise ValueError(f"val_size must be in [0, 1), got {val_size}")

    if not 0.0 <= test_size < 1.0:
        raise ValueError(f"test_size must be in [0, 1), got {test_size}")

    if val_size + test_size >= 1.0:
        raise ValueError(
            f"val_size + test_size must be less than 1. "
            f"Got val_size={val_size}, test_size={test_size}."
        )

    if val_size == 0.0:
        raise ValueError("val_size must be positive for the current training pipeline.")

    result = metadata.copy().reset_index(drop=True)
    result["_original_order"] = range(len(result))

    if test_size > 0.0:
        train_val_df, test_df = train_test_split(
            result,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
            stratify=result[stratify_column],
        )

        relative_val_size = val_size / (1.0 - test_size)

        train_df, val_df = train_test_split(
            train_val_df,
            test_size=relative_val_size,
            random_state=seed,
            shuffle=True,
            stratify=train_val_df[stratify_column],
        )

        train_df = train_df.copy()
        val_df = val_df.copy()
        test_df = test_df.copy()

        train_df[split_column] = "train"
        val_df[split_column] = "val"
        test_df[split_column] = "test"

        result = pd.concat([train_df, val_df, test_df], axis=0)

    else:
        train_df, val_df = train_test_split(
            result,
            test_size=val_size,
            random_state=seed,
            shuffle=True,
            stratify=result[stratify_column],
        )

        train_df = train_df.copy()
        val_df = val_df.copy()

        train_df[split_column] = "train"
        val_df[split_column] = "val"

        result = pd.concat([train_df, val_df], axis=0)

    result["split_seed"] = seed
    result["split_strategy"] = strategy_name
    result["split_stratify_column"] = stratify_column

    result = result.sort_values("_original_order").drop(columns=["_original_order"])
    result = result.reset_index(drop=True)

    return result


def get_split_summary(
    metadata: pd.DataFrame,
    split_column: str = "split",
    target_column: str = "crack_count",
) -> dict[str, Any]:
    """Return compact split summary for logging and printing."""
    if split_column not in metadata.columns:
        raise ValueError(f"Column {split_column!r} not found in metadata.")

    if target_column not in metadata.columns:
        raise ValueError(f"Column {target_column!r} not found in metadata.")

    split_counts = metadata[split_column].value_counts().sort_index().to_dict()

    class_counts = (
        metadata.groupby([split_column, target_column]).size().sort_index().to_dict()
    )

    class_counts_jsonable = {
        f"{split_name}:{target_value}": int(count)
        for (split_name, target_value), count in class_counts.items()
    }

    return {
        "split_counts": {str(key): int(value) for key, value in split_counts.items()},
        "class_counts": class_counts_jsonable,
    }
