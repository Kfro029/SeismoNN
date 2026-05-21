import pandas as pd

from seismonn.data.splits import create_stratified_split, get_split_summary


def test_create_stratified_train_val_split():
    metadata = pd.DataFrame(
        {
            "sample_id": list(range(30)),
            "path": [f"sample_{idx}.npy" for idx in range(30)],
            "crack_count": [3] * 10 + [4] * 10 + [5] * 10,
            "class_id": [0] * 10 + [1] * 10 + [2] * 10,
            "split": ["old"] * 30,
        }
    )

    result = create_stratified_split(
        metadata=metadata,
        val_size=0.2,
        test_size=0.0,
        seed=42,
        stratify_column="class_id",
    )

    counts = result.groupby(["split", "class_id"]).size()

    assert set(result["split"]) == {"train", "val"}

    assert counts.loc[("train", 0)] == 8
    assert counts.loc[("train", 1)] == 8
    assert counts.loc[("train", 2)] == 8

    assert counts.loc[("val", 0)] == 2
    assert counts.loc[("val", 1)] == 2
    assert counts.loc[("val", 2)] == 2

    assert set(result["split_seed"]) == {42}
    assert set(result["split_strategy"]) == {"stratified_random"}
    assert set(result["split_stratify_column"]) == {"class_id"}


def test_create_stratified_train_val_test_split():
    metadata = pd.DataFrame(
        {
            "sample_id": list(range(30)),
            "path": [f"sample_{idx}.npy" for idx in range(30)],
            "crack_count": [3] * 10 + [4] * 10 + [5] * 10,
            "class_id": [0] * 10 + [1] * 10 + [2] * 10,
            "split": ["old"] * 30,
        }
    )

    result = create_stratified_split(
        metadata=metadata,
        val_size=0.2,
        test_size=0.2,
        seed=42,
        stratify_column="class_id",
    )

    counts = result.groupby(["split", "class_id"]).size()

    assert set(result["split"]) == {"train", "val", "test"}

    assert counts.loc[("train", 0)] == 6
    assert counts.loc[("train", 1)] == 6
    assert counts.loc[("train", 2)] == 6

    assert counts.loc[("val", 0)] == 2
    assert counts.loc[("val", 1)] == 2
    assert counts.loc[("val", 2)] == 2

    assert counts.loc[("test", 0)] == 2
    assert counts.loc[("test", 1)] == 2
    assert counts.loc[("test", 2)] == 2


def test_create_stratified_split_is_reproducible():
    metadata = pd.DataFrame(
        {
            "sample_id": list(range(30)),
            "path": [f"sample_{idx}.npy" for idx in range(30)],
            "crack_count": [3] * 10 + [4] * 10 + [5] * 10,
            "class_id": [0] * 10 + [1] * 10 + [2] * 10,
            "split": ["old"] * 30,
        }
    )

    first = create_stratified_split(
        metadata=metadata,
        val_size=0.2,
        test_size=0.0,
        seed=42,
        stratify_column="class_id",
    )

    second = create_stratified_split(
        metadata=metadata,
        val_size=0.2,
        test_size=0.0,
        seed=42,
        stratify_column="class_id",
    )

    assert first["split"].tolist() == second["split"].tolist()


def test_get_split_summary():
    metadata = pd.DataFrame(
        {
            "split": ["train", "train", "val", "val"],
            "crack_count": [3, 4, 3, 4],
        }
    )

    summary = get_split_summary(metadata)

    assert summary == {
        "split_counts": {
            "train": 2,
            "val": 2,
        },
        "class_counts": {
            "train:3": 1,
            "train:4": 1,
            "val:3": 1,
            "val:4": 1,
        },
    }
