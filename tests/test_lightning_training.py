import numpy as np
import pandas as pd
import torch
from lightning.pytorch import Trainer

from seismonn.lightning.classifier import SeismoClassifierLightningModule
from seismonn.lightning.datamodule import SeismoDataModule


def create_tiny_metadata(tmp_path):
    rows = []

    class_ids = [0, 1, 2, 0, 1, 2]
    crack_counts = [3, 4, 5, 3, 4, 5]

    for sample_index in range(6):
        sample = np.random.randn(2, 16, 8).astype("float32")
        sample_path = tmp_path / f"sample_{sample_index}.npy"
        np.save(sample_path, sample)

        rows.append(
            {
                "sample_id": f"{sample_index:06d}",
                "path": sample_path.name,
                "filename": sample_path.name,
                "split": "train" if sample_index < 3 else "val",
                "crack_count": crack_counts[sample_index],
                "class_id": class_ids[sample_index],
            }
        )

    metadata = pd.DataFrame(rows)
    metadata_path = tmp_path / "metadata.csv"
    metadata.to_csv(metadata_path, index=False)

    return metadata_path


def test_lightning_datamodule_builds_dataloaders(tmp_path):
    metadata_path = create_tiny_metadata(tmp_path)

    data_module = SeismoDataModule(
        metadata_path=metadata_path,
        data_root=tmp_path,
        batch_size=2,
        num_workers=0,
        normalize=True,
        pin_memory=False,
    )

    data_module.setup("fit")

    train_batch = next(iter(data_module.train_dataloader()))
    val_batch = next(iter(data_module.val_dataloader()))

    train_features, train_targets = train_batch
    val_features, val_targets = val_batch

    assert train_features.shape[1:] == torch.Size([2, 16, 8])
    assert val_features.shape[1:] == torch.Size([2, 16, 8])
    assert train_targets.dtype == torch.long
    assert val_targets.dtype == torch.long


def test_lightning_classifier_forward_shape():
    model = SeismoClassifierLightningModule(
        model_config={
            "name": "cnn_baseline",
            "in_channels": 2,
            "num_classes": 3,
            "dropout": 0.2,
        },
        optimizer_config={
            "name": "adamw",
            "lr": 0.001,
            "weight_decay": 0.0,
        },
    )

    features = torch.randn(4, 2, 16, 8)
    logits = model(features)

    assert logits.shape == torch.Size([4, 3])


def test_lightning_training_fast_dev_run(tmp_path):
    metadata_path = create_tiny_metadata(tmp_path)

    data_module = SeismoDataModule(
        metadata_path=metadata_path,
        data_root=tmp_path,
        batch_size=2,
        num_workers=0,
        normalize=True,
        pin_memory=False,
    )

    model = SeismoClassifierLightningModule(
        model_config={
            "name": "cnn_baseline",
            "in_channels": 2,
            "num_classes": 3,
            "dropout": 0.2,
        },
        optimizer_config={
            "name": "adamw",
            "lr": 0.001,
            "weight_decay": 0.0,
        },
    )

    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        fast_dev_run=True,
        logger=False,
        enable_checkpointing=False,
    )

    trainer.fit(
        model=model,
        datamodule=data_module,
    )
