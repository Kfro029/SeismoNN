from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from seismonn.data.dataset import SeismoDataset
from seismonn.models.factory import create_model
from seismonn.tracking.mlflow import (
    is_mlflow_enabled,
    log_mlflow_artifacts,
    log_mlflow_metrics,
    log_mlflow_params,
    start_mlflow_run,
)
from seismonn.training.evaluate import evaluate_classifier
from seismonn.training.transfer import maybe_load_pretrained_encoder
from seismonn.training.utils import (
    get_device,
    resolve_path,
    save_json,
    set_seed,
    to_jsonable,
)


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML training config."""
    config_path = Path(config_path)

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Config must be a dictionary, got: {type(config)}")

    return config


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """Train classifier for one epoch."""
    model.train()

    total_loss = 0.0
    total_samples = 0
    correct = 0

    for features, targets in dataloader:
        features = features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(features)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

        predictions = torch.argmax(logits, dim=1)
        correct += int((predictions == targets).sum().item())

    if total_samples == 0:
        raise ValueError("Cannot train on an empty dataloader.")

    return {
        "loss": total_loss / total_samples,
        "accuracy": correct / total_samples,
    }


def build_optimizer(
    model: nn.Module,
    optimizer_config: dict[str, Any],
) -> torch.optim.Optimizer:
    """Build optimizer from config."""
    optimizer_name = str(optimizer_config.get("name", "adam")).lower()
    learning_rate = float(optimizer_config.get("lr", 1e-3))
    weight_decay = float(optimizer_config.get("weight_decay", 0.0))

    if optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def get_optimized_metric(
    metric_name: str,
    train_metrics: dict[str, Any],
    val_metrics: dict[str, Any],
) -> float:
    """Get metric value used for best checkpoint selection."""
    if metric_name == "val_accuracy":
        return float(val_metrics["accuracy"])

    if metric_name == "val_macro_f1":
        return float(val_metrics["macro_f1"])

    if metric_name == "val_balanced_accuracy":
        return float(val_metrics["balanced_accuracy"])

    if metric_name == "train_accuracy":
        return float(train_metrics["accuracy"])

    raise ValueError(f"Unsupported metric_to_optimize: {metric_name}")


def save_training_plots(history: list[dict[str, Any]], output_dir: Path) -> None:
    """Save training curves."""
    history_df = pd.DataFrame(history)

    plt.figure()
    plt.plot(history_df["epoch"], history_df["train_loss"], label="train")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss.png")
    plt.close()

    plt.figure()
    plt.plot(history_df["epoch"], history_df["train_accuracy"], label="train")
    plt.plot(history_df["epoch"], history_df["val_accuracy"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy.png")
    plt.close()

    plt.figure()
    plt.plot(history_df["epoch"], history_df["val_macro_f1"], label="val_macro_f1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "macro_f1.png")
    plt.close()


def save_confusion_matrix_plot(
    confusion_matrix: Any,
    output_dir: Path,
) -> None:
    """Save confusion matrix plot."""
    plt.figure()
    plt.imshow(confusion_matrix)
    plt.title("Validation confusion matrix")
    plt.xlabel("Predicted class_id")
    plt.ylabel("True class_id")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png")
    plt.close()


def run_classifier_training(
    config_path: str | Path = "configs/train/cnn.yaml",
) -> None:
    """Run classifier training from YAML config."""
    project_root = Path(__file__).resolve().parents[2]
    config = load_config(resolve_path(project_root, config_path))

    set_seed(int(config["seed"]))
    device = get_device(str(config["device"]))

    data_config = config["data"]
    model_config = config["model"]
    optimizer_config = config["optimizer"]
    training_config = config["training"]
    tracking_config = config.get("tracking", {})
    pretrained_config = config.get("pretrained", {})

    output_dir = resolve_path(project_root, training_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = resolve_path(project_root, data_config["metadata_path"])
    data_root = resolve_path(project_root, data_config["data_root"])

    train_dataset = SeismoDataset(
        metadata_path=metadata_path,
        split=data_config["train_split"],
        data_root=data_root,
        normalize=bool(data_config["normalize"]),
    )

    val_dataset = SeismoDataset(
        metadata_path=metadata_path,
        split=data_config["val_split"],
        data_root=data_root,
        normalize=bool(data_config["normalize"]),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(data_config["batch_size"]),
        shuffle=True,
        num_workers=int(data_config["num_workers"]),
        pin_memory=device.type == "cuda",
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=int(data_config["batch_size"]),
        shuffle=False,
        num_workers=int(data_config["num_workers"]),
        pin_memory=device.type == "cuda",
    )

    model = create_model(model_config).to(device)

    transfer_summary = maybe_load_pretrained_encoder(
        model=model,
        pretrained_config=pretrained_config,
        project_root=project_root,
        device=device,
    )

    if transfer_summary is not None:
        print("Loaded pretrained encoder weights.")
        print(f"Checkpoint: {transfer_summary['checkpoint_path']}")
        print(f"Loaded keys: {transfer_summary['loaded_key_count']}")

        save_json(
            to_jsonable(transfer_summary),
            output_dir / "pretrained_transfer.json",
        )

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, optimizer_config)

    num_epochs = int(training_config["num_epochs"])
    metric_to_optimize = str(training_config["metric_to_optimize"])

    best_metric = float("-inf")
    best_epoch = -1
    history: list[dict[str, Any]] = []

    print(f"Device: {device}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Output dir: {output_dir}")

    labels = list(range(int(model_config["num_classes"])))

    with start_mlflow_run(project_root=project_root, tracking_config=tracking_config):
        if is_mlflow_enabled(tracking_config):
            log_mlflow_params(config)

        for epoch in range(1, num_epochs + 1):
            train_metrics = train_one_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
            )

            val_metrics = evaluate_classifier(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
                labels=labels,
            )

            current_metric = get_optimized_metric(
                metric_name=metric_to_optimize,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
            )

            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "val_macro_precision": val_metrics["macro_precision"],
                "val_macro_recall": val_metrics["macro_recall"],
                "val_macro_f1": val_metrics["macro_f1"],
            }

            history.append(row)

            print(
                f"Epoch {epoch:03d}/{num_epochs:03d} | "
                f"train_loss={row['train_loss']:.4f} | "
                f"train_acc={row['train_accuracy']:.4f} | "
                f"val_loss={row['val_loss']:.4f} | "
                f"val_acc={row['val_accuracy']:.4f} | "
                f"val_bal_acc={row['val_balanced_accuracy']:.4f} | "
                f"val_prec={row['val_macro_precision']:.4f} | "
                f"val_rec={row['val_macro_recall']:.4f} | "
                f"val_macro_f1={row['val_macro_f1']:.4f}"
            )

            if is_mlflow_enabled(tracking_config):
                log_mlflow_metrics(
                    {
                        "train_loss": row["train_loss"],
                        "train_accuracy": row["train_accuracy"],
                        "val_loss": row["val_loss"],
                        "val_accuracy": row["val_accuracy"],
                        "val_balanced_accuracy": row["val_balanced_accuracy"],
                        "val_macro_precision": row["val_macro_precision"],
                        "val_macro_recall": row["val_macro_recall"],
                        "val_macro_f1": row["val_macro_f1"],
                    },
                    step=epoch,
                )

            if current_metric > best_metric:
                best_metric = current_metric
                best_epoch = epoch

                checkpoint = {
                    "model_name": model_config["name"],
                    "model_state_dict": model.state_dict(),
                    "model_config": model_config,
                    "data_config": data_config,
                    "pretrained": transfer_summary,
                    "epoch": epoch,
                    "metric_to_optimize": metric_to_optimize,
                    "best_metric": best_metric,
                    "input_shape": data_config["input_shape"],
                    "class_id_to_crack_count": config["checkpoint"][
                        "class_id_to_crack_count"
                    ],
                    "val_metrics": {
                        "loss": val_metrics["loss"],
                        "accuracy": val_metrics["accuracy"],
                        "balanced_accuracy": val_metrics["balanced_accuracy"],
                        "macro_precision": val_metrics["macro_precision"],
                        "macro_recall": val_metrics["macro_recall"],
                        "macro_f1": val_metrics["macro_f1"],
                        "confusion_matrix": val_metrics["confusion_matrix"],
                    },
                }

                torch.save(checkpoint, output_dir / "best.pt")

        final_val_metrics = evaluate_classifier(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            labels=labels,
        )

        torch.save(
            {
                "model_name": model_config["name"],
                "model_state_dict": model.state_dict(),
                "model_config": model_config,
                "data_config": data_config,
                "pretrained": transfer_summary,
                "epoch": num_epochs,
                "input_shape": data_config["input_shape"],
                "class_id_to_crack_count": config["checkpoint"][
                    "class_id_to_crack_count"
                ],
            },
            output_dir / "last.pt",
        )

        history_df = pd.DataFrame(history)
        history_df.to_csv(output_dir / "history.csv", index=False)

        save_training_plots(history, output_dir)
        save_confusion_matrix_plot(final_val_metrics["confusion_matrix"], output_dir)

        metrics = {
            "best_epoch": best_epoch,
            "best_metric": best_metric,
            "metric_to_optimize": metric_to_optimize,
            "final_val_loss": final_val_metrics["loss"],
            "final_val_accuracy": final_val_metrics["accuracy"],
            "final_val_balanced_accuracy": final_val_metrics["balanced_accuracy"],
            "final_val_macro_precision": final_val_metrics["macro_precision"],
            "final_val_macro_recall": final_val_metrics["macro_recall"],
            "final_val_macro_f1": final_val_metrics["macro_f1"],
            "final_val_confusion_matrix": final_val_metrics["confusion_matrix"],
        }

        save_json(to_jsonable(metrics), output_dir / "metrics.json")

        if is_mlflow_enabled(tracking_config):
            log_mlflow_metrics(
                {
                    "best_metric": float(best_metric),
                    "final_val_loss": final_val_metrics["loss"],
                    "final_val_accuracy": final_val_metrics["accuracy"],
                    "final_val_balanced_accuracy": final_val_metrics[
                        "balanced_accuracy"
                    ],
                    "final_val_macro_precision": final_val_metrics["macro_precision"],
                    "final_val_macro_recall": final_val_metrics["macro_recall"],
                    "final_val_macro_f1": final_val_metrics["macro_f1"],
                },
                step=num_epochs,
            )

            if bool(tracking_config.get("log_artifacts", True)):
                log_mlflow_artifacts(output_dir)

    print("Training finished.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best {metric_to_optimize}: {best_metric:.4f}")
    print(f"Saved best checkpoint to: {output_dir / 'best.pt'}")


def main(config: str = "configs/train/cnn.yaml") -> None:
    run_classifier_training(config_path=config)


if __name__ == "__main__":
    main()
