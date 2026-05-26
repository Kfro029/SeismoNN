from __future__ import annotations

from pathlib import Path
from typing import Any

import fire
import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from seismonn.data.multitask_dataset import (
    SeismoMultiTaskDataset,
    fit_regression_target_scaler_from_metadata,
)
from seismonn.models.cnn_multitask import SeismoCNNMultiTask
from seismonn.tracking.mlflow import (
    is_mlflow_enabled,
    log_mlflow_artifacts,
    log_mlflow_metrics,
    log_mlflow_params,
    start_mlflow_run,
)
from seismonn.training.multitask import (
    evaluate_multitask,
    train_multitask_one_epoch,
)
from seismonn.training.train_classifier import build_optimizer
from seismonn.training.utils import (
    get_device,
    resolve_path,
    save_json,
    set_seed,
    to_jsonable,
)


def load_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Config must be a dictionary, got: {type(config)}")

    return config


def create_multitask_model(model_config: dict[str, Any]) -> nn.Module:
    model_name = str(model_config.get("name", "cnn_multitask")).lower()

    if model_name != "cnn_multitask":
        raise ValueError(f"Unsupported multi-task model: {model_name}")

    return SeismoCNNMultiTask(
        in_channels=int(model_config.get("in_channels", 2)),
        num_classes=int(model_config.get("num_classes", 3)),
        num_regression_targets=int(model_config.get("num_regression_targets", 4)),
        dropout=float(model_config.get("dropout", 0.2)),
    )


def get_optimized_metric(metric_name: str, val_metrics: dict[str, Any]) -> float:
    if metric_name == "val_classification_macro_f1":
        return float(val_metrics["classification_macro_f1"])

    if metric_name == "val_classification_accuracy":
        return float(val_metrics["classification_accuracy"])

    if metric_name == "val_regression_mae_mean":
        return -float(val_metrics["regression_mae_mean"])

    if metric_name == "val_regression_rmse_mean":
        return -float(val_metrics["regression_rmse_mean"])

    raise ValueError(f"Unsupported metric_to_optimize: {metric_name}")


def save_training_plots(history: list[dict[str, Any]], output_dir: Path) -> None:
    history_df = pd.DataFrame(history)

    plt.figure()
    plt.plot(history_df["epoch"], history_df["train_loss"], label="train")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Total loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss.png")
    plt.close()

    plt.figure()
    plt.plot(
        history_df["epoch"],
        history_df["train_classification_accuracy"],
        label="train",
    )
    plt.plot(
        history_df["epoch"],
        history_df["val_classification_accuracy"],
        label="val",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Classification accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "classification_accuracy.png")
    plt.close()

    plt.figure()
    plt.plot(
        history_df["epoch"],
        history_df["val_regression_mae_mean"],
        label="val_regression_mae_mean",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Mean MAE in original target units")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "regression_mae.png")
    plt.close()


def run_multitask_training(
    config_path: str | Path = "configs/train/cnn_multitask.yaml",
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(config_path)

    set_seed(int(config["seed"]))
    device = get_device(str(config["device"]))

    data_config = config["data"]
    model_config = config["model"]
    optimizer_config = config["optimizer"]
    training_config = config["training"]
    tracking_config = config.get("tracking", {})
    targets_config = config["targets"]
    loss_config = config["loss"]

    output_dir = resolve_path(project_root, training_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = resolve_path(project_root, data_config["metadata_path"])
    data_root = resolve_path(project_root, data_config["data_root"])

    regression_columns = list(targets_config["regression_columns"])

    target_scaler = fit_regression_target_scaler_from_metadata(
        metadata_path=metadata_path,
        train_split=data_config["train_split"],
        target_columns=regression_columns,
    )

    train_dataset = SeismoMultiTaskDataset(
        metadata_path=metadata_path,
        split=data_config["train_split"],
        data_root=data_root,
        regression_target_columns=regression_columns,
        target_scaler=target_scaler,
        normalize_input=bool(data_config["normalize"]),
    )

    val_dataset = SeismoMultiTaskDataset(
        metadata_path=metadata_path,
        split=data_config["val_split"],
        data_root=data_root,
        regression_target_columns=regression_columns,
        target_scaler=target_scaler,
        normalize_input=bool(data_config["normalize"]),
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

    model = create_multitask_model(model_config).to(device)

    classification_criterion = nn.CrossEntropyLoss()
    regression_criterion = nn.MSELoss()
    optimizer = build_optimizer(model, optimizer_config)

    num_epochs = int(training_config["num_epochs"])
    metric_to_optimize = str(training_config["metric_to_optimize"])
    regression_loss_weight = float(loss_config["regression_loss_weight"])

    best_metric = float("-inf")
    best_epoch = -1
    history: list[dict[str, Any]] = []

    labels = list(range(int(model_config["num_classes"])))

    print(f"Device: {device}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Regression columns: {regression_columns}")
    print(f"Output dir: {output_dir}")

    with start_mlflow_run(project_root=project_root, tracking_config=tracking_config):
        if is_mlflow_enabled(tracking_config):
            log_mlflow_params(config)

        for epoch in range(1, num_epochs + 1):
            train_metrics = train_multitask_one_epoch(
                model=model,
                dataloader=train_loader,
                classification_criterion=classification_criterion,
                regression_criterion=regression_criterion,
                optimizer=optimizer,
                device=device,
                regression_loss_weight=regression_loss_weight,
            )

            val_metrics = evaluate_multitask(
                model=model,
                dataloader=val_loader,
                classification_criterion=classification_criterion,
                regression_criterion=regression_criterion,
                device=device,
                target_scaler=target_scaler,
                regression_loss_weight=regression_loss_weight,
                labels=labels,
            )

            current_metric = get_optimized_metric(
                metric_name=metric_to_optimize,
                val_metrics=val_metrics,
            )

            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_classification_loss": train_metrics["classification_loss"],
                "train_regression_loss": train_metrics["regression_loss"],
                "train_classification_accuracy": train_metrics[
                    "classification_accuracy"
                ],
                "val_loss": val_metrics["loss"],
                "val_classification_loss": val_metrics["classification_loss"],
                "val_regression_loss": val_metrics["regression_loss"],
                "val_classification_accuracy": val_metrics["classification_accuracy"],
                "val_classification_macro_f1": val_metrics["classification_macro_f1"],
                "val_regression_mae_mean": val_metrics["regression_mae_mean"],
                "val_regression_rmse_mean": val_metrics["regression_rmse_mean"],
            }
            history.append(row)

            print(
                f"Epoch {epoch:03d}/{num_epochs:03d} | "
                f"train_loss={row['train_loss']:.4f} | "
                f"train_acc={row['train_classification_accuracy']:.4f} | "
                f"val_loss={row['val_loss']:.4f} | "
                f"val_acc={row['val_classification_accuracy']:.4f} | "
                f"val_f1={row['val_classification_macro_f1']:.4f} | "
                f"val_mae={row['val_regression_mae_mean']:.4f} | "
                f"val_rmse={row['val_regression_rmse_mean']:.4f}"
            )

            if is_mlflow_enabled(tracking_config):
                log_mlflow_metrics(row, step=epoch)

            if current_metric > best_metric:
                best_metric = current_metric
                best_epoch = epoch

                checkpoint = {
                    "model_name": model_config["name"],
                    "model_state_dict": model.state_dict(),
                    "model_config": model_config,
                    "data_config": data_config,
                    "target_scaler": target_scaler.to_dict(),
                    "regression_columns": regression_columns,
                    "epoch": epoch,
                    "metric_to_optimize": metric_to_optimize,
                    "best_metric": best_metric,
                    "input_shape": data_config["input_shape"],
                    "class_id_to_crack_count": config["checkpoint"][
                        "class_id_to_crack_count"
                    ],
                    "val_metrics": val_metrics,
                }

                torch.save(checkpoint, output_dir / "best.pt")

        torch.save(
            {
                "model_name": model_config["name"],
                "model_state_dict": model.state_dict(),
                "model_config": model_config,
                "data_config": data_config,
                "target_scaler": target_scaler.to_dict(),
                "regression_columns": regression_columns,
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

        final_val_metrics = evaluate_multitask(
            model=model,
            dataloader=val_loader,
            classification_criterion=classification_criterion,
            regression_criterion=regression_criterion,
            device=device,
            target_scaler=target_scaler,
            regression_loss_weight=regression_loss_weight,
            labels=labels,
        )

        metrics = {
            "best_epoch": best_epoch,
            "best_metric": best_metric,
            "metric_to_optimize": metric_to_optimize,
            "final_val_metrics": final_val_metrics,
            "target_scaler": target_scaler.to_dict(),
        }

        save_json(to_jsonable(metrics), output_dir / "metrics.json")
        save_training_plots(history, output_dir)

        if is_mlflow_enabled(tracking_config):
            log_mlflow_metrics(
                {
                    "best_metric": float(best_metric),
                    "final_val_classification_accuracy": final_val_metrics[
                        "classification_accuracy"
                    ],
                    "final_val_classification_macro_f1": final_val_metrics[
                        "classification_macro_f1"
                    ],
                    "final_val_regression_mae_mean": final_val_metrics[
                        "regression_mae_mean"
                    ],
                    "final_val_regression_rmse_mean": final_val_metrics[
                        "regression_rmse_mean"
                    ],
                },
                step=num_epochs,
            )

            if bool(tracking_config.get("log_artifacts", True)):
                log_mlflow_artifacts(output_dir)

    print("Multi-task training finished.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best {metric_to_optimize}: {best_metric:.4f}")
    print(f"Saved best checkpoint to: {output_dir / 'best.pt'}")


def main(config: str = "configs/train/cnn_multitask.yaml") -> None:
    run_multitask_training(config_path=config)


if __name__ == "__main__":
    fire.Fire(main)
