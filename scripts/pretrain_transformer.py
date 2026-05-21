from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from seismonn.data.dataset import SeismoDataset
from seismonn.models.masked_transformer import MaskedTraceReconstructionTransformer
from seismonn.tracking.mlflow import (
    is_mlflow_enabled,
    log_mlflow_artifacts,
    log_mlflow_metrics,
    log_mlflow_params,
    start_mlflow_run,
)
from seismonn.training.pretraining import (
    evaluate_pretraining,
    get_pretraining_metric,
    pretrain_one_epoch,
)
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


def build_optimizer(
    model: torch.nn.Module,
    optimizer_config: dict[str, Any],
) -> torch.optim.Optimizer:
    optimizer_name = optimizer_config.get("name", "adamw").lower()
    lr = float(optimizer_config.get("lr", 3e-4))
    weight_decay = float(optimizer_config.get("weight_decay", 0.0))

    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def create_pretraining_model(
    model_config: dict[str, Any],
) -> MaskedTraceReconstructionTransformer:
    model_name = str(model_config.get("name", "masked_trace_transformer")).lower()

    if model_name != "masked_trace_transformer":
        raise ValueError(f"Unsupported pretraining model: {model_name}")

    return MaskedTraceReconstructionTransformer(
        in_channels=int(model_config.get("in_channels", 2)),
        time_steps=int(model_config["time_steps"]),
        d_model=int(model_config.get("d_model", 64)),
        nhead=int(model_config.get("nhead", 4)),
        num_layers=int(model_config.get("num_layers", 1)),
        dim_feedforward=int(model_config.get("dim_feedforward", 128)),
        dropout=float(model_config.get("dropout", 0.1)),
        temporal_hidden_channels=int(model_config.get("temporal_hidden_channels", 16)),
        max_receivers=int(model_config.get("max_receivers", 1024)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Self-supervised pre-training for Trace Transformer."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pretrain/trace_transformer.yaml"),
        help="Path to pretraining config.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    config = load_config(args.config)

    set_seed(int(config["seed"]))
    device = get_device(str(config["device"]))

    data_config = config["data"]
    model_config = config["model"]
    optimizer_config = config["optimizer"]
    pretraining_config = config["pretraining"]
    tracking_config = config.get("tracking", {})

    output_dir = resolve_path(project_root, pretraining_config["output_dir"])
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

    model = create_pretraining_model(model_config).to(device)
    optimizer = build_optimizer(model, optimizer_config)

    num_epochs = int(pretraining_config["num_epochs"])
    mask_ratio = float(pretraining_config["mask_ratio"])
    metric_to_optimize = str(pretraining_config["metric_to_optimize"])

    best_metric = float("inf")
    best_epoch = -1
    history: list[dict[str, Any]] = []

    print(f"Device: {device}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Mask ratio: {mask_ratio}")
    print(f"Output dir: {output_dir}")

    with start_mlflow_run(project_root=project_root, tracking_config=tracking_config):
        if is_mlflow_enabled(tracking_config):
            log_mlflow_params(config)

        for epoch in range(1, num_epochs + 1):
            train_metrics = pretrain_one_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                device=device,
                mask_ratio=mask_ratio,
            )

            val_metrics = evaluate_pretraining(
                model=model,
                dataloader=val_loader,
                device=device,
                mask_ratio=mask_ratio,
            )

            current_metric = get_pretraining_metric(
                metric_name=metric_to_optimize,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
            )

            row = {
                "epoch": epoch,
                "train_masked_mse": train_metrics["masked_mse"],
                "val_masked_mse": val_metrics["masked_mse"],
            }
            history.append(row)

            print(
                f"Epoch {epoch:03d}/{num_epochs:03d} | "
                f"train_masked_mse={row['train_masked_mse']:.6f} | "
                f"val_masked_mse={row['val_masked_mse']:.6f}"
            )

            if is_mlflow_enabled(tracking_config):
                log_mlflow_metrics(
                    {
                        "train_masked_mse": row["train_masked_mse"],
                        "val_masked_mse": row["val_masked_mse"],
                    },
                    step=epoch,
                )

            if current_metric < best_metric:
                best_metric = current_metric
                best_epoch = epoch

                checkpoint = {
                    "model_name": model_config["name"],
                    "model_state_dict": model.state_dict(),
                    "model_config": model_config,
                    "data_config": data_config,
                    "epoch": epoch,
                    "metric_to_optimize": metric_to_optimize,
                    "best_metric": best_metric,
                    "input_shape": data_config["input_shape"],
                    "pretraining": {
                        "mask_ratio": mask_ratio,
                        "task": "masked_trace_reconstruction",
                    },
                    "val_metrics": val_metrics,
                }

                torch.save(checkpoint, output_dir / "best.pt")

        torch.save(
            {
                "model_name": model_config["name"],
                "model_state_dict": model.state_dict(),
                "model_config": model_config,
                "data_config": data_config,
                "epoch": num_epochs,
                "input_shape": data_config["input_shape"],
                "pretraining": {
                    "mask_ratio": mask_ratio,
                    "task": "masked_trace_reconstruction",
                },
            },
            output_dir / "last.pt",
        )

        history_df = pd.DataFrame(history)
        history_df.to_csv(output_dir / "history.csv", index=False)

        metrics = {
            "best_epoch": best_epoch,
            "best_metric": best_metric,
            "metric_to_optimize": metric_to_optimize,
            "final_train_masked_mse": history[-1]["train_masked_mse"],
            "final_val_masked_mse": history[-1]["val_masked_mse"],
            "mask_ratio": mask_ratio,
        }

        save_json(to_jsonable(metrics), output_dir / "metrics.json")

        if is_mlflow_enabled(tracking_config):
            log_mlflow_metrics(
                {
                    "best_metric": float(best_metric),
                    "final_train_masked_mse": history[-1]["train_masked_mse"],
                    "final_val_masked_mse": history[-1]["val_masked_mse"],
                },
                step=num_epochs,
            )

            if bool(tracking_config.get("log_artifacts", True)):
                log_mlflow_artifacts(output_dir)

    print("Pre-training finished.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best {metric_to_optimize}: {best_metric:.6f}")
    print(f"Saved best checkpoint to: {output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
