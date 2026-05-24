from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import lightning as L
import pandas as pd
from hydra.utils import to_absolute_path
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, MLFlowLogger
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf

from seismonn.lightning.classifier import SeismoClassifierLightningModule
from seismonn.lightning.datamodule import SeismoDataModule


def config_to_container(config: DictConfig) -> dict[str, Any]:
    """Convert Hydra config to a plain dictionary."""
    return OmegaConf.to_container(
        config,
        resolve=True,
        throw_on_missing=True,
    )


def get_git_commit_id() -> str:
    """Return current git commit id or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"

    return result.stdout.strip()


def resolve_tracking_uri(tracking_uri: str) -> str:
    """Resolve local MLflow tracking URI.

    Remote URIs such as http://127.0.0.1:8080 are left unchanged.
    Local relative paths are converted to absolute paths.
    """
    if tracking_uri.startswith(("http://", "https://", "file://")):
        return tracking_uri

    return str(Path(to_absolute_path(tracking_uri)).resolve())


def create_loggers(
    config: DictConfig,
    output_dir: Path,
) -> list[Any]:
    """Create Lightning loggers."""
    loggers: list[Any] = [
        CSVLogger(
            save_dir=str(output_dir),
            name="csv_logs",
        )
    ]

    if bool(config.tracking.enabled):
        mlflow_logger = MLFlowLogger(
            experiment_name=str(config.tracking.experiment_name),
            run_name=str(config.tracking.run_name),
            tracking_uri=resolve_tracking_uri(str(config.tracking.tracking_uri)),
            log_model=bool(config.tracking.log_model),
        )

        git_commit = get_git_commit_id()
        mlflow_logger.experiment.set_tag(
            mlflow_logger.run_id,
            "git_commit",
            git_commit,
        )
        mlflow_logger.experiment.set_tag(
            mlflow_logger.run_id,
            "project",
            "SeismoNN",
        )
        mlflow_logger.experiment.set_tag(
            mlflow_logger.run_id,
            "stage",
            "lightning_training",
        )

        loggers.append(mlflow_logger)

    return loggers


def create_callbacks(config: DictConfig, output_dir: Path) -> list[Any]:
    """Create Lightning callbacks."""
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="best-{epoch:03d}-{val_macro_f1:.4f}",
        monitor=str(config.trainer.metric_to_monitor),
        mode=str(config.trainer.monitor_mode),
        save_top_k=1,
        save_last=True,
    )

    return [
        checkpoint_callback,
        LearningRateMonitor(logging_interval="epoch"),
    ]


def plot_metric(
    metrics: pd.DataFrame,
    metric_name: str,
    output_path: Path,
) -> None:
    """Plot one metric from Lightning CSV logs."""
    if metric_name not in metrics.columns:
        return

    metric_data = metrics[["epoch", metric_name]].dropna()

    if metric_data.empty:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(metric_data["epoch"], metric_data[metric_name])
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.title(metric_name)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_lightning_plots(
    csv_log_dir: Path,
    plots_dir: Path,
) -> None:
    """Save at least three training plots from CSV logger metrics."""
    metrics_path = csv_log_dir / "metrics.csv"

    if not metrics_path.exists():
        return

    metrics = pd.read_csv(metrics_path)

    plot_metric(
        metrics=metrics,
        metric_name="train_loss",
        output_path=plots_dir / "train_loss.png",
    )
    plot_metric(
        metrics=metrics,
        metric_name="val_loss",
        output_path=plots_dir / "val_loss.png",
    )
    plot_metric(
        metrics=metrics,
        metric_name="val_accuracy",
        output_path=plots_dir / "val_accuracy.png",
    )
    plot_metric(
        metrics=metrics,
        metric_name="val_macro_f1",
        output_path=plots_dir / "val_macro_f1.png",
    )


def run_lightning_training(config: DictConfig) -> None:
    """Run Hydra-configured PyTorch Lightning training."""
    L.seed_everything(int(config.seed), workers=True)

    output_dir = Path(to_absolute_path(str(config.output_dir)))
    plots_dir = Path(to_absolute_path(str(config.plots_dir)))

    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    data_module = SeismoDataModule(
        metadata_path=to_absolute_path(str(config.data.metadata_path)),
        data_root=to_absolute_path(str(config.data.data_root)),
        train_split=str(config.data.train_split),
        val_split=str(config.data.val_split),
        batch_size=int(config.data.batch_size),
        num_workers=int(config.data.num_workers),
        normalize=bool(config.data.normalize),
        pin_memory=bool(config.data.pin_memory),
    )

    model = SeismoClassifierLightningModule(
        model_config=dict(config_to_container(config.model)),
        optimizer_config=dict(config_to_container(config.optimizer)),
    )

    loggers = create_loggers(
        config=config,
        output_dir=output_dir,
    )

    callbacks = create_callbacks(
        config=config,
        output_dir=output_dir,
    )

    trainer = L.Trainer(
        max_epochs=int(config.trainer.max_epochs),
        accelerator=str(config.trainer.accelerator),
        devices=int(config.trainer.devices),
        precision=str(config.trainer.precision),
        log_every_n_steps=int(config.trainer.log_every_n_steps),
        enable_checkpointing=bool(config.trainer.enable_checkpointing),
        logger=loggers,
        callbacks=callbacks,
    )

    trainer.fit(
        model=model,
        datamodule=data_module,
    )

    csv_logger = next(logger for logger in loggers if isinstance(logger, CSVLogger))

    save_lightning_plots(
        csv_log_dir=Path(csv_logger.log_dir),
        plots_dir=plots_dir,
    )

    print(f"Lightning training finished. Output dir: {output_dir}")
    print(f"Plots dir: {plots_dir}")
