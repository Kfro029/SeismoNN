from __future__ import annotations

import hydra
from omegaconf import DictConfig

from seismonn.lightning.train import run_lightning_training


@hydra.main(
    version_base="1.3",
    config_path="../configs/hydra",
    config_name="config",
)
def main(config: DictConfig) -> None:
    run_lightning_training(config)


if __name__ == "__main__":
    main()
