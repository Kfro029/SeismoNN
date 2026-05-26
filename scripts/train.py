from __future__ import annotations

import fire

from seismonn.training.train_classifier import run_classifier_training


def main(config: str = "configs/train/cnn.yaml") -> None:
    run_classifier_training(config_path=config)


if __name__ == "__main__":
    fire.Fire(main)
