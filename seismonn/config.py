import shlex
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


def parse_overrides(overrides: str | list[str] | tuple[str, ...] | None) -> list[str]:
    """Parse Hydra override string/list into a list."""
    if overrides is None:
        return []

    if isinstance(overrides, str):
        if not overrides.strip():
            return []
        return shlex.split(overrides)

    return [str(override) for override in overrides]


def load_hydra_config(
    overrides: str | list[str] | tuple[str, ...] | None = None,
    config_name: str = "config",
) -> DictConfig:
    """Compose project Hydra config outside @hydra.main."""
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(
        version_base="1.3",
        config_dir=str(CONFIG_DIR),
    ):
        return compose(
            config_name=config_name,
            overrides=parse_overrides(overrides),
        )
