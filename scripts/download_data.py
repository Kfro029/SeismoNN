import json
from typing import Any

import fire

from seismonn.config import load_hydra_config, parse_overrides
from seismonn.data.download import ensure_data_available
from seismonn.training.utils import to_jsonable


def print_json(data: dict[str, Any]) -> None:
    """Print JSON to stdout."""
    print(json.dumps(to_jsonable(data), indent=2, ensure_ascii=False))


def main(overrides: str = "", config_name: str = "config") -> None:
    """Ensure SeismoNN data is available using Hydra config.

    Example:
        uv run python scripts/download_data.py --overrides "data=full"
    """
    config = load_hydra_config(
        overrides=parse_overrides(overrides),
        config_name=config_name,
    )

    ensure_data_available(
        metadata_path=str(config.data.metadata_path),
        data_dir=str(config.data.data_dir),
        repo_root=str(config.data.repo_root),
        use_dvc=bool(config.data.use_dvc),
        dvc_remote=str(config.data.dvc_remote),
        huggingface_repo_id=str(config.data.huggingface_repo_id),
        allow_huggingface_fallback=bool(config.data.allow_huggingface_fallback),
    )

    print_json(
        {
            "status": "ok",
            "metadata_path": str(config.data.metadata_path),
            "data_dir": str(config.data.data_dir),
            "repo_root": str(config.data.repo_root),
            "use_dvc": bool(config.data.use_dvc),
            "dvc_remote": str(config.data.dvc_remote),
            "huggingface_repo_id": str(config.data.huggingface_repo_id),
        }
    )


if __name__ == "__main__":
    fire.Fire(main)
