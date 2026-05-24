from __future__ import annotations

import subprocess
import sys
from pathlib import Path


DEFAULT_HF_DATASET_REPO_ID = "FAKIrik/Seismo_datasets"


def run_command(command: list[str], cwd: str | Path = ".") -> None:
    """Run command and raise readable RuntimeError on failure."""
    try:
        subprocess.run(command, cwd=cwd, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Command {command[0]!r} was not found. Install required dependency first."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Command failed with exit code {exc.returncode}: {' '.join(command)}"
        ) from exc


def dvc_pull(
    targets: list[str] | None = None,
    remote: str | None = None,
    repo_root: str | Path = ".",
) -> None:
    """Pull artifacts with DVC."""
    command = ["dvc", "pull"]

    if remote is not None:
        command.extend(["-r", remote])

    if targets:
        command.extend(targets)

    run_command(command, cwd=repo_root)


def download_from_huggingface(
    repo_id: str = DEFAULT_HF_DATASET_REPO_ID,
    local_dir: str | Path = ".",
) -> None:
    """Fallback download from Hugging Face dataset repository."""
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
    )


def build_metadata_if_possible(repo_root: str | Path = ".") -> None:
    """Build metadata.csv if raw split/data files are available."""
    repo_root = Path(repo_root)

    metadata_path = repo_root / "data" / "metadata.csv"
    split_json_path = repo_root / "2nd_sel.json"
    data_dir = repo_root / "2nd_selection"
    build_script_path = repo_root / "scripts" / "build_metadata.py"

    if metadata_path.exists():
        return

    if (
        not split_json_path.exists()
        or not data_dir.exists()
        or not build_script_path.exists()
    ):
        return

    run_command(
        [
            sys.executable,
            str(build_script_path),
            "--split-json",
            str(split_json_path),
            "--data-dir",
            str(data_dir),
            "--output",
            str(metadata_path),
            "--test-split-name",
            "val",
            "--validate-files",
        ],
        cwd=repo_root,
    )


def ensure_data_available(
    metadata_path: str | Path = "data/metadata.csv",
    data_dir: str | Path = "2nd_selection",
    repo_root: str | Path = ".",
    use_dvc: bool = True,
    dvc_remote: str = "data_storage",
    huggingface_repo_id: str = DEFAULT_HF_DATASET_REPO_ID,
    allow_huggingface_fallback: bool = True,
) -> None:
    """Ensure metadata and raw data directory are available.

    Order:
    1. If metadata and data_dir exist, do nothing.
    2. Try DVC pull.
    3. Try Hugging Face fallback.
    4. Try to build metadata.csv from 2nd_sel.json + 2nd_selection.
    """
    repo_root = Path(repo_root)

    full_metadata_path = repo_root / metadata_path
    full_data_dir = repo_root / data_dir

    if full_metadata_path.exists() and full_data_dir.exists():
        return

    if use_dvc:
        try:
            dvc_pull(remote=dvc_remote, repo_root=repo_root)
        except RuntimeError:
            if not allow_huggingface_fallback:
                raise

    if full_metadata_path.exists() and full_data_dir.exists():
        return

    if allow_huggingface_fallback:
        download_from_huggingface(
            repo_id=huggingface_repo_id,
            local_dir=repo_root,
        )
        build_metadata_if_possible(repo_root=repo_root)

    if not full_metadata_path.exists():
        raise FileNotFoundError(f"Metadata file is missing: {full_metadata_path}")

    if not full_data_dir.exists():
        raise FileNotFoundError(f"Data directory is missing: {full_data_dir}")
