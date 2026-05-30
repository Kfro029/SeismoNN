# Data metadata

This directory stores lightweight data metadata and DVC pointer files.

Expected tracked files include:

- `metadata.example.csv` — small example metadata file.
- `metadata.csv.dvc` — DVC pointer for the default metadata file.
- `.gitignore` — ignores local materialized metadata and generated files.

Large datasets are not stored in Git. Local dataset directories such as
`2nd_selection/` and `all_selection/` are ignored and should be restored via DVC
or downloaded from Hugging Face.

Download and data-availability logic lives in:

- `seismonn/data/download.py` — reusable package code.
- `scripts/download_data.py` — script wrapper.
- `uv run seismonn download-data` — main CLI command.
