# SeismoNN

SeismoNN is an MLOps project for fracture parameter prediction from synthetic seismic wavefield simulations.

The current MVP solves a classification task: predicting the number of fractures in a fractured medium from a seismic response tensor.

## Task

The current version solves the classification problem:

```text
Input:  .npy tensor with shape (2, 1723, 501)
Output: predicted number of fractures: 3, 4, or 5
```

Input tensor format:

```text
2     — velocity components: vx and vy
1723  — time steps
501   — receiver positions
```

The output is a JSON object with the predicted class and class probabilities:

```json
{
  "model_name": "cnn_baseline",
  "predicted_class_id": 1,
  "predicted_crack_count": 4,
  "class_probabilities": {
    "3": 0.12,
    "4": 0.81,
    "5": 0.07
  }
}
```

In future versions, the project may be extended to multi-task prediction:

```text
- fracture count classification
- mean fracture length regression
- fracture length spread regression
- mean fracture angle regression
- fracture angle spread regression
```

## Project structure

```text
SeismoNN/
├── configs/
│   └── train/
│       └── cnn.yaml
├── data/
│   └── metadata.csv
├── scripts/
│   ├── baseline.py
│   ├── build_metadata.py
│   ├── predict.py
│   └── train_cnn.py
├── seismonn/
│   ├── api/
│   │   └── main.py
│   ├── data/
│   │   └── dataset.py
│   ├── inference/
│   │   └── predictor.py
│   ├── models/
│   │   └── cnn.py
│   └── training/
│       ├── evaluate.py
│       └── utils.py
├── tests/
├── Dockerfile
├── .dockerignore
├── pyproject.toml
├── uv.lock
└── README.md
```

## Data

The dataset consists of synthetic seismic simulations stored as `.npy` files.

Each sample is described in `data/metadata.csv`.

Current metadata columns:

```text
sample_id
path
filename
split
crack_count
class_id
cluster_center_x
cluster_center_y
cluster_half_size_x
cluster_half_size_y
mean_length
length_spread
mean_angle_deg
angle_spread_deg
shape
dtype
```

The current split is:

```text
train: 532 samples
val:   133 samples
total: 665 samples
```

Class mapping:

```text
class_id = 0 → crack_count = 3
class_id = 1 → crack_count = 4
class_id = 2 → crack_count = 5
```

The full dataset is not stored directly in git. Large `.npy` files are expected to be stored separately, for example through DVC or an external dataset storage.

## Installation

This project uses `uv` for dependency management.

Install dependencies:

```bash
uv sync
```

Install development dependencies if needed:

```bash
uv sync --all-extras --dev
```

Run all commands from the repository root.

## Build metadata

If `2nd_sel.json` and the `2nd_selection/` directory are available, metadata can be regenerated with:

```bash
uv run python scripts/build_metadata.py \
  --split-json 2nd_sel.json \
  --data-dir 2nd_selection \
  --output data/metadata.csv \
  --test-split-name val \
  --validate-files
```

Check metadata:

```bash
uv run python - <<'PY'
import pandas as pd

df = pd.read_csv("data/metadata.csv")

print(df.head())
print()
print(df.groupby(["split", "crack_count"]).size())
print()
print(df["shape"].value_counts())
PY
```

## Train CNN baseline

Train the CNN baseline using the YAML config:

```bash
uv run python scripts/train_cnn.py --config configs/train/cnn.yaml
```

Training artifacts are saved to:

```text
outputs/cnn_baseline/
```

Expected artifacts:

```text
best.pt
last.pt
history.csv
metrics.json
loss.png
accuracy.png
macro_f1.png
confusion_matrix.png
```

The `outputs/` directory is ignored by git.

## CLI inference

After training, run prediction for a single `.npy` sample:

```bash
uv run python scripts/predict.py \
  --checkpoint outputs/cnn_baseline/best.pt \
  --input 2nd_selection/<sample_name>.npy
```

Save prediction to JSON:

```bash
uv run python scripts/predict.py \
  --checkpoint outputs/cnn_baseline/best.pt \
  --input 2nd_selection/<sample_name>.npy \
  --output outputs/sample_prediction.json
```

To get a sample path from `metadata.csv`:

```bash
uv run python - <<'PY'
import pandas as pd

df = pd.read_csv("data/metadata.csv")
print(df.iloc[0]["path"])
PY
```

## Run FastAPI service locally

The project provides an HTTP inference API.

Start the API:

```bash
SEISMONN_CHECKPOINT=outputs/cnn_baseline/best.pt \
uv run uvicorn seismonn.api.main:app --host 127.0.0.1 --port 8000
```

Healthcheck:

```bash
curl -s http://127.0.0.1:8000/health | python -m json.tool
```

Prediction request:

```bash
curl -s -X POST \
  http://127.0.0.1:8000/predict \
  -F "file=@2nd_selection/<sample_name>.npy" \
  | python -m json.tool
```

API endpoints:

```text
GET  /health   — check service status and model loading status
POST /predict  — upload .npy file and get fracture count prediction
```

## Run API with Docker

Build Docker image:

```bash
docker build -t seismonn-api .
```

Run container with mounted checkpoint:

```bash
docker run --rm \
  -p 8000:8000 \
  -e SEISMONN_CHECKPOINT=/app/checkpoints/best.pt \
  -e SEISMONN_DEVICE=cpu \
  -v "$(pwd)/outputs/cnn_baseline/best.pt:/app/checkpoints/best.pt:ro" \
  seismonn-api
```

In another terminal, check the service:

```bash
curl -s http://127.0.0.1:8000/health | python -m json.tool
```

Send prediction request:

```bash
curl -s -X POST \
  http://127.0.0.1:8000/predict \
  -F "file=@2nd_selection/<sample_name>.npy" \
  | python -m json.tool
```

Run Docker container in detached mode:

```bash
docker run -d --name seismonn-api \
  -p 8000:8000 \
  -e SEISMONN_CHECKPOINT=/app/checkpoints/best.pt \
  -e SEISMONN_DEVICE=cpu \
  -v "$(pwd)/outputs/cnn_baseline/best.pt:/app/checkpoints/best.pt:ro" \
  seismonn-api
```

Stop detached container:

```bash
docker stop seismonn-api
```

## Testing

Run all tests:

```bash
uv run pytest
```

Run selected tests:

```bash
uv run pytest \
  tests/test_dataset.py \
  tests/test_cnn.py \
  tests/test_evaluate.py \
  tests/test_predictor.py \
  tests/test_api.py
```

## Code quality

Run pre-commit checks:

```bash
uv run pre-commit run --all-files
```

If pre-commit automatically modifies files, stage them and run checks again:

```bash
git add .
uv run pre-commit run --all-files
```

## Current model

The current baseline model is a CNN classifier:

```text
Conv2d → ReLU → MaxPool2d
Conv2d → ReLU → MaxPool2d
Conv2d → ReLU → MaxPool2d
AdaptiveAvgPool2d
Linear → ReLU → Dropout → Linear
```

The model predicts logits for three classes:

```text
3 fractures
4 fractures
5 fractures
```

## Metrics

For classification, the project tracks:

```text
Accuracy
Macro-F1
Confusion matrix
Validation loss
```

Training script saves:

```text
history.csv
metrics.json
confusion_matrix.png
accuracy.png
loss.png
macro_f1.png
```

## MLOps components

Implemented components:

```text
Metadata-based dataset description
Reusable PyTorch Dataset
CNN baseline model
YAML training config
Training script
Validation metrics
Best checkpoint saving
CLI inference
FastAPI inference service
Docker deployment
Pytest tests
Pre-commit checks
```

Planned components:

```text
MLflow experiment tracking
Stratified/grouped split generation
Transformer encoder model
Self-supervised pre-training
CI workflow improvements
```

## Notes

The current version is an MVP. The main goal is to provide a reproducible ML pipeline:

```text
metadata.csv
→ Dataset
→ training config
→ CNN training
→ best checkpoint
→ CLI inference
→ FastAPI inference
→ Docker deployment
```

The current classification task is used as the first stable baseline. More complex regression and multi-task formulations can be added later.
