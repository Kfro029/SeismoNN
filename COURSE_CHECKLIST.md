# COURSE_CHECKLIST.md

Этот файл показывает, как проект SeismoNN закрывает требования курса.

## 1. Быстрая проверка проекта

Клонирование репозитория:

    git clone <repo-url>
    cd SeismoNN

Установка зависимостей:

    uv sync --all-extras --dev

Установка pre-commit hooks:

    uv run pre-commit install

Проверка pre-commit:

    uv run pre-commit run -a

Запуск тестов:

    uv run pytest

Короткий запуск основного training pipeline для проверки задания:

    uv run seismonn train --overrides "trainer.max_epochs=1 tracking.enabled=false data.ensure_data=false"

Проверка основной CLI-точки входа:

    uv run seismonn --help

Проверка валидации данных:

    uv run seismonn validate-data --validate_files=False

## 2. Таблица соответствия требованиям

| Требование | Реализация в проекте | Команда / файл |
|---|---|---|
| Открытый репозиторий | Код находится в GitHub-репозитории, основная ветка `master` | `git log --oneline -5` |
| Python package | Основной пакет проекта — `seismonn` | `uv run python -c "import seismonn"` |
| Зависимости через uv | Зависимости описаны в `pyproject.toml`, lock-файл — `uv.lock` | `uv sync --all-extras --dev` |
| Code quality tools | Используется `pre-commit`, `ruff`, `ruff-format`, базовые хуки | `.pre-commit-config.yaml`, `uv run pre-commit run -a` |
| Training framework | Baseline training для проверки задания реализован через PyTorch Lightning | `seismonn/lightning/`, `scripts/train_lightning.py` |
| Hydra configs | Основной Lightning pipeline использует Hydra-конфиги с `defaults` | `configs/hydra/config.yaml` |
| DVC data management | Данные и модельные артефакты отслеживаются DVC | `.dvc/config`, `data/metadata.csv.dvc`, `models/**/*.dvc` |
| DVC remotes | Разные remotes для данных и моделей | `data_storage`, `model_storage` в `.dvc/config` |
| download_data | Реализована проверка/загрузка данных через DVC и fallback на Hugging Face | `seismonn/data/download.py` |
| Train проверяет данные | Lightning/Hydra train вызывает `ensure_data_available()` | `seismonn/lightning/train.py` |
| Logging | Используется MLflow tracking | `seismonn/tracking/mlflow.py`, `configs/hydra/tracking/mlflow.yaml` |
| MLflow tracking URI | В Hydra-конфиге указан `http://127.0.0.1:8080` | `configs/hydra/tracking/mlflow.yaml` |
| Графики обучения | Графики сохраняются в `plots/` | `plots/lightning/cnn_baseline/` |
| Git commit id | Git commit id логируется в Lightning/MLflow pipeline | `seismonn/lightning/train.py` |
| CLI без argparse | Основная CLI-точка входа реализована через Fire | `seismonn/commands.py`, `uv run seismonn --help` |
| Train-команда для проверки | Запуск через Fire CLI + Hydra + Lightning | `uv run seismonn train --overrides "trainer.max_epochs=1"` |
| Infer CLI | Предсказание через основную Fire CLI-точку | `uv run seismonn predict --checkpoint ... --input_path ...` |
| ONNX export | Реализован экспорт checkpoint в ONNX + ONNX Runtime smoke test | `seismonn/exporting/onnx.py`, `scripts/export_onnx.py` |
| TensorRT export | Реализован wrapper над `trtexec`; есть dry-run | `seismonn/exporting/tensorrt.py`, `scripts/export_tensorrt.py` |
| Inference server | Реализован MLflow PyFunc + MLflow Serving | `seismonn/serving/mlflow_model.py`, `scripts/save_mlflow_model.py` |
| FastAPI service | Дополнительный HTTP inference service | `seismonn/api/main.py` |
| Docker | Есть Dockerfile для API | `Dockerfile`, `docker build -t seismonn-api .` |
| Tests | Unit/integration tests через pytest | `uv run pytest` |
| CI | GitHub Actions: pre-commit, pytest, Docker build | `.github/workflows/ci.yaml` |
| Dataset docs | Отдельное описание датасета | `DATASET.md` |
| Project spec | Обновлённая спецификация проекта | `PROJECT.md` |
| Results report | Сводка результатов экспериментов | `RESULTS.md` |

## 3. Setup

Установка зависимостей:

    uv sync --all-extras --dev

Установка git hooks:

    uv run pre-commit install

Проверка hooks:

    uv run pre-commit run -a

Проверка тестов:

    uv run pytest

## 4. Data management

Большие данные и модельные артефакты не хранятся в git. Для них используется DVC.

В проекте настроены два DVC remote:

    data_storage   — для датасета и metadata.csv
    model_storage  — для модельных checkpoint-ов

Получить данные через DVC:

    uv run dvc pull -r data_storage

Получить модельные артефакты через DVC:

    uv run dvc pull -r model_storage

Если DVC remote недоступен, в проекте есть fallback-загрузка из Hugging Face:

    uv run seismonn download-data

Проверить metadata без чтения всех `.npy` файлов:

    uv run seismonn validate-data --validate_files=False

Проверить metadata и реальные `.npy` файлы:

    uv run seismonn validate-data --validate_files=True

Файл с логикой загрузки данных:

    seismonn/data/download.py

Hydra-конфиг данных:

    configs/hydra/data/seismonn.yaml

## 5. Train

Baseline training pipeline для проверки задания реализован через Hydra + PyTorch Lightning.

Короткий запуск без MLflow-сервера:

    uv run seismonn train --overrides "trainer.max_epochs=1 tracking.enabled=false data.ensure_data=false"

Запуск MLflow tracking server на адресе из ТЗ:

    uv run mlflow server \
      --host 127.0.0.1 \
      --port 8080 \
      --backend-store-uri ./mlruns \
      --default-artifact-root ./mlartifacts

После этого обучение можно запустить так:

    uv run seismonn train --overrides "trainer.max_epochs=5"

Hydra-конфиги:

    configs/hydra/config.yaml
    configs/hydra/data/seismonn.yaml
    configs/hydra/model/cnn.yaml
    configs/hydra/optimizer/adamw.yaml
    configs/hydra/trainer/default.yaml
    configs/hydra/tracking/mlflow.yaml

Главный Hydra-конфиг содержит defaults:

    defaults:
      - data: seismonn
      - model: cnn
      - optimizer: adamw
      - trainer: default
      - tracking: mlflow
      - _self_

Артефакты Lightning training:

    outputs/lightning/cnn_baseline/
    plots/lightning/cnn_baseline/

Графики обучения:

    plots/lightning/cnn_baseline/train_loss.png
    plots/lightning/cnn_baseline/val_loss.png
    plots/lightning/cnn_baseline/val_accuracy.png
    plots/lightning/cnn_baseline/val_macro_f1.png

## 6. Дополнительные training pipelines

Помимо основного Lightning/Hydra baseline, в проекте есть исследовательские пайплайны:

CNN baseline:

    uv run python scripts/train.py --config configs/train/cnn.yaml

Supervised Trace Transformer:

    uv run python scripts/train.py --config configs/train/transformer.yaml

Self-supervised masked trace pre-training:

    uv run python scripts/pretrain_transformer.py \
      --config configs/pretrain/trace_transformer.yaml

Fine-tuning Transformer после pre-training:

    uv run python scripts/train.py --config configs/train/transformer_finetune.yaml

CNN multi-task baseline:

    uv run python scripts/train_multitask.py \
      --config configs/train/cnn_multitask.yaml

## 7. Infer

Рекомендуемый CLI inference через Fire CLI:

    uv run seismonn predict \
      --checkpoint outputs/cnn_multitask_50ep/best.pt \
      --input_path 2nd_selection/<sample_name>.npy \
      --device cpu \
      --predictor_type auto \
      --output outputs/seismonn_cli_prediction.json

Формат входа:

    .npy tensor
    shape: (2, 1723, 501)
    dtype: float32

Формат выхода для multi-task модели:

    {
      "model_name": "cnn_multitask",
      "predicted_class_id": 2,
      "predicted_crack_count": 5,
      "expected_crack_count": 4.18,
      "class_probabilities": {
        "3": 0.22,
        "4": 0.36,
        "5": 0.41
      },
      "regression": {
        "mean_length": 32.5,
        "length_spread": 3.9,
        "mean_angle_deg": 9.0,
        "angle_spread_deg": 8.9
      }
    }

## 8. Production preparation

### 8.1 ONNX

Экспорт classification checkpoint в ONNX:

    uv run seismonn export-onnx \
      --checkpoint outputs/cnn_baseline/best.pt \
      --output outputs/cnn_baseline/model.onnx \
      --device cpu

Экспорт multi-task checkpoint в ONNX:

    uv run seismonn export-onnx \
      --checkpoint outputs/cnn_multitask_50ep/best.pt \
      --output outputs/cnn_multitask_50ep/model.onnx \
      --device cpu

ONNX export реализован в:

    seismonn/exporting/onnx.py

ONNX Runtime smoke test включён в export pipeline.

### 8.2 TensorRT

Dry-run TensorRT export:

    uv run seismonn export-tensorrt \
      --onnx outputs/cnn_baseline/model.onnx \
      --engine outputs/cnn_baseline/model.engine \
      --input_shape 2,1723,501 \
      --dry_run=True

Для настоящего TensorRT export требуется установленный NVIDIA TensorRT и доступный `trtexec`:

    uv run seismonn export-tensorrt \
      --onnx outputs/cnn_baseline/model.onnx \
      --engine outputs/cnn_baseline/model.engine \
      --input_shape 2,1723,501

TensorRT wrapper реализован в:

    seismonn/exporting/tensorrt.py
    scripts/export_tensorrt.py
    scripts/export_tensorrt.sh

### 8.3 TorchScript

Дополнительно реализован TorchScript export:

    uv run python scripts/export_torchscript.py \
      --checkpoint outputs/cnn_baseline/best.pt \
      --output outputs/cnn_baseline/model_torchscript.pt \
      --device cpu

TorchScript export реализован в:

    seismonn/exporting/torchscript.py

## 9. Inference server

Для формального inference server используется MLflow Serving.

Сначала нужно сохранить checkpoint как MLflow PyFunc model:

    uv run seismonn save-mlflow-model \
      --checkpoint outputs/cnn_multitask_50ep/best.pt \
      --output outputs/mlflow_models/cnn_multitask \
      --device cpu \
      --predictor_type auto

Затем поднять MLflow Serving:

    uv run mlflow models serve \
      -m outputs/mlflow_models/cnn_multitask \
      --host 127.0.0.1 \
      --port 5001 \
      --no-conda

Пример запроса:

    export SAMPLE=2nd_selection/<sample_name>.npy

    uv run python - <<'PY'
    import json
    import os
    import urllib.request

    sample = os.environ["SAMPLE"]

    payload = {
        "inputs": [
            [sample]
        ]
    }

    request = urllib.request.Request(
        "http://127.0.0.1:5001/invocations",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request) as response:
        print(json.dumps(json.loads(response.read()), indent=2, ensure_ascii=False))
    PY

MLflow Serving model реализован в:

    seismonn/serving/mlflow_model.py

Дополнительно реализован FastAPI inference service:

    seismonn/api/main.py

FastAPI запуск:

    SEISMONN_CHECKPOINT=outputs/cnn_multitask_50ep/best.pt \
    SEISMONN_PREDICTOR_TYPE=auto \
    uv run uvicorn seismonn.api.main:app --host 127.0.0.1 --port 8000

## 10. Model artifacts

Модельные артефакты не хранятся в git.

Пример DVC-tracked checkpoint:

    models/cnn_multitask_50ep/best.pt.dvc

Получить model artifacts:

    uv run dvc pull -r model_storage

Рабочие training artifacts сохраняются в `outputs/` и игнорируются git.

## 11. Code quality and tests

Запуск pre-commit:

    uv run pre-commit run -a

Запуск тестов:

    uv run pytest

На момент финальной проверки проект проходит полный test suite:

    pre-commit: passed
    pytest: passed

## 12. CI

CI workflow:

    .github/workflows/ci.yaml

CI запускается на:

    push
    pull_request

CI проверяет:

    pre-commit
    pytest
    docker build

## 13. Docker

Сборка Docker image:

    docker build -t seismonn-api .

Запуск FastAPI Docker service с classification checkpoint:

    docker run --rm \
      -p 8000:8000 \
      -e SEISMONN_CHECKPOINT=/app/checkpoints/best.pt \
      -e SEISMONN_DEVICE=cpu \
      -e SEISMONN_PREDICTOR_TYPE=auto \
      -v "$(pwd)/outputs/cnn_baseline/best.pt:/app/checkpoints/best.pt:ro" \
      seismonn-api

Запуск FastAPI Docker service с multi-task checkpoint:

    docker run --rm \
      -p 8000:8000 \
      -e SEISMONN_CHECKPOINT=/app/checkpoints/best.pt \
      -e SEISMONN_DEVICE=cpu \
      -e SEISMONN_PREDICTOR_TYPE=auto \
      -v "$(pwd)/outputs/cnn_multitask_50ep/best.pt:/app/checkpoints/best.pt:ro" \
      seismonn-api

## 14. Makefile

Основные команды вынесены в `Makefile`.

Показать список команд:

    make help

Проверки:

    make test
    make lint
    make check

DVC:

    make dvc-pull-data
    make dvc-pull-models

Train:

    make train-lightning
    make train-lightning-local

Inference:

    make predict
    make predict-multitask

Export:

    make export-onnx-cnn
    make export-onnx-multitask
    make export-tensorrt-cnn-dry-run

MLflow Serving:

    make save-mlflow-model
    make serve-mlflow-model

## 15. Дополнительные реализованные компоненты

Помимо обязательных требований, в проекте есть:

    FastAPI inference service
    Docker API deployment
    TorchScript export
    Inference benchmark
    Trace Transformer classifier
    Self-supervised masked trace pre-training
    Fine-tuning Transformer после pre-training
    CNN multi-task baseline
    Multi-task inference
    Multi-task checkpoint evaluation
    Per-sample multi-task prediction export
    Parity plots для regression targets
    RESULTS.md
    DATASET.md
    PROJECT.md

## 16. Известные ограничения

    Данные синтетические.
    В текущей MVP-выборке 665 объектов.
    Нет отдельного официального test split.
    Validation split стратифицирован, но пока не является group split.
    Реальный перенос на field data не проверялся.
    TensorRT engine создаётся только при наличии установленного NVIDIA TensorRT и trtexec.
    DVC remote может быть локальным; для полной внешней воспроизводимости нужен доступный внешний remote.
    Transformer и self-supervised pipeline реализованы, но требуют дополнительных долгих экспериментов для устойчивого качества.

## 17. Короткий итог

Проект закрывает полный MLOps-контур:

    data management
      ↓
    validation
      ↓
    Hydra configs
      ↓
    PyTorch Lightning training
      ↓
    MLflow tracking
      ↓
    checkpoint evaluation
      ↓
    ONNX / TensorRT export
      ↓
    MLflow Serving
      ↓
    tests / pre-commit / CI

Основная MVP-задача:

    классификация количества трещин: 3 / 4 / 5

Расширенная задача:

    classification + regression:
      crack_count
      mean_length
      length_spread
      mean_angle_deg
      angle_spread_deg