# SeismoNN: анализ сейсмограмм трещиноватых сред

**SeismoNN** — учебно-исследовательский ML/MLOps-проект для предсказания параметров трещиноватой среды по синтетическим сейсмическим данным.

Текущая базовая задача — классификация количества трещин по сейсмограмме:

```text
class_id = 0 → 3 трещины
class_id = 1 → 4 трещины
class_id = 2 → 5 трещин
```

Расширенная постановка — multi-task learning:

```text
classification:
- crack_count / class_id

regression:
- mean_length
- length_spread
- mean_angle_deg
- angle_spread_deg
```

## Документация проекта

Дополнительные документы:

```text
COURSE_CHECKLIST.md — соответствие требованиям курса и команды для проверки
PROJECT.md          — обновлённое описание проекта и постановки задачи
DATASET.md          — подробное описание датасета, ограничений и рисков
RESULTS.md          — сводка результатов экспериментов
```

## Практическая мотивация

Трещиноватые среды встречаются в геофизике, инженерной геологии и задачах анализа подземных структур. Трещины влияют на распространение волн, поэтому по сейсмическому отклику можно пытаться оценивать параметры среды.

Практическая ценность проекта:

```text
- автоматизация анализа сейсмических откликов;
- быстрое получение первичной оценки параметров трещиноватой среды;
- воспроизводимое сравнение разных ML-подходов;
- подготовка инфраструктуры для CNN, Transformer и self-supervised learning;
- возможность дальнейшего перехода от синтетических данных к более реалистичным или полевым данным.
```

В рамках курса MLOps основной акцент сделан на полном воспроизводимом контуре:

```text
DVC data management
  ↓
metadata validation
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
MLflow Serving / FastAPI inference
  ↓
tests / pre-commit / CI
```

## Постановка задачи

### Базовая задача

На вход подаётся `.npy` файл с сейсмическим откликом среды. Модель должна предсказать количество трещин:

```text
3, 4 или 5
```

Это задача многоклассовой классификации.

### Расширенная задача

В более общей постановке модель также предсказывает физические параметры среды:

```text
mean_length       — средняя длина трещин
length_spread     — разброс длин трещин
mean_angle_deg    — средний угол трещин
angle_spread_deg  — разброс углов трещин
```

Для этого реализован CNN multi-task baseline: одна модель имеет classification head для `crack_count` и regression head для физических параметров.

## Формат входных данных

Каждый объект датасета — `.npy` файл с тензором:

```text
shape: (2, 1723, 501)
dtype: float32
```

Расшифровка размерностей:

```text
2     — компоненты скорости отражённой волны: vx и vy
1723  — число временных шагов моделирования
501   — число приёмников / receiver positions на поверхности
```

Один входной тензор содержит:

```text
2 * 1723 * 501 = 1 726 446 float32 значений
```

Оценочный размер одного объекта в памяти:

```text
1 726 446 * 4 bytes ≈ 6.6 MiB
```

В исходном ТЗ была указана форма `(2, 1733, 501)`, но проверка текущего датасета показала фактическую форму `(2, 1723, 501)`.

## Формат выходных данных

Для classification модели пример JSON-ответа:

```json
{
  "model_name": "cnn_baseline",
  "input_path": "sample.npy",
  "predicted_class_id": 1,
  "predicted_crack_count": 4,
  "class_probabilities": {
    "3": 0.12,
    "4": 0.81,
    "5": 0.07
  },
  "checkpoint_path": "outputs/cnn_baseline/best.pt"
}
```

Для multi-task модели пример JSON-ответа:

```json
{
  "model_name": "cnn_multitask",
  "input_path": "sample.npy",
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
```

`expected_crack_count` считается как математическое ожидание по вероятностям классов:

```text
E[crack_count] = 3 * P(3) + 4 * P(4) + 5 * P(5)
```

## Датасет

Данные являются результатом компьютерного моделирования сейсмических откликов для сред с различными параметрами трещин.

Публичное хранилище данных:

```text
https://huggingface.co/datasets/FAKIrik/Seismo_datasets
```

Текущая MVP-выборка:

```text
665 объектов / .npy файлов
```

Распределение классов в исходной выборке:

```text
3 трещины: 228 объектов
4 трещины: 228 объектов
5 трещин: 209 объектов
```

Каждый объект описан в `data/metadata.csv`. В git хранится только пример `data/metadata.example.csv`, а основной `metadata.csv` отслеживается через DVC.

Подробное описание данных находится в [`DATASET.md`](DATASET.md).

## Почему данные синтетические

Для реальных трещиноватых сред сложно получить точную supervised-разметку:

```text
- точное количество трещин обычно неизвестно;
- длины и углы трещин не наблюдаются напрямую;
- положение кластера трещин часто доступно только косвенно;
- реальные данные могут быть дорогими, закрытыми или шумными.
```

Синтетическое моделирование позволяет:

```text
- контролируемо задавать параметры среды;
- получать точные labels;
- воспроизводимо сравнивать модели;
- строить supervised и self-supervised пайплайны;
- проверять MLOps-инфраструктуру до появления field data.
```

Главный риск — `domain gap`: модель, обученная на синтетике, может хуже работать на реальных данных.

## Пример имени файла

Пример файла:

```text
receivers_fractures_4_0.0_-150.0_250.0_150.0_30.0_2.0_14.0_14.0.npy
```

Формат имени:

```text
receivers_fractures_{crack_count}_{cluster_center_x}_{cluster_center_y}_{cluster_half_size_x}_{cluster_half_size_y}_{mean_length}_{length_spread}_{mean_angle_deg}_{angle_spread_deg}.npy
```

Расшифровка примера:

```text
crack_count          = 4
cluster_center_x     = 0.0
cluster_center_y     = -150.0
cluster_half_size_x  = 250.0
cluster_half_size_y  = 150.0
mean_length          = 30.0
length_spread        = 2.0
mean_angle_deg       = 14.0
angle_spread_deg     = 14.0
```

## Метрики

Для classification задачи используются:

```text
Cross-Entropy Loss
Accuracy
Balanced Accuracy
Macro Precision
Macro Recall
Macro-F1
Confusion Matrix
Classification Report
```

Для regression targets используются:

```text
MAE
RMSE
per-target MAE/RMSE
```

Для inference/deployment также реализованы:

```text
latency p50
latency p95
latency p99
throughput samples/sec
```

Ориентиры качества:

```text
Random baseline для 3 классов: accuracy ≈ 0.33
Majority-class baseline: accuracy ≈ 228 / 665 ≈ 0.34
Минимальная цель MVP: accuracy и macro-F1 выше случайного baseline
```

## Setup

Проект использует `uv`.

Установка зависимостей:

```bash
uv sync --all-extras --dev
```

Установка git hooks:

```bash
uv run pre-commit install
```

Проверка hooks:

```bash
uv run pre-commit run -a
```

Запуск тестов:

```bash
uv run pytest
```

## Data management через DVC

Большие данные и модельные артефакты не хранятся в git. Для них используется DVC.

В проекте настроены два DVC remote:

```text
data_storage   — для датасета и metadata.csv
model_storage  — для модельных checkpoint-ов
```

Получить данные:

```bash
uv run dvc pull -r data_storage
```

Получить модельные артефакты:

```bash
uv run dvc pull -r model_storage
```

Если DVC remote недоступен, используется fallback-загрузка из Hugging Face:

```bash
uv run seismonn download-data
```

Параметры загрузки находятся в Hydra-конфиге:

```text
configs/hydra/data/seismonn.yaml
```

## Валидация данных

Проверка `metadata.csv` без чтения всех `.npy` файлов:

```bash
uv run seismonn validate-data --validate_files=False
```

Проверка metadata и реальных `.npy` файлов:

```bash
uv run seismonn validate-data --validate_files=True
```

Скрипт проверяет:

```text
- наличие обязательных колонок;
- отсутствие дубликатов sample_id, path, filename;
- корректность split;
- соответствие class_id и crack_count;
- shape из metadata.csv;
- dtype из metadata.csv;
- существование .npy файлов;
- реальный shape и dtype .npy файлов.
```

## Просмотр и визуализация одного объекта

Информация об одном объекте:

```bash
uv run python scripts/inspect_sample.py \
  --metadata data/metadata.csv \
  --data-root . \
  --index 0
```

Визуализация одного объекта:

```bash
uv run python scripts/visualize_sample.py \
  --metadata data/metadata.csv \
  --data-root . \
  --index 0 \
  --output-dir outputs/sample_visualization_small \
  --max-time-steps 400 \
  --max-receivers 200
```

Создаются:

```text
vx_heatmap.png
vy_heatmap.png
vx_receiver_trace.png
vy_receiver_trace.png
sample_info.json
```

## Train: Hydra + PyTorch Lightning

Основной baseline training pipeline для проверки задания реализован через Hydra + PyTorch Lightning.

Короткий запуск без MLflow-сервера:

```bash
uv run seismonn train --overrides "trainer.max_epochs=1 tracking.enabled=false data.ensure_data=false"
```

Запуск MLflow tracking server на адресе из задания:

```bash
uv run mlflow server \
  --host 127.0.0.1 \
  --port 8080 \
  --backend-store-uri ./mlruns \
  --default-artifact-root ./mlartifacts
```

После этого обучение можно запустить так:

```bash
uv run seismonn train --overrides "trainer.max_epochs=5"
```

Hydra-конфиги:

```text
configs/hydra/config.yaml
configs/hydra/data/seismonn.yaml
configs/hydra/model/cnn.yaml
configs/hydra/optimizer/adamw.yaml
configs/hydra/trainer/default.yaml
configs/hydra/tracking/mlflow.yaml
```

Главный Hydra-конфиг содержит defaults:

```yaml
defaults:
  - data: seismonn
  - model: cnn
  - optimizer: adamw
  - trainer: default
  - tracking: mlflow
  - _self_
```

Артефакты Lightning training:

```text
outputs/lightning/cnn_baseline/
plots/lightning/cnn_baseline/
```

Графики обучения:

```text
plots/lightning/cnn_baseline/train_loss.png
plots/lightning/cnn_baseline/val_loss.png
plots/lightning/cnn_baseline/val_accuracy.png
plots/lightning/cnn_baseline/val_macro_f1.png
```

## Дополнительные research pipelines

Помимо основного Lightning/Hydra baseline, в проекте есть исследовательские пайплайны.

CNN baseline:

```bash
uv run python scripts/train.py --config configs/train/cnn.yaml
```

Supervised Trace Transformer:

```bash
uv run python scripts/train.py --config configs/train/transformer.yaml
```

Self-supervised masked trace pre-training:

```bash
uv run python scripts/pretrain_transformer.py \
  --config configs/pretrain/trace_transformer.yaml
```

Fine-tuning Transformer после pre-training:

```bash
uv run python scripts/train.py --config configs/train/transformer_finetune.yaml
```

CNN multi-task baseline:

```bash
uv run python scripts/train_multitask.py \
  --config configs/train/cnn_multitask.yaml
```

## Реализованные модели

### CNN baseline

Архитектура:

```text
Conv2d → ReLU → MaxPool2d
Conv2d → ReLU → MaxPool2d
Conv2d → ReLU → MaxPool2d
AdaptiveAvgPool2d
Linear → ReLU → Dropout → Linear
```

### Trace Transformer classifier

Идея:

```text
1. Каждый receiver / trace рассматривается как токен.
2. Temporal CNN frontend сжимает временную ось.
3. Получается последовательность [B, R, d_model].
4. Добавляется синусоидальное позиционное кодирование.
5. TransformerEncoder моделирует зависимости между трассами.
6. Classification head предсказывает количество трещин.
```

Модель вдохновлена статьёй:

```text
StorSeismic: A new paradigm in deep learning for seismic processing
https://arxiv.org/abs/2205.00222
```

### Self-supervised pre-training

Задача pre-training:

```text
masked trace reconstruction
```

Идея:

```text
1. Берётся сейсмограмма x с формой [B, 2, T, R].
2. Случайно маскируется часть receiver-трасс.
3. Transformer получает повреждённый вход.
4. Модель восстанавливает исходные трассы.
5. MSE loss считается только на замаскированных receiver-трассах.
```

### Multi-task CNN

Модель одновременно предсказывает:

```text
classification:
- crack_count / class_id

regression:
- mean_length
- length_spread
- mean_angle_deg
- angle_spread_deg
```

Loss:

```text
total_loss = classification_loss + regression_loss_weight * regression_loss
```

## Infer через CLI

Рекомендуемый CLI inference через Fire CLI:

```bash
uv run seismonn predict \
  --checkpoint outputs/cnn_multitask_50ep/best.pt \
  --input_path 2nd_selection/<sample_name>.npy \
  --device cpu \
  --predictor_type auto \
  --output outputs/seismonn_cli_prediction.json
```

## Оценка checkpoint

Оценка classification checkpoint:

```bash
uv run python scripts/evaluate_checkpoint.py \
  --checkpoint outputs/cnn_baseline/best.pt \
  --metadata data/metadata.csv \
  --split val \
  --batch-size 16 \
  --device cpu \
  --output outputs/cnn_baseline/evaluation_val.json
```

Оценка multi-task checkpoint:

```bash
uv run python scripts/evaluate_multitask_checkpoint.py \
  --checkpoint outputs/cnn_multitask_50ep/best.pt \
  --metadata data/metadata.csv \
  --split val \
  --batch-size 8 \
  --device cpu \
  --output outputs/cnn_multitask_50ep/evaluation_val.json
```

Сравнение evaluation reports:

```bash
uv run python scripts/compare_evaluations.py \
  --reports \
    outputs/cnn_baseline/evaluation_val.json \
    outputs/trace_transformer/evaluation_val.json \
    outputs/cnn_multitask_50ep/evaluation_val.json \
  --output-csv outputs/model_comparison.csv \
  --output-md outputs/model_comparison.md
```

## Per-sample multi-task predictions

Для анализа регрессионных предсказаний можно сохранить таблицу `true vs predicted` по каждому объекту validation split:

```bash
uv run python scripts/export_multitask_predictions.py \
  --checkpoint outputs/cnn_multitask_50ep/best.pt \
  --metadata data/metadata.csv \
  --split val \
  --batch-size 8 \
  --device cpu \
  --output-csv outputs/cnn_multitask_50ep/predictions_val.csv \
  --summary-output outputs/cnn_multitask_50ep/predictions_summary_val.json \
  --plots-dir outputs/cnn_multitask_50ep/parity_plots
```

Скрипт сохраняет:

```text
predictions_val.csv
predictions_summary_val.json
parity_plots/parity_mean_length.png
parity_plots/parity_length_spread.png
parity_plots/parity_mean_angle_deg.png
parity_plots/parity_angle_spread_deg.png
```

## Production preparation

### ONNX export

Экспорт classification checkpoint в ONNX:

```bash
uv run seismonn export-onnx \
  --checkpoint outputs/cnn_baseline/best.pt \
  --output outputs/cnn_baseline/model.onnx \
  --device cpu
```

Экспорт multi-task checkpoint в ONNX:

```bash
uv run seismonn export-onnx \
  --checkpoint outputs/cnn_multitask_50ep/best.pt \
  --output outputs/cnn_multitask_50ep/model.onnx \
  --device cpu
```

ONNX export реализован в:

```text
seismonn/exporting/onnx.py
```

### TensorRT export

Dry-run TensorRT export:

```bash
uv run seismonn export-tensorrt \
  --onnx outputs/cnn_baseline/model.onnx \
  --engine outputs/cnn_baseline/model.engine \
  --input_shape 2,1723,501 \
  --dry_run=True
```

Для настоящего TensorRT export требуется установленный NVIDIA TensorRT и доступный `trtexec`:

```bash
uv run seismonn export-tensorrt \
  --onnx outputs/cnn_baseline/model.onnx \
  --engine outputs/cnn_baseline/model.engine \
  --input_shape 2,1723,501
```

TensorRT wrapper реализован в:

```text
seismonn/exporting/tensorrt.py
scripts/export_tensorrt.py
scripts/export_tensorrt.sh
```

### TorchScript export

Дополнительно реализован TorchScript export:

```bash
uv run python scripts/export_torchscript.py \
  --checkpoint outputs/cnn_baseline/best.pt \
  --output outputs/cnn_baseline/model_torchscript.pt \
  --device cpu
```

## Inference server: MLflow Serving

Сначала нужно сохранить checkpoint как MLflow PyFunc model:

```bash
uv run seismonn save-mlflow-model \
  --checkpoint outputs/cnn_multitask_50ep/best.pt \
  --output outputs/mlflow_models/cnn_multitask \
  --device cpu \
  --predictor_type auto
```

Затем поднять MLflow Serving:

```bash
uv run mlflow models serve \
  -m outputs/mlflow_models/cnn_multitask \
  --host 127.0.0.1 \
  --port 5001 \
  --no-conda
```

Пример запроса:

```bash
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
```

MLflow Serving model реализован в:

```text
seismonn/serving/mlflow_model.py
```

## FastAPI inference service

Дополнительно реализован FastAPI inference service.

Запуск с classification checkpoint:

```bash
SEISMONN_CHECKPOINT=outputs/cnn_baseline/best.pt \
SEISMONN_PREDICTOR_TYPE=auto \
uv run uvicorn seismonn.api.main:app --host 127.0.0.1 --port 8000
```

Запуск с multi-task checkpoint:

```bash
SEISMONN_CHECKPOINT=outputs/cnn_multitask_50ep/best.pt \
SEISMONN_PREDICTOR_TYPE=auto \
uv run uvicorn seismonn.api.main:app --host 127.0.0.1 --port 8000
```

Проверка:

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

## Docker

Сборка Docker image:

```bash
docker build -t seismonn-api .
```

Запуск FastAPI Docker service с classification checkpoint:

```bash
docker run --rm \
  -p 8000:8000 \
  -e SEISMONN_CHECKPOINT=/app/checkpoints/best.pt \
  -e SEISMONN_DEVICE=cpu \
  -e SEISMONN_PREDICTOR_TYPE=auto \
  -v "$(pwd)/outputs/cnn_baseline/best.pt:/app/checkpoints/best.pt:ro" \
  seismonn-api
```

Запуск FastAPI Docker service с multi-task checkpoint:

```bash
docker run --rm \
  -p 8000:8000 \
  -e SEISMONN_CHECKPOINT=/app/checkpoints/best.pt \
  -e SEISMONN_DEVICE=cpu \
  -e SEISMONN_PREDICTOR_TYPE=auto \
  -v "$(pwd)/outputs/cnn_multitask_50ep/best.pt:/app/checkpoints/best.pt:ro" \
  seismonn-api
```

## Inference benchmark

Benchmark inference:

```bash
uv run python scripts/benchmark_inference.py \
  --checkpoint outputs/cnn_baseline/best.pt \
  --input 2nd_selection/<sample_name>.npy \
  --device cpu \
  --warmup-runs 3 \
  --timed-runs 20 \
  --output outputs/inference_benchmark.json
```

Скрипт измеряет:

```text
model_only latency
end_to_end latency
p50 / p95 / p99
throughput samples/sec
```

## Makefile

Основные команды вынесены в `Makefile`.

Показать список команд:

```bash
make help
```

Проверки:

```bash
make test
make lint
make check
```

DVC:

```bash
make dvc-pull-data
make dvc-pull-models
```

Train:

```bash
make train-lightning
make train-lightning-local
```

Inference:

```bash
make predict
make predict-multitask
```

Export:

```bash
make export-onnx-cnn
make export-onnx-multitask
make export-tensorrt-cnn-dry-run
```

MLflow Serving:

```bash
make save-mlflow-model
make serve-mlflow-model
```

## Tests and code quality

Запуск pre-commit:

```bash
uv run pre-commit run -a
```

Запуск тестов:

```bash
uv run pytest
```

## CI

CI workflow:

```text
.github/workflows/ci.yaml
```

CI запускается на:

```text
push
pull_request
```

CI проверяет:

```text
pre-commit
pytest
docker build
```

## Структура проекта

```text
SeismoNN/
├── configs/
│   ├── hydra/
│   ├── pretrain/
│   └── train/
├── data/
│   ├── metadata.csv.dvc
│   └── metadata.example.csv
├── models/
│   └── cnn_multitask_50ep/
│       └── best.pt.dvc
├── scripts/
├── seismonn/
│   ├── api/
│   ├── data/
│   ├── evaluation/
│   ├── exporting/
│   ├── inference/
│   ├── lightning/
│   ├── models/
│   ├── reporting/
│   ├── serving/
│   ├── tracking/
│   ├── training/
│   └── commands.py
├── tests/
├── .dvc/
├── .github/workflows/
├── COURSE_CHECKLIST.md
├── DATASET.md
├── PROJECT.md
├── RESULTS.md
├── Dockerfile
├── Makefile
├── pyproject.toml
├── uv.lock
└── README.md
```

## Что уже реализовано

```text
Metadata-based dataset description
DVC data/model artifact management
download_data / ensure_data_available
Hydra hierarchical configs
PyTorch Lightning baseline training
Fire CLI entrypoint
Pre-commit + Ruff
Pytest test suite
GitHub Actions CI
MLflow tracking
MLflow Serving
FastAPI inference service
Docker deployment
CNN classification baseline
Trace Transformer classifier
Self-supervised masked trace pre-training
Transformer fine-tuning after pre-training
CNN multi-task baseline: classification + regression
Multi-task inference
Multi-task checkpoint evaluation
Per-sample multi-task prediction export
Parity plots for regression targets
ONNX export
TensorRT export wrapper
TorchScript export
Inference benchmark
DATASET.md
PROJECT.md
RESULTS.md
COURSE_CHECKLIST.md
```

## Что планируется добавить

```text
1. Group split:
   более строгая проверка качества без утечки близких параметрических конфигураций между train и validation.

2. Официальный test split:
   отдельная тестовая выборка, которая не используется при выборе гиперпараметров.

3. Triton Inference Server:
   сейчас реализован MLflow Serving, но Triton можно добавить как более production-oriented inference server.

4. ONNX/TensorRT для Transformer и multi-task моделей:
   сейчас основной ONNX/TensorRT path проверен на CNN baseline.

5. Проверка на более реалистичных или реальных данных:
   оценка domain gap между синтетическими и field-data сейсмограммами.

6. Улучшение DVC remote:
   заменить локальный storage на полностью доступный преподавателям внешний remote.
```

## Ограничения текущей версии

```text
- Данные синтетические.
- Количество объектов небольшое: 665.
- Нет отдельного официального test split.
- Реальный перенос на field-data пока не проверялся.
- Validation split стратифицирован, но пока не является group split.
- TensorRT engine создаётся только при наличии установленного NVIDIA TensorRT/trtexec.
- DVC remote может быть локальным; для полной внешней воспроизводимости нужен доступный внешний remote.
- Transformer и self-supervised pipeline реализованы, но требуют дополнительных долгих экспериментов для устойчивого качества.
```

## Краткий итог

Текущая версия SeismoNN закрывает полный MLOps-контур:

```text
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
```

Основная MVP-задача:

```text
классификация количества трещин: 3 / 4 / 5
```

Расширенная задача:

```text
classification + regression:
  crack_count
  mean_length
  length_spread
  mean_angle_deg
  angle_spread_deg
```
