# SeismoNN: MLOps-система для анализа сейсмограмм трещиноватых сред

**SeismoNN** — учебно-исследовательский MLOps-проект для анализа синтетических сейсмических откликов трещиноватых сред. Проект строит воспроизводимый ML-пайплайн: от описания данных и обучения моделей до inference API, Docker, CI и анализа результатов.

Краткая спецификация проекта находится в [`PROJECT.md`](PROJECT.md), подробное описание датасета — в [`DATASET.md`](DATASET.md).

## Кратко о задаче

Текущая MVP-задача — классификация количества трещин в среде по сейсмограмме:

```text
3 трещины
4 трещины
5 трещин
```

Расширенная постановка уже поддерживается через multi-task baseline:

```text
classification:
- crack_count / class_id

regression:
- mean_length
- length_spread
- mean_angle_deg
- angle_spread_deg
```

Количество трещин остаётся классификационной целью, потому что это дискретная величина. Непрерывные физические параметры трещин предсказываются регрессионной головой.

## Практическая мотивация

Трещиноватые среды встречаются в геофизике, инженерной геологии и задачах анализа подземных структур. Трещины влияют на распространение волн, поэтому по сейсмическому отклику можно оценивать параметры среды.

Практическая ценность проекта:

```text
- автоматизация анализа сейсмических откликов;
- первичная оценка параметров трещиноватой среды;
- сравнение CNN, Transformer и self-supervised подходов;
- воспроизводимый MLOps-пайплайн для экспериментов;
- подготовка инфраструктуры для будущей проверки на более реалистичных данных.
```

## Формат входных данных

Каждый объект датасета — `.npy` файл:

```text
shape: (2, 1723, 501)
dtype: float32
```

Расшифровка размерностей:

```text
2     — компоненты скорости отражённой волны: vx и vy;
1723  — временные шаги моделирования;
501   — приёмники / receiver positions на поверхности.
```

Один объект содержит:

```text
2 * 1723 * 501 = 1 726 446 float32 значений
```

Оценочный размер одного объекта в памяти:

```text
1 726 446 * 4 bytes ≈ 6.6 MiB
```

В исходном ТЗ фигурировала форма `(2, 1733, 501)`, но фактическая форма всех файлов текущего датасета — `(2, 1723, 501)`.

## Формат выходных данных

Для классификационной модели JSON-ответ выглядит так:

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

Для multi-task модели ответ дополнительно содержит регрессионные параметры:

```json
{
  "model_name": "cnn_multitask",
  "predicted_class_id": 1,
  "predicted_crack_count": 4,
  "expected_crack_count": 3.92,
  "class_probabilities": {
    "3": 0.10,
    "4": 0.86,
    "5": 0.04
  },
  "regression": {
    "mean_length": 29.4,
    "length_spread": 2.2,
    "mean_angle_deg": 13.7,
    "angle_spread_deg": 4.5
  }
}
```

`expected_crack_count` считается как математическое ожидание:

```text
E[crack_count] = 3 * P(3) + 4 * P(4) + 5 * P(5)
```

## Пример входного файла

Пример имени файла:

```text
receivers_fractures_4_0.0_-150.0_250.0_150.0_30.0_2.0_14.0_14.0.npy
```

Формат имени:

```text
receivers_fractures_{crack_count}_{cluster_center_x}_{cluster_center_y}_{cluster_half_size_x}_{cluster_half_size_y}_{mean_length}_{length_spread}_{mean_angle_deg}_{angle_spread_deg}.npy
```

Расшифровка:

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

## Датасет

Данные являются результатом компьютерного моделирования сейсмических откликов для сред с различными параметрами трещин.

Публичное хранилище:

```text
https://huggingface.co/datasets/FAKIrik/Seismo_datasets
```

Текущее состояние Hugging Face страницы:

```text
- общий размер файлов: 17.6 GB;
- dataset viewer недоступен;
- dataset card пока отсутствует.
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

Каждый объект описан в:

```text
data/metadata.csv
```

Основные колонки metadata:

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
split_seed
split_strategy
split_stratify_column
```

## Почему данные синтетические

Для реальных трещиноватых сред трудно получить точную supervised-разметку: обычно неизвестны точное количество трещин, длины, углы и положение кластера. Синтетическое моделирование позволяет контролируемо задавать параметры среды и получать точные labels.

Главный риск такого подхода:

```text
domain gap
```

Модель, обученная на синтетике, может хуже переноситься на реальные полевые данные. Возможные будущие меры: шумы, аугментации, domain randomization, self-supervised pre-training на неразмеченных данных и fine-tuning на более реалистичных данных.

## Особенности и риски данных

```text
1. Небольшое число объектов: 665.
2. Большой размер одного объекта: (2, 1723, 501).
3. Синтетическая природа данных.
4. Возможная неоднозначность обратной задачи.
5. Риск утечки через параметрическую сетку при случайном split.
```

В проекте используются:

```text
- metadata.csv для описания данных;
- lazy loading .npy файлов;
- np.load(..., mmap_mode="r");
- фиксированное стратифицированное разбиение;
- тесты без загрузки полного датасета.
```

## Train/validation split

Используется фиксированное стратифицированное разбиение по `class_id`:

```text
train: 532 объекта
val:   133 объекта
total: 665 объектов
```

Команда генерации split:

```bash
uv run python scripts/create_split.py \
  --input data/metadata.csv \
  --output data/metadata_stratified.csv \
  --val-size 0.2 \
  --test-size 0.0 \
  --seed 42 \
  --stratify-column class_id
```

В metadata сохраняются поля:

```text
split_seed
split_strategy
split_stratify_column
```

## Метрики

Для классификации используются:

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

Для regression targets в multi-task модели используются:

```text
MAE
RMSE
per-target MAE/RMSE
```

Для deployment дополнительно измеряются:

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
Ожидаемый уровень CNN baseline после настройки: accuracy > 0.40–0.50
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

Вход:

```text
[B, 2, 1723, 501]
```

Выход:

```text
[B, 3]
```

### Trace Transformer classifier

Transformer-модель вдохновлена статьёй:

```text
StorSeismic: A new paradigm in deep learning for seismic processing
https://arxiv.org/abs/2205.00222
```

Адаптация к сейсмическим данным:

```text
1. Каждый receiver / trace рассматривается как токен.
2. Temporal CNN frontend сжимает временную ось.
3. Получается последовательность [B, R, d_model].
4. Добавляется синусоидальное позиционное кодирование.
5. TransformerEncoder моделирует зависимости между трассами.
6. Classification head предсказывает количество трещин.
```

### Self-supervised masked trace pre-training

Задача pre-training:

```text
masked trace reconstruction
```

Схема:

```text
1. Берётся сейсмограмма x с формой [B, 2, T, R].
2. Случайно маскируется часть receiver-трасс.
3. Transformer получает повреждённый вход.
4. Модель восстанавливает исходные трассы.
5. MSE loss считается только на замаскированных receiver-трассах.
```

### Fine-tuning после pre-training

После self-supervised pre-training веса `temporal_encoder.*` и `encoder.*` переносятся в supervised Transformer-классификатор. Голова реконструкции не переносится, потому что в supervised задаче используется classification head.

### CNN multi-task baseline

Модель одновременно решает классификацию и регрессию:

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
classification_loss = CrossEntropyLoss
regression_loss = MSELoss по нормализованным regression targets
```

## Установка

Проект использует `uv`.

```bash
uv sync
```

Установка с dev-зависимостями:

```bash
uv sync --all-extras --dev
```

Все команды рекомендуется запускать из корня репозитория.

## Makefile

Основные команды вынесены в `Makefile`.

Показать список доступных команд:

```bash
make help
```

Базовые проверки:

```bash
make test
make lint
make check
```

Работа с данными:

```bash
make validate-metadata
make validate-files
make inspect-sample
make visualize-sample
```

Обучение:

```bash
make train-cnn
make train-transformer
make pretrain-transformer
make train-transformer-finetuned
make train-multitask
```

Inference и deployment:

```bash
make predict
make predict-multitask
make api
make api-multitask
make docker-build
make docker-run
make docker-run-multitask
```

Некоторые переменные можно переопределить:

```bash
make predict SAMPLE=2nd_selection/<sample_name>.npy
make api PORT=8080 DEVICE=cpu
make evaluate-cnn CNN_CKPT=outputs/cnn_baseline_50ep/best.pt
```

## Работа с данными

### Генерация metadata.csv

```bash
uv run python scripts/build_metadata.py \
  --split-json 2nd_sel.json \
  --data-dir 2nd_selection \
  --output data/metadata.csv \
  --test-split-name val \
  --validate-files
```

### Валидация metadata и .npy файлов

Только metadata:

```bash
uv run python scripts/validate_metadata.py \
  --metadata data/metadata.csv \
  --data-root . \
  --expected-shape 2 1723 501 \
  --expected-dtype float32 \
  --expected-splits train val
```

Metadata + реальные `.npy` файлы:

```bash
uv run python scripts/validate_metadata.py \
  --metadata data/metadata.csv \
  --data-root . \
  --expected-shape 2 1723 501 \
  --expected-dtype float32 \
  --expected-splits train val \
  --validate-files \
  --output outputs/metadata_validation.json
```

### Анализ одного sample

```bash
uv run python scripts/inspect_sample.py \
  --metadata data/metadata.csv \
  --data-root . \
  --index 0
```

### Визуализация одного sample

```bash
uv run python scripts/visualize_sample.py \
  --metadata data/metadata.csv \
  --data-root . \
  --index 0 \
  --output-dir outputs/sample_visualization_small \
  --max-time-steps 400 \
  --max-receivers 200
```

Скрипт сохраняет:

```text
vx_heatmap.png
vy_heatmap.png
vx_receiver_trace.png
vy_receiver_trace.png
sample_info.json
```

## Обучение

CNN baseline:

```bash
uv run python scripts/train.py --config configs/train/cnn.yaml
```

Trace Transformer:

```bash
uv run python scripts/train.py --config configs/train/transformer.yaml
```

Self-supervised pre-training:

```bash
uv run python scripts/pretrain_transformer.py \
  --config configs/pretrain/trace_transformer.yaml
```

Fine-tuning после pre-training:

```bash
uv run python scripts/train.py \
  --config configs/train/transformer_finetune.yaml
```

CNN multi-task baseline:

```bash
uv run python scripts/train_multitask.py \
  --config configs/train/cnn_multitask.yaml
```

Артефакты обучения сохраняются в `outputs/`:

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

`outputs/` не коммитится в git.

## MLflow tracking

MLflow включается в training config:

```yaml
tracking:
  enabled: true
  tracking_uri: mlruns
  experiment_name: seismonn
  run_name: cnn_baseline
  log_artifacts: true
```

При обучении логируются:

```text
- параметры конфига;
- train/validation метрики;
- финальные метрики;
- checkpoint-и;
- графики;
- metrics.json;
- history.csv.
```

Запуск UI:

```bash
uv run mlflow ui --backend-store-uri mlruns
```

Адрес:

```text
http://127.0.0.1:5000
```

## Оценка и сравнение моделей

### Оценка classification checkpoint

```bash
uv run python scripts/evaluate_checkpoint.py \
  --checkpoint outputs/cnn_baseline/best.pt \
  --metadata data/metadata.csv \
  --split val \
  --batch-size 16 \
  --device cpu \
  --output outputs/cnn_baseline/evaluation_val.json
```

### Оценка multi-task checkpoint

```bash
uv run python scripts/evaluate_multitask_checkpoint.py \
  --checkpoint outputs/cnn_multitask_50ep/best.pt \
  --metadata data/metadata.csv \
  --split val \
  --batch-size 8 \
  --device cpu \
  --output outputs/cnn_multitask_50ep/evaluation_val.json
```

### Сравнение нескольких evaluation reports

```bash
uv run python scripts/compare_evaluations.py \
  --reports \
    outputs/cnn_baseline/evaluation_val.json \
    outputs/trace_transformer/evaluation_val.json \
    outputs/trace_transformer_finetuned/evaluation_val.json \
    outputs/cnn_multitask_50ep/evaluation_val.json \
  --output-csv outputs/model_comparison.csv \
  --output-md outputs/model_comparison.md
```

### Per-sample multi-task predictions

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

Сохраняются:

```text
predictions_val.csv
predictions_summary_val.json
parity_plots/parity_mean_length.png
parity_plots/parity_length_spread.png
parity_plots/parity_mean_angle_deg.png
parity_plots/parity_angle_spread_deg.png
```

## CLI inference

Classification checkpoint:

```bash
uv run python scripts/predict.py \
  --checkpoint outputs/cnn_baseline/best.pt \
  --input 2nd_selection/<sample_name>.npy \
  --output outputs/sample_prediction.json
```

Multi-task checkpoint:

```bash
uv run python scripts/predict_multitask.py \
  --checkpoint outputs/cnn_multitask_50ep/best.pt \
  --input 2nd_selection/<sample_name>.npy \
  --device cpu \
  --output outputs/cnn_multitask_50ep/sample_multitask_prediction.json
```

Получить пример пути из metadata:

```bash
uv run python - <<'PY'
import pandas as pd

df = pd.read_csv("data/metadata.csv")
print(df.iloc[0]["path"])
PY
```

## FastAPI inference service

Classification checkpoint:

```bash
SEISMONN_CHECKPOINT=outputs/cnn_baseline/best.pt \
SEISMONN_PREDICTOR_TYPE=auto \
uv run uvicorn seismonn.api.main:app --host 127.0.0.1 --port 8000
```

Multi-task checkpoint:

```bash
SEISMONN_CHECKPOINT=outputs/cnn_multitask_50ep/best.pt \
SEISMONN_PREDICTOR_TYPE=auto \
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

API автоматически определяет тип checkpoint:

```text
cnn_baseline / trace_transformer → classification predictor
cnn_multitask                   → multi-task predictor
```

## Docker

Сборка image:

```bash
docker build -t seismonn-api .
```

Classification checkpoint:

```bash
docker run --rm \
  -p 8000:8000 \
  -e SEISMONN_CHECKPOINT=/app/checkpoints/best.pt \
  -e SEISMONN_DEVICE=cpu \
  -e SEISMONN_PREDICTOR_TYPE=auto \
  -v "$(pwd)/outputs/cnn_baseline/best.pt:/app/checkpoints/best.pt:ro" \
  seismonn-api
```

Multi-task checkpoint:

```bash
docker run --rm \
  -p 8000:8000 \
  -e SEISMONN_CHECKPOINT=/app/checkpoints/best.pt \
  -e SEISMONN_DEVICE=cpu \
  -e SEISMONN_PREDICTOR_TYPE=auto \
  -v "$(pwd)/outputs/cnn_multitask_50ep/best.pt:/app/checkpoints/best.pt:ro" \
  seismonn-api
```

## Benchmark инференса

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

## TorchScript export

Classification checkpoint:

```bash
uv run python scripts/export_torchscript.py \
  --checkpoint outputs/cnn_baseline/best.pt \
  --output outputs/cnn_baseline/model_torchscript.pt \
  --device cpu
```

Multi-task checkpoint:

```bash
uv run python scripts/export_torchscript.py \
  --checkpoint outputs/cnn_multitask_50ep/best.pt \
  --output outputs/cnn_multitask_50ep/model_torchscript.pt \
  --device cpu
```

После экспорта создаются:

```text
model_torchscript.pt
model_torchscript.metadata.json
```

Для classification модели TorchScript возвращает:

```text
logits: [B, 3]
```

Для multi-task модели TorchScript возвращает:

```text
logits:     [B, 3]
regression: [B, 4]
```

## Тесты, code quality и CI

Запуск всех тестов:

```bash
uv run pytest
```

Pre-commit:

```bash
uv run pre-commit run --all-files
```

CI:

```text
.github/workflows/ci.yaml
```

CI запускается на push и pull request и проверяет:

```text
pre-commit checks
pytest tests
Docker image build
```

## Структура проекта

```text
SeismoNN/
├── configs/
│   ├── pretrain/
│   └── train/
├── data/
│   └── metadata.csv
├── scripts/
│   ├── build_metadata.py
│   ├── create_split.py
│   ├── train.py
│   ├── train_multitask.py
│   ├── pretrain_transformer.py
│   ├── predict.py
│   ├── predict_multitask.py
│   ├── evaluate_checkpoint.py
│   ├── evaluate_multitask_checkpoint.py
│   ├── compare_evaluations.py
│   ├── export_multitask_predictions.py
│   ├── export_torchscript.py
│   ├── benchmark_inference.py
│   ├── inspect_sample.py
│   ├── validate_metadata.py
│   └── visualize_sample.py
├── seismonn/
│   ├── api/
│   ├── data/
│   ├── evaluation/
│   ├── exporting/
│   ├── inference/
│   ├── models/
│   ├── tracking/
│   └── training/
├── tests/
├── DATASET.md
├── PROJECT.md
├── Dockerfile
├── Makefile
├── pyproject.toml
├── uv.lock
└── README.md
```

## Что уже реализовано

```text
- metadata.csv и описание датасета;
- DATASET.md и PROJECT.md;
- reproducible stratified split;
- validation metadata и .npy файлов;
- sample inspection и visualization;
- PyTorch Dataset;
- CNN classification baseline;
- Trace Transformer classifier;
- masked trace self-supervised pre-training;
- fine-tuning Transformer после pre-training;
- CNN multi-task baseline: classification + regression;
- расширенные метрики классификации;
- regression MAE/RMSE для multi-task модели;
- MLflow experiment tracking;
- checkpoint evaluation;
- model comparison from evaluation reports;
- per-sample multi-task prediction export;
- parity plots для regression targets;
- CLI inference;
- universal FastAPI inference;
- Docker deployment;
- inference benchmark;
- TorchScript export;
- Makefile;
- pytest tests;
- pre-commit;
- GitHub Actions CI.
```

## Что планируется добавить

```text
1. Group split:
   более строгая проверка качества без утечки близких параметрических конфигураций между train и validation.

2. Официальный test split:
   отдельная тестовая выборка, которая не используется при выборе гиперпараметров.

3. Multi-task Transformer:
   расширение multi-task постановки с CNN baseline на Transformer encoder.

4. Аугментации данных:
   шум, receiver dropout, crop/pad/resample.

5. DVC remote или другой воспроизводимый способ загрузки данных:
   сейчас данные не хранятся в git, но нужен явно настроенный внешний storage.

6. Проверка на более реалистичных или реальных данных:
   оценка domain gap между синтетическими и field-data сейсмограммами.

7. ONNX или torch.export:
   более современный deployment export, если это потребуется.
```

## Ограничения текущей версии

```text
- данные синтетические;
- текущая MVP-выборка содержит 665 объектов;
- нет отдельного официального test split;
- перенос на реальные field data пока не проверялся;
- возможен domain gap;
- возможна утечка близких параметрических конфигураций между train и validation;
- Transformer и self-supervised pipeline являются экспериментальными;
- multi-task regression baseline реализован только для CNN;
- TorchScript export добавлен, но ONNX / torch.export пока не реализованы.
```

## Краткий итог

SeismoNN — это воспроизводимый MLOps-пайплайн для анализа сейсмических откликов трещиноватых сред.

Текущий pipeline:

```text
данные
  ↓
metadata.csv
  ↓
validation
  ↓
training
  ↓
tracking
  ↓
evaluation
  ↓
inference
  ↓
API / Docker
  ↓
CI
```

Проект уже поддерживает классификацию количества трещин, multi-task baseline для регрессии физических параметров, Transformer-модель, self-supervised pre-training и полный набор MLOps-компонентов для воспроизводимых экспериментов.
