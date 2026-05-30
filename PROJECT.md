# PROJECT.md: обновлённое описание проекта SeismoNN

## 1. Название проекта

**SeismoNN** — MLOps-система для анализа сейсмических откликов трещиноватых сред.

Название расшифровывается как:

```text
SeismoNN = Seismic Neural Network
```

Для неспециалиста проект можно описать так:

> Система получает на вход результат сейсмического моделирования и предсказывает, сколько трещин находится в исследуемой среде. В дальнейшем система может быть расширена до предсказания физических параметров трещин: длины, углов и разбросов.

## 2. Краткое описание задачи

Проект посвящён задаче восстановления параметров трещиноватой среды по сейсмическому отклику.

В текущей MVP-версии решается задача классификации количества трещин:

```text
3 трещины
4 трещины
5 трещин
```

Это задача многоклассовой классификации.

Полная исследовательская цель шире: по сейсмограмме предсказывать не только количество трещин, но и другие параметры среды:

```text
- средняя длина трещин;
- разброс длин;
- средний угол трещин;
- разброс углов;
- положение и размер области, где расположены трещины.
```

## 3. Зачем это нужно

Трещиноватые среды встречаются в геофизике, инженерной геологии и задачах анализа подземных структур. Трещины влияют на распространение волн, поэтому по сейсмическому отклику можно пытаться оценивать параметры среды.

Практическая ценность проекта:

```text
- ускорение анализа сейсмических откликов;
- автоматизация первичной оценки параметров среды;
- создание воспроизводимого ML/MLOps-пайплайна;
- подготовка инфраструктуры для экспериментов с CNN, Transformer и self-supervised learning;
- возможность дальнейшего перехода от синтетических данных к полевым данным.
```

## 4. Почему текущая задача — классификация, а не сразу регрессия

Изначальная постановка задачи предполагала предсказание средней длины и угла трещин. Однако в рамках MVP выбрана более устойчивая первая подзадача:

```text
классификация количества трещин: 3 / 4 / 5
```

Причины:

```text
- есть размеченный датасет для этой постановки;
- целевая переменная дискретная и хорошо подходит для baseline;
- проще построить воспроизводимый train/validation split;
- проще определить метрики качества;
- можно быстрее проверить полный MLOps-контур:
  данные → обучение → checkpoint → inference → API → Docker → CI.
```

После стабилизации MVP проект расширяется до:

```text
- supervised Transformer;
- self-supervised pre-training;
- fine-tuning после pre-training;
- multi-task learning;
- регрессии физических параметров трещиноватой среды.
```

## 5. Формат входных данных

На вход подаётся `.npy` файл.

Форма одного объекта:

```text
shape: (2, 1723, 501)
dtype: float32
```

Расшифровка размерностей:

```text
2     — две компоненты скорости отражённой волны: vx и vy;
1723  — число временных шагов моделирования;
501   — число приёмников / receiver positions на поверхности.
```

Изначально в ТЗ была указана форма `(2, 1733, 501)`, но после проверки всех файлов текущего датасета фактическая форма оказалась:

```text
(2, 1723, 501)
```

Для проверки формата данных реализован скрипт:

```bash
uv run python scripts/validate_metadata.py \
  --metadata data/metadata.csv \
  --data_root . \
  --expected_shape 2 1723 501 \
  --expected_dtype float32 \
  --expected_splits train val \
  --validate_files
```

## 6. Формат выходных данных

На выходе модель возвращает JSON.

Пример:

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

Поля:

```text
predicted_class_id     — предсказанный номер класса;
predicted_crack_count  — физически интерпретируемый результат: число трещин;
class_probabilities    — вероятности классов после softmax;
checkpoint_path        — checkpoint модели.
```

## 7. Датасет

Данные являются результатом компьютерного моделирования сейсмического отклика трещиноватой среды.

Публичное хранилище:

```text
https://huggingface.co/datasets/FAKIrik/Seismo_datasets
```

Текущее состояние страницы Hugging Face:

```text
- общий размер файлов: 17.6 GB;
- dataset viewer недоступен;
- dataset card пока отсутствует.
```

Поэтому в репозитории добавлен собственный документ:

```text
DATASET.md
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

## 8. Пример входного файла

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

Для просмотра конкретного объекта реализован скрипт:

```bash
uv run python scripts/inspect_sample.py \
  --metadata data/metadata.csv \
  --data_root . \
  --index 0
```

Для визуализации:

```bash
uv run python scripts/visualize_sample.py \
  --metadata data/metadata.csv \
  --data_root . \
  --index 0 \
  --output_dir outputs/sample_visualization
```

## 9. Почему данные синтетические

Данные синтетические, потому что для реальных трещиноватых сред трудно получить точную supervised-разметку.

Для реальной среды обычно неизвестны:

```text
- точное количество трещин;
- длины трещин;
- углы трещин;
- положение кластера трещин;
- геометрия трещиноватой области.
```

Синтетическое моделирование позволяет:

```text
- контролируемо задавать параметры среды;
- получать точные labels;
- воспроизводимо генерировать данные;
- строить supervised и self-supervised ML-пайплайны;
- проверять MLOps-инфраструктуру до появления field data.
```

Главный риск:

```text
domain gap
```

То есть модель, обученная на синтетике, может хуже работать на реальных данных. Это ограничение явно указано в README и DATASET.md.

## 10. Валидация и split

Для текущего проекта используется фиксированное стратифицированное разбиение по `class_id`.

Текущее разбиение:

```text
train: 532 объекта
val:   133 объекта
total: 665 объектов
```

Разбиение создаётся командой:

```bash
uv run python scripts/create_split.py \
  --input data/metadata.csv \
  --output data/metadata_stratified.csv \
  --val_size 0.2 \
  --test_size 0.0 \
  --seed 42 \
  --stratify_column class_id
```

В metadata сохраняются поля:

```text
split_seed
split_strategy
split_stratify_column
```

Это делает разбиение воспроизводимым.

Ограничение текущего split:

```text
stratified split не исключает попадание близких параметрических конфигураций одновременно в train и validation.
```

Планируемое улучшение:

```text
group split по параметрам моделирования
```

## 11. Метрики

Для текущей задачи классификации используются метрики:

```text
1. Cross-Entropy Loss
2. Accuracy
3. Balanced Accuracy
4. Macro Precision
5. Macro Recall
6. Macro-F1
7. Confusion Matrix
8. Classification Report
```

Обоснование:

```text
Accuracy
  показывает общую долю правильных ответов.

Balanced Accuracy
  полезна при возможном дисбалансе классов.

Macro Precision
  показывает качество положительных предсказаний по классам.

Macro Recall
  показывает, насколько хорошо модель находит каждый класс.

Macro-F1
  объединяет precision и recall и одинаково учитывает классы.

Confusion Matrix
  показывает, какие классы модель путает.

Cross-Entropy Loss
  используется как функция потерь при обучении.
```

Ориентиры качества:

```text
Random baseline для 3 классов:
accuracy ≈ 0.33

Majority-class baseline:
accuracy ≈ 228 / 665 ≈ 0.34

Минимальная цель MVP:
accuracy и macro-F1 выше случайного baseline.

Ожидаемый уровень для CNN baseline после настройки:
accuracy > 0.40–0.50
macro-F1 > 0.35–0.45

Цель основной модели:
улучшение относительно CNN baseline.
```

Для будущей регрессии будут использоваться:

```text
MAE
RMSE
R²
ошибка по углу в градусах
относительная ошибка длины трещин
```

Для внедрения добавлен benchmark инференса:

```text
latency p50
latency p95
latency p99
throughput samples/sec
```

## 12. Baseline

Baseline-модель — CNN-классификатор.

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

Baseline pipeline:

```text
.npy файл
  ↓
metadata.csv
  ↓
SeismoDataset
  ↓
нормализация sample
  ↓
DataLoader
  ↓
CNN
  ↓
CrossEntropyLoss
  ↓
Adam / AdamW
  ↓
валидация
  ↓
best.pt по val_macro_f1
  ↓
CLI / API / Docker inference
```

## 13. Основная модель

Основная модель-кандидат — Trace Transformer classifier.

Идея:

```text
x имеет форму [B, 2, T, R]

B — batch size
2 — компоненты скорости
T — временные шаги
R — receiver positions
```

Адаптация Transformer:

```text
1. Каждый receiver / trace рассматривается как токен.
2. Temporal CNN frontend сжимает временную ось.
3. Получается последовательность [B, R, d_model].
4. Добавляется синусоидальное позиционное кодирование.
5. TransformerEncoder моделирует зависимости между трассами.
6. Classification head предсказывает количество трещин.
```

Также реализована self-supervised схема:

```text
masked trace reconstruction
```

Она состоит из двух этапов:

```text
1. Self-supervised pre-training:
   модель восстанавливает замаскированные receiver-трассы.

2. Supervised fine-tuning:
   encoder переносится в классификатор и дообучается на crack_count.
```

## 14. Пайплайн обучения

Универсальный training entrypoint:

```bash
uv run python scripts/train.py --config configs/train/cnn.yaml
```

CNN baseline:

```bash
uv run python scripts/train.py --config configs/train/cnn.yaml
```

Supervised Transformer:

```bash
uv run python scripts/train.py --config configs/train/transformer.yaml
```

Fine-tuned Transformer:

```bash
uv run python scripts/train.py --config configs/train/transformer_finetune.yaml
```

Self-supervised pre-training:

```bash
uv run python scripts/pretrain_transformer.py \
  --config configs/pretrain/trace_transformer.yaml
```

Артефакты обучения:

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

## 15. Experiment tracking

Реализовано логирование экспериментов через MLflow.

Логируются:

```text
- параметры конфига;
- train/validation метрики;
- финальные метрики;
- checkpoint-и;
- графики;
- metrics.json;
- history.csv.
```

Запуск MLflow UI:

```bash
uv run mlflow ui --backend-store-uri mlruns
```

После запуска UI доступен по адресу:

```text
http://127.0.0.1:5000
```

## 16. Оценка моделей

Для независимой оценки checkpoint реализован скрипт:

```bash
uv run python scripts/evaluate_checkpoint.py \
  --checkpoint outputs/cnn_baseline/best.pt \
  --metadata data/metadata.csv \
  --split val \
  --batch_size 16 \
  --device cpu \
  --output outputs/cnn_baseline/evaluation_val.json
```

Для сравнения нескольких моделей:

```bash
uv run python scripts/compare_evaluations.py \
  --reports \
    outputs/cnn_baseline/evaluation_val.json \
    outputs/trace_transformer/evaluation_val.json \
    outputs/trace_transformer_finetuned/evaluation_val.json \
  --output_csv outputs/model_comparison.csv \
  --output_md outputs/model_comparison.md
```

## 17. Внедрение

Форматы модели:

- PyTorch checkpoint: best.pt / last.pt;
- TorchScript export: model_torchscript.pt.

Реализованные способы использования модели:

```text
1. Python-пакет seismonn
2. CLI inference
3. FastAPI HTTP service
4. Docker container
```

CLI inference:

```bash
uv run python scripts/predict.py \
  --checkpoint outputs/cnn_baseline/best.pt \
  --input_path 2nd_selection/<sample_name>.npy
```

FastAPI:

```bash
SEISMONN_CHECKPOINT=outputs/cnn_baseline/best.pt \
uv run uvicorn seismonn.api.main:app --host 127.0.0.1 --port 8000
```

Docker:

```bash
docker build -t seismonn-api .
```

```bash
docker run --rm \
  -p 8000:8000 \
  -e SEISMONN_CHECKPOINT=/app/checkpoints/best.pt \
  -e SEISMONN_DEVICE=cpu \
  -v "$(pwd)/outputs/cnn_baseline/best.pt:/app/checkpoints/best.pt:ro" \
  seismonn-api
```

API endpoints:

```text
GET  /health
POST /predict
```

## 18. Ресурсы

CNN baseline:

```text
CPU: возможно, но обучение может быть медленным
GPU: желательно
RAM: зависит от batch_size
```

Transformer:

```text
GPU: желательно
CPU: возможно для тестов и коротких запусков, но медленно
VRAM: зависит от d_model, числа слоёв, числа heads и batch_size
```

Inference:

```text
CNN inference на одном sample возможен на CPU.
Transformer inference также возможен на CPU, но может быть медленнее.
```

Для измерения скорости реализован benchmark:

```bash
uv run python scripts/benchmark_inference.py \
  --checkpoint outputs/cnn_baseline/best.pt \
  --input_path 2nd_selection/<sample_name>.npy \
  --device cpu \
  --warmup_runs 3 \
  --timed_runs 20 \
  --output outputs/inference_benchmark.json
```

## 19. Технологический стек

```text
Python 3.12
PyTorch
NumPy
Pandas
Scikit-learn
Matplotlib
PyYAML
FastAPI
Uvicorn
Docker
MLflow
Pytest
pre-commit
Ruff
GitHub Actions
uv
```

Назначение:

```text
PyTorch        — модели и обучение
NumPy          — работа с .npy файлами
Pandas         — metadata.csv
Scikit-learn   — метрики и split
Matplotlib     — графики и визуализация
FastAPI        — HTTP inference
Docker         — контейнеризация
MLflow         — experiment tracking
Pytest         — тесты
pre-commit     — проверки перед коммитом
Ruff           — линтинг и форматирование
GitHub Actions — CI
uv             — управление зависимостями
```

## 20. CI/CD

В проекте реализован CI через GitHub Actions.

Workflow:

```text
.github/workflows/ci.yaml
```

CI запускается на:

```text
push
pull_request
```

Проверки:

```text
pre-commit
pytest
docker build
```

## 21. Текущий статус реализации

Реализовано:

```text
- metadata.csv;
- DATASET.md;
- валидация metadata и .npy файлов;
- просмотр одного sample;
- визуализация одного sample;
- reproducible stratified split;
- PyTorch Dataset;
- CNN baseline;
- Trace Transformer classifier;
- masked trace self-supervised pre-training;
- fine-tuning Transformer после pre-training;
- расширенные метрики классификации;
- MLflow tracking;
- checkpoint evaluation;
- comparison evaluation reports;
- CLI inference;
- FastAPI inference;
- Docker deployment;
- inference benchmark;
- pytest tests;
- pre-commit;
- GitHub Actions CI.
- CNN multi-task baseline: classification + regression.
- multi-task inference: JSON с predicted_crack_count и регрессионными параметрами среды.
- оценка multi-task checkpoint с classification и regression metrics.
- FastAPI-сервис поддерживает как classification checkpoint, так и multi-task checkpoint. Тип predictor определяется автоматически по model_config.name.
- TorchScript export для classification и multi-task моделей.
- Makefile с основными командами для тестов, обучения, inference, API, Docker, MLflow и экспорта.
- экспорт per-sample multi-task predictions и parity plots для регрессионных параметров.
```

## 22. Ограничения

```text
- данные синтетические;
- всего 665 объектов в текущей MVP-выборке;
- нет отдельного официального test split;
- нет проверки на реальных field data;
- возможен domain gap;
- возможна утечка близких параметрических конфигураций между train и validation;
- Transformer и self-supervised pipeline являются экспериментальными;
- ONNX/TorchScript экспорт пока не реализован.
```

## 23. Планируемые улучшения

```text
1. Добавить официальный test split.

2. Реализовать group split по параметрам моделирования.

3. Добавить regression / multi-task learning:
   - mean_length;
   - length_spread;
   - mean_angle_deg;
   - angle_spread_deg.

4. Добавить аугментации:
   - шум;
   - crop;
   - scaling;
   - receiver dropout.

5. Добавить ONNX или TorchScript export.

6. Добавить DVC remote или другой воспроизводимый способ загрузки данных.

7. Добавить Hugging Face dataset card.

8. Проверить перенос на реальные или более реалистичные данные.
```

## 24. Краткий итог

SeismoNN — это не только модель, а воспроизводимый MLOps-пайплайн для задачи анализа сейсмических откликов трещиноватых сред.

Текущий проект закрывает полный цикл:

```text
данные
  ↓
metadata
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
API
  ↓
Docker
  ↓
CI
```

MVP-задача — классификация количества трещин.

Исследовательское развитие — Transformer encoder, self-supervised pre-training и multi-task восстановление физических параметров трещиноватой среды.
