# SeismoNN: MLOps-система для анализа сейсмограмм трещиноватых сред

**SeismoNN** — учебно-исследовательский MLOps-проект для предсказания параметров трещиноватой среды по синтетическим сейсмическим данным.

Подробное описание датасета, его ограничений и процедуры валидации находится в файле [`DATASET.md`](DATASET.md).

Краткая обновлённая спецификация проекта находится в [`PROJECT.md`](PROJECT.md). Подробное описание датасета — в [`DATASET.md`](DATASET.md).

## Multi-task baseline: классификация + регрессия

Помимо классификации количества трещин, проект поддерживает multi-task baseline.

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

Запуск:

```bash
uv run python scripts/train_multitask.py \
  --config configs/train/cnn_multitask.yaml
```

Для регрессии используется нормализация target-переменных по train split. В checkpoint сохраняется `target_scaler`, чтобы потом можно было восстановить предсказания в физических единицах.

Loss:

```text
total_loss = classification_loss + regression_loss_weight * regression_loss
```

Где:

```text
classification_loss = CrossEntropyLoss
regression_loss = MSELoss по нормализованным regression targets
```

Валидационные метрики:

```text
classification accuracy
classification macro-F1
regression MAE в исходных единицах
regression RMSE в исходных единицах
per-target MAE/RMSE
```

## Multi-task inference

Для multi-task модели используется отдельный inference script:

```bash
uv run python scripts/predict_multitask.py \
  --checkpoint outputs/cnn_multitask_50ep/best.pt \
  --input 2nd_selection/<sample_name>.npy \
  --device cpu \
  --output outputs/cnn_multitask_50ep/sample_multitask_prediction.json
```

Модель возвращает JSON с классификационным и регрессионным результатом:

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

Поле `expected_crack_count` считается как математическое ожидание по вероятностям классов:

```text
E[crack_count] = 3 * P(3) + 4 * P(4) + 5 * P(5)
```

Регрессионные значения возвращаются в исходных физических единицах, потому что в checkpoint сохраняется `target_scaler`.

## Multi-task checkpoint evaluation

Для оценки multi-task модели на validation split используется:

```bash
uv run python scripts/evaluate_multitask_checkpoint.py \
  --checkpoint outputs/cnn_multitask_50ep/best.pt \
  --metadata data/metadata.csv \
  --split val \
  --batch-size 8 \
  --device cpu \
  --output outputs/cnn_multitask_50ep/evaluation_val.json
```

Отчёт содержит классификационные метрики:

```text
accuracy
balanced accuracy
macro precision
macro recall
macro-F1
confusion matrix
classification report
```

И регрессионные метрики:

```text
mean MAE
mean RMSE
per-target MAE/RMSE:
  mean_length
  length_spread
  mean_angle_deg
  angle_spread_deg
```

Такой отчёт можно использовать в `scripts/compare_evaluations.py`, потому что для классификационной части в JSON также сохраняются alias-поля:

```text
accuracy
balanced_accuracy
macro_precision
macro_recall
macro_f1
```

## 1. Практическая мотивация

Трещиноватые среды встречаются в задачах геофизики, инженерной геологии и анализа подземных структур. Наличие, ориентация и характер трещин могут влиять на распространение волн в среде. Поэтому по сейсмическому отклику можно пытаться восстанавливать параметры среды.

Практическая ценность проекта:

```text
- автоматизация анализа сейсмических откликов;
- быстрое получение первичной оценки параметров трещиноватой среды;
- возможность сравнивать разные ML-подходы на воспроизводимом пайплайне;
- подготовка инфраструктуры для дальнейших экспериментов с CNN, Transformer и self-supervised pre-training.
```

В рамках курса MLOps основной акцент сделан не только на качестве модели, но и на воспроизводимом ML-пайплайне:

```text
данные → metadata.csv → split → Dataset → обучение → метрики → checkpoint → CLI/API inference → Docker → CI
```

## 2. Постановка задачи

### Текущая задача MVP

На вход подаётся `.npy` файл с сейсмическим откликом среды.

Модель должна предсказать количество трещин в среде:

```text
3, 4 или 5
```

Это задача многоклассовой классификации.

### Почему MVP отличается от полной исследовательской постановки

Изначальная исследовательская цель — восстановление нескольких параметров трещиноватой среды, включая среднюю длину и угол трещин.

Однако для первой устойчивой MLOps-версии выбрана более простая и проверяемая постановка:

```text
классификация количества трещин
```

Причины:

```text
- для этой задачи уже есть размеченный датасет;
- target является дискретным и хорошо подходит для baseline-модели;
- можно построить воспроизводимое train/validation разбиение;
- можно использовать понятные метрики классификации;
- проще проверить полный MLOps-контур: обучение, сохранение модели, CLI, API, Docker и CI.
```

После стабилизации MVP проект можно расширить до регрессии и multi-task learning.

## 3. Формат входных данных

Каждый объект датасета — это `.npy` файл с тензором:

```text
shape: (2, 1723, 501)
dtype: float32
```

Расшифровка размерностей:

```text
2     — две компоненты скорости отражённой волны: vx и vy;
1723  — количество временных шагов в моделировании;
501   — количество приёмников / receiver positions на поверхности.
```

Один входной тензор содержит:

```text
2 * 1723 * 501 = 1 726 446 float32 значений
```

Оценочный размер одного объекта в памяти:

```text
1 726 446 * 4 bytes ≈ 6.6 MiB
```

В исходном ТЗ указывалась форма `(2, 1733, 501)`, однако после проверки всех файлов в текущем датасете через `metadata.csv` фактическая форма оказалась:

```text
(2, 1723, 501)
```

В будущих версиях можно добавить поддержку переменной длины временной оси через:

```text
- crop;
- padding;
- resampling;
- temporal CNN frontend;
- архитектуры, не завязанные на фиксированное число временных шагов.
```

## 4. Формат выходных данных

Модель возвращает JSON с предсказанным классом и вероятностями классов.

Пример ответа:

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
model_name                 — имя модели;
input_path                 — путь к входному .npy файлу;
predicted_class_id         — предсказанный class_id;
predicted_crack_count      — предсказанное количество трещин;
class_probabilities        — вероятности классов после softmax;
checkpoint_path            — checkpoint, из которого загружена модель.
```

## 5. Пример входного файла

Пример имени файла из датасета:

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

В текущей задаче классификации target — это только `crack_count`.

Остальные параметры сохраняются в `metadata.csv` и могут быть использованы в будущих регрессионных постановках.

## 6. Датасет

Данные являются результатом компьютерного моделирования сейсмических откликов для сред с различными параметрами трещин.

Публичное хранилище данных:

```text
https://huggingface.co/datasets/FAKIrik/Seismo_datasets
```

Текущее состояние Hugging Face страницы:

```text
- общий размер файлов: 17.6 GB;
- dataset viewer недоступен, потому что Hugging Face не распознал поддерживаемые data files;
- dataset card пока отсутствует.
```

Используемая в MVP выборка:

```text
665 объектов / .npy файлов
```

Распределение по классам в исходной выборке:

```text
3 трещины: 228 объектов
4 трещины: 228 объектов
5 трещин: 209 объектов
```

Каждый объект описан в файле:

```text
data/metadata.csv
```

Основные колонки `metadata.csv`:

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

Проверить датасет можно командой:

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

## 7. Почему используются синтетические данные

Данные синтетические, потому что для реальных трещиноватых сред сложно получить точную разметку:

```text
- точное количество трещин обычно неизвестно;
- длины трещин и их углы не наблюдаются напрямую;
- положение трещин в реальной среде часто доступно только косвенно;
- реальные данные могут быть дорогими, закрытыми или шумными.
```

Компьютерное моделирование позволяет:

```text
- контролируемо варьировать параметры среды;
- получать точные labels;
- создавать датасеты для supervised learning;
- проводить воспроизводимые эксперименты.
```

Главный риск такого подхода:

```text
domain gap
```

То есть модель, обученная на синтетических данных, может хуже работать на реальных полевых данных. В будущих версиях этот риск можно снижать с помощью:

```text
- добавления шума и аугментаций;
- domain randomization;
- self-supervised pre-training;
- fine-tuning на реальных или более реалистичных данных;
- отдельной проверки на field data.
```

## 8. Особенности и сложности данных

Основные сложности:

```text
1. Небольшое число объектов.
   В текущей MVP-выборке 665 сэмплов, что мало для больших deep learning моделей.

2. Большой размер одного объекта.
   Каждый входной тензор имеет форму (2, 1723, 501), поэтому нельзя неаккуратно загружать весь датасет в память.

3. Возможная неоднозначность обратной задачи.
   Разные конфигурации трещин могут давать похожий сейсмический отклик.

4. Риск утечки через параметрическую сетку.
   Если данные сгенерированы декартовым произведением параметров, близкие конфигурации могут попадать одновременно в train и validation.

5. Синтетическая природа данных.
   Модель может переобучиться на особенности симулятора.
```

В текущей реализации используются:

```text
- metadata.csv для описания данных;
- lazy loading .npy файлов;
- np.load(..., mmap_mode="r") при загрузке;
- фиксированное стратифицированное разбиение;
- тесты без загрузки полного датасета.
```

## 9. Разбиение на train/validation

Для воспроизводимости используется фиксированное стратифицированное разбиение по `class_id`.

Команда для генерации split:

```bash
uv run python scripts/create_split.py \
  --input data/metadata.csv \
  --output data/metadata_stratified.csv \
  --val-size 0.2 \
  --test-size 0.0 \
  --seed 42 \
  --stratify-column class_id
```

Текущее разбиение:

```text
train: 532 объекта
val:   133 объекта
total: 665 объектов
```

В `metadata.csv` сохраняются служебные поля:

```text
split_seed
split_strategy
split_stratify_column
```

Это позволяет воспроизвести эксперимент и явно понимать, как было получено разбиение.

Проверка распределения классов:

```bash
uv run python - <<'PY'
import pandas as pd

df = pd.read_csv("data/metadata.csv")

print(df.groupby(["split", "crack_count"]).size())
print()
print(df[["split_seed", "split_strategy", "split_stratify_column"]].drop_duplicates())
PY
```

В будущей версии стоит добавить более строгий `group split` по параметрам моделирования, чтобы близкие конфигурации не попадали одновременно в train и validation.

## 10. Метрики

Так как текущая MVP-задача является задачей классификации, используются следующие метрики:

```text
1. Cross-Entropy Loss
2. Accuracy
3. Balanced Accuracy
4. Macro Precision
5. Macro Recall
6. Macro-F1
7. Confusion Matrix
```

Почему нужны не только Accuracy:

```text
- Accuracy показывает общую долю правильных ответов;
- Macro-F1 одинаково учитывает каждый класс;
- Macro Precision показывает качество положительных предсказаний по классам;
- Macro Recall показывает, насколько хорошо модель находит каждый класс;
- Balanced Accuracy полезна при дисбалансе классов;
- Confusion Matrix показывает, какие классы модель путает.
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
улучшение метрик относительно CNN baseline.
```

Значения являются ориентирами, а не гарантией, так как задача исследовательская, а датасет небольшой.

Для будущей регрессионной постановки будут использоваться:

```text
- MAE;
- RMSE;
- R²;
- ошибка по углу в градусах;
- относительная ошибка длины трещин.
```

Для внедрения также планируется оценивать:

```text
- latency p50;
- latency p95;
- throughput;
- размер модели;
- потребление памяти.
```

## 11. Baseline-модель

Текущий baseline — CNN-классификатор.

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

где 3 — количество классов:

```text
3, 4, 5 трещин
```

Почему используется `AdaptiveAvgPool2d`:

```text
- модель не требует заранее вычислять размер flatten-вектора;
- архитектура становится устойчивее к небольшим изменениям пространственно-временной размерности;
- проще использовать модель в inference.
```

## 12. Пайплайн обучения baseline

Полный pipeline обучения:

```text
.npy файл
  ↓
metadata.csv
  ↓
SeismoDataset
  ↓
проверка shape/dtype
  ↓
нормализация одного sample
  ↓
DataLoader
  ↓
CNN baseline
  ↓
CrossEntropyLoss
  ↓
Adam / AdamW
  ↓
валидация на stratified split
  ↓
выбор best checkpoint по val_macro_f1
  ↓
сохранение артефактов
```

Артефакты обучения:

```text
outputs/cnn_baseline/best.pt
outputs/cnn_baseline/last.pt
outputs/cnn_baseline/history.csv
outputs/cnn_baseline/metrics.json
outputs/cnn_baseline/loss.png
outputs/cnn_baseline/accuracy.png
outputs/cnn_baseline/macro_f1.png
outputs/cnn_baseline/confusion_matrix.png
```

Директория `outputs/` не коммитится в git.

## 13. Transformer encoder model

В проект добавлена supervised Transformer encoder модель `trace_transformer`.

Она вдохновлена идеей StorSeismic: сейсмические трассы рассматриваются как последовательность токенов, а self-attention моделирует зависимости между receiver-трассами.

Текущая реализация пока не включает self-supervised pre-training. Сейчас Transformer обучается supervised способом на задачу классификации количества трещин.

Модель вдохновлена статьёй:

```text
StorSeismic: A new paradigm in deep learning for seismic processing
https://arxiv.org/abs/2205.00222
```

Идея адаптации:

```text
x имеет форму [B, 2, T, R]

B — batch size
2 — компоненты скорости
T — временные шаги
R — receiver positions
```

Один возможный вариант токенизации:

```text
1. Каждый receiver / trace рассматривается как токен.
2. Временная ось сжимается temporal CNN frontend.
3. Получается последовательность признаков [B, R, d_model].
4. Добавляется позиционное кодирование по receiver index.
5. TransformerEncoder моделирует зависимости между трассами.
6. Classification head предсказывает количество трещин.
```

Планируемый self-supervised pre-training:

```text
1. Маскировать часть receiver-трасс.
2. Обучать encoder восстанавливать замаскированные трассы или их признаки.
3. После pre-training дообучать модель на supervised задачу классификации.
```

Такая схема полезна, потому что полевые данные могут быть неразмеченными, но всё равно могут использоваться для self-supervised обучения.

## Self-supervised pre-training

В проект добавлен экспериментальный pipeline self-supervised pre-training для Transformer.

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

Запуск:

```bash
uv run python scripts/pretrain_transformer.py \
  --config configs/pretrain/trace_transformer.yaml
```

Артефакты сохраняются в:

```text
outputs/masked_trace_pretraining/
```

Основные артефакты:

```text
best.pt
last.pt
history.csv
metrics.json
```

Текущая реализация pre-training является первым шагом. Следующий шаг — использовать веса предобученного encoder для supervised fine-tuning на задачу классификации количества трещин.

## Fine-tuning после pre-training

После self-supervised pre-training можно загрузить веса encoder-а в supervised Transformer-классификатор.

Схема:

```text
outputs/masked_trace_pretraining/best.pt
  ↓
temporal_encoder + Transformer encoder weights
  ↓
TraceTransformerClassifier
  ↓
supervised fine-tuning на crack_count
```

Запуск fine-tuning:

```bash
uv run python scripts/train.py \
  --config configs/train/transformer_finetune.yaml
```

Веса загружаются из блока конфига:

```yaml
pretrained:
  enabled: true
  checkpoint_path: outputs/masked_trace_pretraining/best.pt
  prefixes:
    - temporal_encoder.
    - encoder.
  min_loaded_keys: 1
```

В fine-tuning checkpoint сохраняется информация о переносе весов:

```text
pretrained_transfer.json
```

Текущая реализация переносит только совместимые веса:

```text
- temporal_encoder.*
- encoder.*
```

Голова реконструкции из pre-training не переносится, потому что в supervised задаче используется classification head.

## 14. Внедрение

Текущие способы использования модели:

```text
1. Python-пакет seismonn
2. CLI inference через scripts/predict.py
3. FastAPI HTTP service
4. Docker container
```

Формат модели:

```text
PyTorch checkpoint: best.pt
```

ONNX и TorchScript пока не используются, но могут быть добавлены позже.

### Ресурсы для обучения

Для CNN baseline:

```text
CPU: возможно, но медленно
GPU: желательно
RAM: зависит от batch_size
```

Для Transformer:

```text
GPU: желательно / практически необходимо
VRAM: зависит от d_model, числа heads, числа слоёв и batch_size
```

### Ресурсы для inference

Для одиночного inference CNN baseline:

```text
CPU: достаточно
GPU: опционально
```

Для production-подобного запуска доступен FastAPI сервис и Docker image.

## 15. Технологический стек

Основной стек:

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
Pytest
pre-commit
Ruff
GitHub Actions
uv
```

Назначение компонентов:

```text
PyTorch       — обучение и inference нейронной сети
NumPy         — работа с .npy данными
Pandas        — metadata.csv и таблицы
Scikit-learn  — метрики и split
FastAPI       — HTTP inference service
Docker        — контейнеризация API
Pytest        — тестирование
pre-commit    — проверки перед коммитом
Ruff          — линтинг и форматирование
GitHub Actions — CI
uv            — управление зависимостями
```

## 16. Установка

Проект использует `uv`.

Установка зависимостей:

```bash
uv sync
```

Установка с dev-зависимостями:

```bash
uv sync --all-extras --dev
```

Все команды рекомендуется запускать из корня репозитория.

## 17. Генерация metadata.csv

Если доступны `2nd_sel.json` и директория `2nd_selection/`, metadata можно пересоздать командой:

```bash
uv run python scripts/build_metadata.py \
  --split-json 2nd_sel.json \
  --data-dir 2nd_selection \
  --output data/metadata.csv \
  --test-split-name val \
  --validate-files
```

Проверка metadata:

```bash
uv run python - <<'PY'
import pandas as pd

df = pd.read_csv("data/metadata.csv")

print(df.head())
print()
print(df.columns.tolist())
print()
print(df.groupby(["split", "crack_count"]).size())
print()
print(df["shape"].value_counts())
PY
```

## Валидация metadata и датасета

Для проверки структуры `metadata.csv` используется скрипт:

```bash
uv run python scripts/validate_metadata.py \
  --metadata data/metadata.csv \
  --data-root . \
  --expected-shape 2 1723 501 \
  --expected-dtype float32 \
  --expected-splits train val
```

Скрипт проверяет:

```text
- наличие обязательных колонок;
- отсутствие дубликатов sample_id, path, filename;
- корректность split;
- соответствие class_id и crack_count;
- shape из metadata.csv;
- dtype из metadata.csv.
```

Если локально доступна директория с `.npy` файлами, можно проверить сами файлы:

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

В режиме `--validate-files` дополнительно проверяется:

```text
- существование каждого .npy файла;
- реальный shape массива;
- реальный dtype массива.
```

## 18. Анализ одного примера данных

Для просмотра конкретного объекта из `metadata.csv` можно использовать:

```bash
uv run python scripts/inspect_sample.py \
  --metadata data/metadata.csv \
  --data-root . \
  --index 0
```

Скрипт выводит:

```text
- sample_id;
- путь к .npy файлу;
- split;
- crack_count и class_id;
- параметры среды из metadata.csv;
- shape и dtype массива;
- min, max, mean, std;
- оценочный размер массива в MiB.
```

Сохранить результат в JSON:

```bash
uv run python scripts/inspect_sample.py \
  --metadata data/metadata.csv \
  --data-root . \
  --index 0 \
  --output outputs/sample_inspection.json
```

## Визуализация одного примера данных

Для визуальной проверки сейсмограммы можно использовать:

```bash
uv run python scripts/visualize_sample.py \
  --metadata data/metadata.csv \
  --data-root . \
  --index 0 \
  --output-dir outputs/sample_visualization
```

Скрипт сохраняет:

```text
vx_heatmap.png
vy_heatmap.png
vx_receiver_trace.png
vy_receiver_trace.png
sample_info.json
```

Также можно ограничить число временных шагов и receiver positions только для построения графиков:

```bash
uv run python scripts/visualize_sample.py \
  --metadata data/metadata.csv \
  --data-root . \
  --index 0 \
  --output-dir outputs/sample_visualization_small \
  --max-time-steps 400 \
  --max-receivers 200
```

Эта визуализация нужна для sanity-check данных и демонстрации конкретного входного объекта.

## 19. Обучение CNN baseline

Запуск обучения:

```bash
uv run python scripts/train.py --config configs/train/cnn.yaml
```

Конфиг обучения:

```text
configs/train/cnn.yaml
```

Основные параметры:

```text
seed
device
batch_size
num_workers
normalize
num_epochs
learning rate
weight decay
output_dir
```

После обучения лучший checkpoint сохраняется в:

```text
outputs/cnn_baseline/best.pt
```

## 20. CLI inference

Запуск предсказания для одного `.npy` файла:

```bash
uv run python scripts/predict.py \
  --checkpoint outputs/cnn_baseline/best.pt \
  --input 2nd_selection/<sample_name>.npy
```

Сохранение результата в JSON:

```bash
uv run python scripts/predict.py \
  --checkpoint outputs/cnn_baseline/best.pt \
  --input 2nd_selection/<sample_name>.npy \
  --output outputs/sample_prediction.json
```

Получить пример пути из metadata:

```bash
uv run python - <<'PY'
import pandas as pd

df = pd.read_csv("data/metadata.csv")
print(df.iloc[0]["path"])
PY
```

## 21. FastAPI inference service

Запуск API локально:

```bash
SEISMONN_CHECKPOINT=outputs/cnn_baseline/best.pt \
uv run uvicorn seismonn.api.main:app --host 127.0.0.1 --port 8000
```

Проверка состояния сервиса:

```bash
curl -s http://127.0.0.1:8000/health | python -m json.tool
```

Отправка `.npy` файла:

```bash
curl -s -X POST \
  http://127.0.0.1:8000/predict \
  -F "file=@2nd_selection/<sample_name>.npy" \
  | python -m json.tool
```

Endpoint-ы:

```text
GET  /health   — проверка состояния сервиса и загрузки модели
POST /predict  — загрузка .npy файла и получение JSON-предсказания
```

### API для multi-task checkpoint

FastAPI-сервис автоматически определяет тип checkpoint.

Если используется `cnn_multitask`, endpoint `/predict` возвращает не только классификацию количества трещин, но и регрессионные параметры среды:

```text
mean_length
length_spread
mean_angle_deg
angle_spread_deg
```

Запуск API с multi-task checkpoint:

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

Для Docker:

```bash
docker run --rm \
  -p 8000:8000 \
  -e SEISMONN_CHECKPOINT=/app/checkpoints/best.pt \
  -e SEISMONN_DEVICE=cpu \
  -e SEISMONN_PREDICTOR_TYPE=auto \
  -v "$(pwd)/outputs/cnn_multitask_50ep/best.pt:/app/checkpoints/best.pt:ro" \
  seismonn-api
```

## 22. Docker

Сборка Docker image:

```bash
docker build -t seismonn-api .
```

Запуск контейнера с примонтированным checkpoint:

```bash
docker run --rm \
  -p 8000:8000 \
  -e SEISMONN_CHECKPOINT=/app/checkpoints/best.pt \
  -e SEISMONN_DEVICE=cpu \
  -v "$(pwd)/outputs/cnn_baseline/best.pt:/app/checkpoints/best.pt:ro" \
  seismonn-api
```

Проверка сервиса:

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

Запуск контейнера в фоне:

```bash
docker run -d --name seismonn-api \
  -p 8000:8000 \
  -e SEISMONN_CHECKPOINT=/app/checkpoints/best.pt \
  -e SEISMONN_DEVICE=cpu \
  -v "$(pwd)/outputs/cnn_baseline/best.pt:/app/checkpoints/best.pt:ro" \
  seismonn-api
```

Остановка контейнера:

```bash
docker stop seismonn-api
```

## 23. Benchmark инференса

Для оценки скорости инференса используется скрипт:

```bash
uv run python scripts/benchmark_inference.py \
  --checkpoint outputs/cnn_baseline/best.pt \
  --input 2nd_selection/<sample_name>.npy \
  --device cpu \
  --warmup-runs 3 \
  --timed-runs 20 \
  --output outputs/inference_benchmark.json
```

Скрипт измеряет два режима:

```text
model_only — только forward pass модели на заранее загруженном tensor;
end_to_end — загрузка .npy, preprocessing, inference и формирование JSON-ответа.
```

Для каждого режима сохраняются:

```text
mean latency
std latency
min latency
p50 latency
p95 latency
p99 latency
max latency
throughput samples/sec
```

## TorchScript export

Помимо PyTorch checkpoint, модель можно экспортировать в TorchScript.

Экспорт classification checkpoint:

```bash
uv run python scripts/export_torchscript.py \
  --checkpoint outputs/cnn_baseline/best.pt \
  --output outputs/cnn_baseline/model_torchscript.pt \
  --device cpu
```

Экспорт multi-task checkpoint:

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

Для multi-task модели TorchScript возвращает tuple:

```text
logits:     [B, 3]
regression: [B, 4]
```

Метаданные экспорта содержат:

```text
model_name
input_shape
output_format
class_id_to_crack_count
regression_columns
target_scaler
```

## 24. Тесты

Запуск всех тестов:

```bash
uv run pytest
```

Запуск отдельных тестов:

```bash
uv run pytest \
  tests/test_dataset.py \
  tests/test_cnn.py \
  tests/test_evaluate.py \
  tests/test_predictor.py \
  tests/test_api.py \
  tests/test_splits.py
```

Тесты не требуют полного датасета. Для unit-тестов используются маленькие синтетические `.npy` массивы.

## 25. Code quality и pre-commit

Запуск pre-commit:

```bash
uv run pre-commit run --all-files
```

Если pre-commit автоматически изменил файлы:

```bash
git add .
uv run pre-commit run --all-files
```

Используемые проверки:

```text
ruff
ruff-format
trailing-whitespace
end-of-file-fixer
check-yaml
```

## 26. Continuous Integration

Проект использует GitHub Actions для CI.

Workflow:

```text
.github/workflows/ci.yaml
```

CI запускается на push и pull request.

Проверки:

```text
pre-commit checks
pytest tests
Docker image build
```

Основные команды CI:

```bash
uv sync --all-extras --dev --frozen
uv run pre-commit run --all-files
uv run pytest
docker build -t seismonn-api:ci .
```

## MLflow tracking

Проект поддерживает логирование экспериментов через MLflow.

MLflow включается в конфиге обучения:

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
- train/validation метрики по эпохам;
- финальные метрики;
- best.pt и last.pt;
- history.csv;
- metrics.json;
- графики loss, accuracy, macro-F1;
- confusion_matrix.png.
```

Запуск обучения:

```bash
uv run python scripts/train.py --config configs/train/cnn.yaml
```

Запуск MLflow UI:

```bash
uv run mlflow ui --backend-store-uri mlruns
```

После запуска UI доступен по адресу:

```text
http://127.0.0.1:5000
```

## Оценка сохранённого checkpoint

Для независимой оценки сохранённой модели используется скрипт:

```bash
uv run python scripts/evaluate_checkpoint.py \
  --checkpoint outputs/cnn_baseline/best.pt \
  --metadata data/metadata.csv \
  --split val \
  --batch-size 16 \
  --device cpu \
  --output outputs/cnn_baseline/evaluation_val.json
```

Скрипт загружает checkpoint, восстанавливает модель по `model_config`, создаёт `SeismoDataset` для выбранного split и считает метрики классификации.

В отчёт входят:

```text
- loss;
- accuracy;
- balanced accuracy;
- macro precision;
- macro recall;
- macro-F1;
- confusion matrix;
- classification report по классам 3 / 4 / 5.
```

Такой скрипт позволяет одинаково оценивать разные модели:

```text
CNN baseline
supervised Trace Transformer
fine-tuned Trace Transformer
```

## Сравнение нескольких моделей

После сохранения evaluation JSON для разных моделей можно собрать сводную таблицу:

```bash
uv run python scripts/compare_evaluations.py \
  --reports \
    outputs/cnn_baseline/evaluation_val.json \
    outputs/trace_transformer/evaluation_val.json \
    outputs/trace_transformer_finetuned/evaluation_val.json \
  --output-csv outputs/model_comparison.csv \
  --output-md outputs/model_comparison.md
```

Если какой-то модели ещё нет, её report можно убрать из списка.

Скрипт строит таблицу с метриками:

```text
- loss;
- accuracy;
- balanced accuracy;
- macro precision;
- macro recall;
- macro-F1.
```

По умолчанию таблица сортируется по `macro_f1` в порядке убывания.

## 27. Структура проекта

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
│   ├── create_split.py
│   ├── predict.py
│   └── train.py
├── seismonn/
│   ├── api/
│   │   └── main.py
│   ├── data/
│   │   ├── dataset.py
│   │   └── splits.py
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

## 28. Что уже реализовано

```text
Metadata-based dataset description
Reproducible stratified split
PyTorch Dataset
CNN baseline model
YAML training config
Training script
Validation metrics
Best checkpoint saving
CLI inference
FastAPI inference service
Docker deployment
Pytest tests
pre-commit checks
GitHub Actions CI
Extended classification metrics
Dataset sample inspection script
Inference benchmark script
MLflow experiment tracking
Trace Transformer classifier
Model factory for CNN and Transformer
Transformer config
Pretrained encoder loading for Transformer fine-tuning
Checkpoint evaluation script
Model cmparison from evaluation reports
Dataset card / DATASET.md
Sample visualization script
CNN multi-task baseline: classification + regression
Universal API inference for classification and multi-task checkpoints
```

## 29. Что планируется добавить

```text
1. Group split:
   более строгая проверка качества без утечки близких конфигураций.

2. Multi-task learning:
   классификация количества трещин + регрессия длины и углов.
```

## 30. Ограничения текущей версии

```text
- Данные синтетические.
- Количество объектов небольшое: 665.
- Текущая модель — baseline CNN, а не финальная исследовательская модель.
- Реальный перенос на полевые данные пока не проверялся.
- Transformer и self-supervised pre-training пока находятся в плане.
- ONNX/TorchScript экспорт пока не реализован.
- Latency/throughput пока не измеряются отдельным benchmark-скриптом.
```

## 31. Краткий вывод

Текущая версия SeismoNN — это воспроизводимый MLOps baseline для задачи классификации количества трещин по синтетическим сейсмограммам.

Основной результат MVP:

```text
metadata.csv
  ↓
reproducible split
  ↓
Dataset
  ↓
CNN training
  ↓
best.pt
  ↓
CLI inference
  ↓
FastAPI inference
  ↓
Docker deployment
  ↓
CI checks
```

Следующий исследовательский шаг — перейти от CNN baseline к Transformer encoder и self-supervised pre-training, а также расширить задачу до регрессии физических параметров трещиноватой среды.
