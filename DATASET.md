# DATASET.md: описание датасета SeismoNN

Этот файл описывает данные, используемые в проекте **SeismoNN**.

Проект использует синтетические сейсмические данные для задачи восстановления параметров трещиноватой среды. В текущей MVP-версии решается задача классификации количества трещин:

```text
3 трещины
4 трещины
5 трещин
```

## 1. Краткое описание

Датасет состоит из `.npy` файлов, полученных в результате компьютерного моделирования сейсмического отклика трещиноватой среды.

Каждый объект — это один сейсмический отклик для конкретной конфигурации среды.

Формат одного объекта:

```text
shape: (2, 1723, 501)
dtype: float32
```

Расшифровка размерностей:

```text
2     — компоненты скорости отражённой волны: vx и vy;
1723  — временные шаги моделирования;
501   — receiver positions / приёмники на поверхности.
```

В текущей версии проекта каждый `.npy` файл соответствует одному sample.

## 2. Источник данных

Публичное хранилище данных:

```text
https://huggingface.co/datasets/FAKIrik/Seismo_datasets
```

Текущий статус страницы Hugging Face:

```text
- общий размер файлов: 17.6 GB;
- dataset viewer недоступен;
- dataset card на Hugging Face пока отсутствует.
```

Поэтому в данном репозитории добавлена собственная карточка датасета `DATASET.md`.

## 3. Почему данные синтетические

Реальные данные для трещиноватых сред часто трудно использовать в supervised learning, потому что точная разметка обычно неизвестна.

Для реальной среды сложно напрямую узнать:

```text
- точное количество трещин;
- среднюю длину трещин;
- разброс длин;
- средний угол трещин;
- разброс углов;
- положение кластера трещин;
- геометрическую структуру трещиноватой области.
```

Компьютерное моделирование позволяет:

```text
- контролируемо задавать параметры среды;
- получать точные labels;
- генерировать данные для supervised learning;
- воспроизводимо сравнивать ML-модели;
- проверять MLOps-пайплайн без доступа к закрытым field data.
```

Главное ограничение синтетических данных:

```text
domain gap
```

То есть модель, обученная на синтетике, может хуже переноситься на реальные полевые данные.

Возможные способы уменьшить domain gap в будущих версиях:

```text
- добавление шума;
- аугментации;
- domain randomization;
- использование более реалистичных симуляций;
- self-supervised pre-training на неразмеченных данных;
- fine-tuning на field data;
- отдельная проверка на реальных данных.
```

## 4. Используемая выборка

В текущей MVP-версии используется выборка из:

```text
665 объектов / .npy файлов
```

Распределение по количеству трещин в исходной выборке:

```text
3 трещины: 228 объектов
4 трещины: 228 объектов
5 трещин: 209 объектов
```

Каждый объект описан в файле:

```text
data/metadata.csv
```

## 5. Формат metadata.csv

`metadata.csv` является основным машинно-читаемым описанием датасета.

Основные колонки:

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

Описание колонок:

| Колонка                 | Описание                                                     |
| ----------------------- | ------------------------------------------------------------ |
| `sample_id`             | Уникальный идентификатор объекта                             |
| `path`                  | Относительный путь к `.npy` файлу                            |
| `filename`              | Имя файла                                                    |
| `split`                 | Разбиение: `train` или `val`                                 |
| `crack_count`           | Количество трещин в среде                                    |
| `class_id`              | Класс для классификации                                      |
| `cluster_center_x`      | x-компонента центра области, где могут располагаться трещины |
| `cluster_center_y`      | y-компонента центра области, где могут располагаться трещины |
| `cluster_half_size_x`   | Половина размера области по x                                |
| `cluster_half_size_y`   | Половина размера области по y                                |
| `mean_length`           | Средняя длина трещин                                         |
| `length_spread`         | Разброс длин трещин                                          |
| `mean_angle_deg`        | Средний угол трещин в градусах                               |
| `angle_spread_deg`      | Разброс углов трещин в градусах                              |
| `shape`                 | Форма `.npy` массива                                         |
| `dtype`                 | Тип данных массива                                           |
| `split_seed`            | Seed, использованный для split                               |
| `split_strategy`        | Стратегия разбиения                                          |
| `split_stratify_column` | Колонка, по которой выполнялась стратификация                |

## 6. Class mapping

Текущая задача — классификация количества трещин.

Используется следующее соответствие:

```text
class_id = 0 → crack_count = 3
class_id = 1 → crack_count = 4
class_id = 2 → crack_count = 5
```

В текущей supervised-задаче target — это `class_id`.

Физически интерпретируемый результат — `crack_count`.

## 7. Пример имени файла

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

## 8. Train/validation split

Для воспроизводимости используется фиксированное стратифицированное разбиение по `class_id`.

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
  --val-size 0.2 \
  --test-size 0.0 \
  --seed 42 \
  --stratify-column class_id
```

После генерации официального split файл `data/metadata_stratified.csv` может быть сохранён как `data/metadata.csv`.

Служебные поля split:

```text
split_seed
split_strategy
split_stratify_column
```

Они позволяют явно восстановить, каким образом было получено разбиение.

## 9. Почему используется stratified split

Классы в датасете близки к сбалансированным, но не идеально:

```text
3 трещины: 228
4 трещины: 228
5 трещин: 209
```

Если использовать полностью случайное разбиение, validation split может получить заметный перекос по классам.

Стратификация по `class_id` нужна, чтобы сохранить похожие пропорции классов в train и validation.

## 10. Ограничения текущего split

Текущий stratified split является улучшением относительно случайного split, но не решает все возможные проблемы.

Потенциальный риск:

```text
утечка близких конфигураций между train и validation
```

Так как данные могут быть получены из параметрической сетки, похожие конфигурации среды могут оказаться в разных split.

Более строгий вариант для будущей версии:

```text
group split
```

Например, можно группировать объекты по параметрам моделирования:

```text
cluster_center_x
cluster_center_y
cluster_half_size_x
cluster_half_size_y
mean_length
length_spread
mean_angle_deg
angle_spread_deg
```

Такой split позволит проверять обобщающую способность модели на менее похожих конфигурациях.

## 11. Проверка metadata.csv

Для проверки структуры `metadata.csv` используется команда:

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

## 12. Просмотр одного объекта

Для анализа одного sample используется команда:

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
- crack_count;
- class_id;
- физические параметры среды;
- shape массива;
- dtype массива;
- min, max, mean, std;
- примерный размер массива в MiB.
```

Сохранить результат в JSON:

```bash
uv run python scripts/inspect_sample.py \
  --metadata data/metadata.csv \
  --data-root . \
  --index 0 \
  --output outputs/sample_inspection.json
```

## Визуализация одного объекта

Для построения heatmap компонент скорости и receiver trace используется команда:

```bash
uv run python scripts/visualize_sample.py \
  --metadata data/metadata.csv \
  --data-root . \
  --index 0 \
  --output-dir outputs/sample_visualization
```

На выходе создаются:

```text
vx_heatmap.png
vy_heatmap.png
vx_receiver_trace.png
vy_receiver_trace.png
sample_info.json
```

Назначение:

```text
- визуальная проверка формы данных;
- проверка наличия осмысленной структуры в сейсмограмме;
- подготовка примеров для README, отчёта или презентации.
```

## 13. Особенности данных

Основные особенности:

```text
1. Большой размер одного объекта.
   Каждый sample имеет форму (2, 1723, 501), то есть содержит более 1.7 млн float32 значений.

2. Небольшое число объектов.
   В текущей MVP-выборке 665 объектов. Это немного для deep learning.

3. Синтетическое происхождение.
   Модель может выучить особенности симулятора, а не универсальные физические закономерности.

4. Возможная неоднозначность обратной задачи.
   Разные конфигурации трещин могут давать похожие сейсмические отклики.

5. Возможная близость объектов из параметрической сетки.
   Это может завышать качество при случайном split.
```

## 14. Риски при обучении моделей

Риски:

```text
- переобучение из-за малого числа объектов;
- завышенная оценка качества из-за похожих train/val конфигураций;
- слабый перенос на реальные данные;
- чувствительность к нормализации;
- высокая стоимость обучения Transformer-моделей;
- возможная физическая неоднозначность предсказаний.
```

Меры снижения рисков в текущем проекте:

```text
- фиксированный seed;
- stratified split;
- metadata validation;
- отдельный validation split;
- несколько метрик классификации;
- сравнение CNN baseline и Transformer;
- MLflow tracking;
- checkpoint evaluation script;
- Docker/CI для воспроизводимости.
```

## 15. Использование датасета в текущем проекте

Данные используются в нескольких сценариях:

```text
1. CNN baseline training
2. Supervised Trace Transformer training
3. Self-supervised masked trace pre-training
4. Fine-tuning Transformer после pre-training
5. CLI inference
6. FastAPI inference
7. Docker inference service
8. Checkpoint evaluation
9. Model comparison
10. Inference benchmark
```

## 16. Что не входит в текущую версию датасета

В текущей MVP-версии отсутствуют:

```text
- реальные полевые данные;
- отдельный официальный test split;
- group split;
- проверка переноса на field data;
- информация о шуме измерений;
- uncertainty labels;
- экспертная разметка реальных геологических объектов.
```

## 17. Планируемые улучшения

Возможные улучшения датасета и data pipeline:

```text
1. Добавить официальный test split.

2. Добавить group split по параметрам моделирования.

3. Добавить data card на Hugging Face.

4. Добавить визуализации примеров:
   - heatmap vx;
   - heatmap vy;
   - отдельные receiver traces.

5. Добавить шумы и аугментации.

6. Добавить больше значений crack_count.

7. Расширить задачу до регрессии:
   - mean_length;
   - length_spread;
   - mean_angle_deg;
   - angle_spread_deg.

8. Подготовить field-data validation subset, если появятся реальные данные.

9. Добавить описание симулятора:
   - физическая модель;
   - граничные условия;
   - параметры сетки;
   - параметры источника;
   - параметры receiver positions.
```

## 18. Краткий вывод

Датасет SeismoNN — это синтетический набор сейсмических откликов для трещиноватых сред.

Текущий датасет подходит для MVP-задачи:

```text
классификация количества трещин: 3 / 4 / 5
```

Его сильные стороны:

```text
- точные labels;
- контролируемые параметры среды;
- воспроизводимость;
- возможность строить supervised и self-supervised пайплайны.
```

Основные ограничения:

```text
- синтетическая природа;
- малое число объектов;
- потенциальный domain gap;
- отсутствие реального test set;
- риск переоценки качества при близких train/val конфигурациях.
```

Поэтому текущий датасет следует рассматривать как первый воспроизводимый benchmark для MLOps-проекта, а не как полностью готовое решение для production-геофизики.
