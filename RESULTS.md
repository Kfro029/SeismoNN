# RESULTS.md: результаты экспериментов SeismoNN

Этот файл содержит сводку фактических результатов обучения, оценки и инференса.

Файл можно пересоздать командой:

```bash
uv run python scripts/generate_results_report.py --output RESULTS.md
```

## Сравнение моделей

| model | model_name | split | num_samples | loss | accuracy | balanced_accuracy | macro_f1 | regression_mae_mean | regression_rmse_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cnn_multitask_50ep | cnn_multitask | val | 133 | 1.4696 | 0.5789 | 0.5764 | 0.5396 | 3.1667 | 4.1610 |
| cnn_baseline | cnn_baseline | val | 133 | 1.1022 | 0.3910 | 0.3571 | 0.2286 | — | — |
| trace_transformer_finetuned | trace_transformer | val | 133 | 1.0957 | 0.4135 | 0.3333 | 0.1950 | — | — |
| trace_transformer | trace_transformer | val | 133 | 1.0915 | 0.3759 | 0.3333 | 0.1821 | — | — |

Использованные evaluation reports:

- `outputs/cnn_baseline/evaluation_val.json`
- `outputs/trace_transformer/evaluation_val.json`
- `outputs/trace_transformer_finetuned/evaluation_val.json`
- `outputs/cnn_multitask_50ep/evaluation_val.json`

## Multi-task regression analysis

| summary | num_samples | classification_accuracy | classification_macro_f1 | regression_mae_mean | regression_rmse_mean |
| --- | --- | --- | --- | --- | --- |
| cnn_multitask_50ep | 133 | 0.5789 | 0.5396 | 3.1667 | 4.1610 |

### `outputs/cnn_multitask_50ep/predictions_summary_val.json`

| target | mae | rmse | max_abs_error |
| --- | --- | --- | --- |
| mean_length | 4.4782 | 5.2856 | 10.9201 |
| length_spread | 1.4777 | 1.7072 | 2.3680 |
| mean_angle_deg | 3.3359 | 4.3619 | 16.6123 |
| angle_spread_deg | 3.3748 | 4.4021 | 16.9471 |

## Inference benchmark

| benchmark | device | model_only_p50_ms | model_only_p95_ms | model_only_throughput | end_to_end_p50_ms | end_to_end_p95_ms | end_to_end_throughput |
| --- | --- | --- | --- | --- | --- | --- | --- |
| outputs | cpu | 85.5119 | 96.9264 | 11.8267 | 91.7476 | 100.0998 | 11.1848 |

## Выводы

Основные выводы следует интерпретировать с учётом ограничений датасета:

```text
- данные синтетические;
- объектов немного: 665;
- validation split стратифицирован, но пока не является group split;
- перенос на реальные field data не проверялся.
```

Фактические численные выводы зависят от запущенных экспериментов и доступных файлов в `outputs/`.
