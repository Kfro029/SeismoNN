SHELL := /bin/bash

UV := uv
PYTHON := uv run python

HOST ?= 127.0.0.1
PORT ?= 8000
DEVICE ?= cpu

CNN_CONFIG ?= configs/train/cnn.yaml
TRANSFORMER_CONFIG ?= configs/train/transformer.yaml
TRANSFORMER_FINETUNE_CONFIG ?= configs/train/transformer_finetune.yaml
MULTITASK_CONFIG ?= configs/train/cnn_multitask.yaml
PRETRAIN_CONFIG ?= configs/pretrain/trace_transformer.yaml

CNN_CKPT ?= outputs/cnn_baseline/best.pt
TRANSFORMER_CKPT ?= outputs/trace_transformer/best.pt
TRANSFORMER_FINETUNED_CKPT ?= outputs/trace_transformer_finetuned/best.pt
MULTITASK_CKPT ?= outputs/cnn_multitask_50ep/best.pt
PRETRAIN_CKPT ?= outputs/masked_trace_pretraining/best.pt

SAMPLE ?= $(shell uv run python -c 'import pandas as pd; print(pd.read_csv("data/metadata.csv").iloc[0]["path"])')

.PHONY: help sync test lint check \
	metadata split validate-metadata validate-files inspect-sample visualize-sample \
	train-cnn train-transformer train-transformer-finetuned train-multitask pretrain-transformer \
	evaluate-cnn evaluate-transformer evaluate-transformer-finetuned evaluate-multitask compare \
	predict predict-multitask api api-multitask docker-build docker-run docker-run-multitask \
	mlflow-ui benchmark-cnn export-torchscript-cnn export-torchscript-multitask sample-path clean-cache \
	export-multitask-predictions
	results

help:
	@echo "SeismoNN Makefile commands"
	@echo ""
	@echo "Environment:"
	@echo "  make sync                         Install/sync dependencies"
	@echo "  make test                         Run pytest"
	@echo "  make lint                         Run pre-commit"
	@echo "  make check                        Run lint + tests"
	@echo ""
	@echo "Data:"
	@echo "  make metadata                     Build data/metadata.csv from 2nd_sel.json"
	@echo "  make split                        Create reproducible stratified split"
	@echo "  make validate-metadata            Validate metadata.csv only"
	@echo "  make validate-files               Validate metadata.csv and referenced .npy files"
	@echo "  make inspect-sample               Print JSON info for one sample"
	@echo "  make visualize-sample             Save plots for one sample"
	@echo "  make sample-path                  Print default sample path from metadata.csv"
	@echo ""
	@echo "Training:"
	@echo "  make train-cnn                    Train CNN classification baseline"
	@echo "  make train-transformer            Train supervised Trace Transformer"
	@echo "  make pretrain-transformer         Run self-supervised masked trace pretraining"
	@echo "  make train-transformer-finetuned  Fine-tune Transformer from pretraining checkpoint"
	@echo "  make train-multitask              Train CNN multi-task baseline"
	@echo ""
	@echo "Evaluation:"
	@echo "  make evaluate-cnn                 Evaluate CNN checkpoint"
	@echo "  make evaluate-transformer         Evaluate Transformer checkpoint"
	@echo "  make evaluate-transformer-finetuned Evaluate fine-tuned Transformer checkpoint"
	@echo "  make evaluate-multitask           Evaluate multi-task checkpoint"
	@echo "  make compare                      Compare available evaluation reports"
	@echo ""
	@echo "Inference:"
	@echo "  make predict                      Classification CLI prediction"
	@echo "  make predict-multitask            Multi-task CLI prediction"
	@echo "  make api                          Run FastAPI with classification checkpoint"
	@echo "  make api-multitask                Run FastAPI with multi-task checkpoint"
	@echo ""
	@echo "Deployment:"
	@echo "  make docker-build                 Build Docker image"
	@echo "  make docker-run                   Run Docker API with classification checkpoint"
	@echo "  make docker-run-multitask         Run Docker API with multi-task checkpoint"
	@echo "  make export-torchscript-cnn       Export CNN checkpoint to TorchScript"
	@echo "  make export-torchscript-multitask Export multi-task checkpoint to TorchScript"
	@echo ""
	@echo "Tracking and benchmarks:"
	@echo "  make mlflow-ui                    Run MLflow UI"
	@echo "  make benchmark-cnn                Benchmark CNN inference"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean-cache                  Remove local Python/test caches"
	@echo "  make export-multitask-predictions Export per-sample multi-task predictions"
	@echo "  make results                      Generate RESULTS.md from output artifacts"

sync:
	$(UV) sync --all-extras --dev

test:
	$(UV) run pytest

lint:
	$(UV) run pre-commit run --all-files

check: lint test

metadata:
	$(PYTHON) scripts/build_metadata.py \
		--split-json 2nd_sel.json \
		--data-dir 2nd_selection \
		--output data/metadata.csv \
		--test-split-name val \
		--validate-files

split:
	$(PYTHON) scripts/create_split.py \
		--input data/metadata.csv \
		--output data/metadata_stratified.csv \
		--val-size 0.2 \
		--test-size 0.0 \
		--seed 42 \
		--stratify-column class_id

validate-metadata:
	$(PYTHON) scripts/validate_metadata.py \
		--metadata data/metadata.csv \
		--data-root . \
		--expected-shape 2 1723 501 \
		--expected-dtype float32 \
		--expected-splits train val

validate-files:
	$(PYTHON) scripts/validate_metadata.py \
		--metadata data/metadata.csv \
		--data-root . \
		--expected-shape 2 1723 501 \
		--expected-dtype float32 \
		--expected-splits train val \
		--validate-files \
		--output outputs/metadata_validation.json

inspect-sample:
	$(PYTHON) scripts/inspect_sample.py \
		--metadata data/metadata.csv \
		--data-root . \
		--index 0

visualize-sample:
	$(PYTHON) scripts/visualize_sample.py \
		--metadata data/metadata.csv \
		--data-root . \
		--index 0 \
		--output-dir outputs/sample_visualization_small \
		--max-time-steps 400 \
		--max-receivers 200

sample-path:
	@echo "$(SAMPLE)"

train-cnn:
	$(PYTHON) scripts/train.py --config $(CNN_CONFIG)

train-transformer:
	$(PYTHON) scripts/train.py --config $(TRANSFORMER_CONFIG)

pretrain-transformer:
	$(PYTHON) scripts/pretrain_transformer.py --config $(PRETRAIN_CONFIG)

train-transformer-finetuned:
	$(PYTHON) scripts/train.py --config $(TRANSFORMER_FINETUNE_CONFIG)

train-multitask:
	$(PYTHON) scripts/train_multitask.py --config $(MULTITASK_CONFIG)

evaluate-cnn:
	$(PYTHON) scripts/evaluate_checkpoint.py \
		--checkpoint $(CNN_CKPT) \
		--metadata data/metadata.csv \
		--split val \
		--batch-size 16 \
		--device $(DEVICE) \
		--output outputs/cnn_baseline/evaluation_val.json

evaluate-transformer:
	$(PYTHON) scripts/evaluate_checkpoint.py \
		--checkpoint $(TRANSFORMER_CKPT) \
		--metadata data/metadata.csv \
		--split val \
		--batch-size 4 \
		--device $(DEVICE) \
		--output outputs/trace_transformer/evaluation_val.json

evaluate-transformer-finetuned:
	$(PYTHON) scripts/evaluate_checkpoint.py \
		--checkpoint $(TRANSFORMER_FINETUNED_CKPT) \
		--metadata data/metadata.csv \
		--split val \
		--batch-size 2 \
		--device $(DEVICE) \
		--output outputs/trace_transformer_finetuned/evaluation_val.json

evaluate-multitask:
	$(PYTHON) scripts/evaluate_multitask_checkpoint.py \
		--checkpoint $(MULTITASK_CKPT) \
		--metadata data/metadata.csv \
		--split val \
		--batch-size 8 \
		--device $(DEVICE) \
		--output outputs/cnn_multitask_50ep/evaluation_val.json

compare:
	@reports=(); \
	for path in \
		outputs/cnn_baseline/evaluation_val.json \
		outputs/trace_transformer/evaluation_val.json \
		outputs/trace_transformer_finetuned/evaluation_val.json \
		outputs/cnn_multitask_50ep/evaluation_val.json; do \
		if [ -f "$$path" ]; then reports+=("$$path"); fi; \
	done; \
	if [ "$${#reports[@]}" -eq 0 ]; then \
		echo "No evaluation reports found."; \
		exit 1; \
	fi; \
	$(PYTHON) scripts/compare_evaluations.py \
		--reports "$${reports[@]}" \
		--output-csv outputs/model_comparison.csv \
		--output-md outputs/model_comparison.md

predict:
	$(PYTHON) scripts/predict.py \
		--checkpoint $(CNN_CKPT) \
		--input "$(SAMPLE)" \
		--device $(DEVICE) \
		--output outputs/sample_prediction.json

predict-multitask:
	$(PYTHON) scripts/predict_multitask.py \
		--checkpoint $(MULTITASK_CKPT) \
		--input "$(SAMPLE)" \
		--device $(DEVICE) \
		--output outputs/cnn_multitask_50ep/sample_multitask_prediction.json

api:
	SEISMONN_CHECKPOINT=$(CNN_CKPT) \
	SEISMONN_DEVICE=$(DEVICE) \
	SEISMONN_PREDICTOR_TYPE=auto \
	$(UV) run uvicorn seismonn.api.main:app --host $(HOST) --port $(PORT)

api-multitask:
	SEISMONN_CHECKPOINT=$(MULTITASK_CKPT) \
	SEISMONN_DEVICE=$(DEVICE) \
	SEISMONN_PREDICTOR_TYPE=auto \
	$(UV) run uvicorn seismonn.api.main:app --host $(HOST) --port $(PORT)

docker-build:
	docker build -t seismonn-api .

docker-run:
	docker run --rm \
		-p 8000:8000 \
		-e SEISMONN_CHECKPOINT=/app/checkpoints/best.pt \
		-e SEISMONN_DEVICE=cpu \
		-e SEISMONN_PREDICTOR_TYPE=auto \
		-v "$$(pwd)/$(CNN_CKPT):/app/checkpoints/best.pt:ro" \
		seismonn-api

docker-run-multitask:
	docker run --rm \
		-p 8000:8000 \
		-e SEISMONN_CHECKPOINT=/app/checkpoints/best.pt \
		-e SEISMONN_DEVICE=cpu \
		-e SEISMONN_PREDICTOR_TYPE=auto \
		-v "$$(pwd)/$(MULTITASK_CKPT):/app/checkpoints/best.pt:ro" \
		seismonn-api

mlflow-ui:
	$(UV) run mlflow ui --backend-store-uri mlruns --host 127.0.0.1 --port 5000

benchmark-cnn:
	$(PYTHON) scripts/benchmark_inference.py \
		--checkpoint $(CNN_CKPT) \
		--input "$(SAMPLE)" \
		--device $(DEVICE) \
		--warmup-runs 3 \
		--timed-runs 20 \
		--output outputs/inference_benchmark.json

results:
	$(PYTHON) scripts/generate_results_report.py --output RESULTS.md

export-torchscript-cnn:
	$(PYTHON) scripts/export_torchscript.py \
		--checkpoint $(CNN_CKPT) \
		--output outputs/cnn_baseline/model_torchscript.pt \
		--device cpu

export-torchscript-multitask:
	$(PYTHON) scripts/export_torchscript.py \
		--checkpoint $(MULTITASK_CKPT) \
		--output outputs/cnn_multitask_50ep/model_torchscript.pt \
		--device cpu

export-multitask-predictions:
	$(PYTHON) scripts/export_multitask_predictions.py \
		--checkpoint $(MULTITASK_CKPT) \
		--metadata data/metadata.csv \
		--split val \
		--batch-size 8 \
		--device $(DEVICE) \
		--output-csv outputs/cnn_multitask_50ep/predictions_val.csv \
		--summary-output outputs/cnn_multitask_50ep/predictions_summary_val.json \
		--plots-dir outputs/cnn_multitask_50ep/parity_plots

clean-cache:
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
