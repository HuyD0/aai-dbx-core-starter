SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

UV ?= uv
PYTHON := .venv/bin/python
KERNEL_NAME := aai-local-classification
MLFLOW_DIR := $(CURDIR)/.aai/mlflow
MLFLOW_DB := $(MLFLOW_DIR)/mlflow.db
MLFLOW_URI := sqlite:///$(MLFLOW_DB)

.PHONY: help install prepare pipeline check test format format-check notebook-kernel \
	notebook notebook-check mlflow-ui

help: ## Show the local classification commands.
	@awk 'BEGIN {FS = ":.*## "; printf "Usage: make <target>\n\nTargets:\n"} /^[a-zA-Z0-9_-]+:.*## / {printf "  %-18s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Create the exact locked local environment.
	$(UV) sync --all-extras --locked

prepare: ## Generate and validate deterministic local data.
	$(UV) run --frozen --no-sync aai-classification prepare

pipeline: ## Run baseline, candidate selection, frozen test gate, and registration.
	$(UV) run --frozen --no-sync aai-classification run

check: format-check test notebook-check ## Run the complete local quality gate.

test: ## Run fast unit and workflow tests.
	$(UV) run --frozen --no-sync pytest -q

format: ## Apply project formatting.
	$(UV) run --frozen --no-sync ruff check --fix .
	$(UV) run --frozen --no-sync black .

format-check: ## Check Python and notebook formatting.
	$(UV) run --frozen --no-sync ruff check .
	$(UV) run --frozen --no-sync black --check .

notebook-kernel: ## Register this project's exact Python environment as a kernel.
	@test -x "$(PYTHON)" || { echo "Missing environment. Run 'make install'."; exit 2; }
	$(PYTHON) -m ipykernel install --prefix "$(CURDIR)/.venv" \
		--name "$(KERNEL_NAME)" --display-name "AAI Local Classification"

notebook: notebook-kernel ## Open the course at notebook 00.
	PATH="$(CURDIR)/.venv/bin:$$PATH" \
		JUPYTER_PATH="$(CURDIR)/.venv/share/jupyter" \
		$(PYTHON) -m jupyterlab --no-browser \
		--ServerApp.root_dir="$(CURDIR)" \
		--LabApp.default_url="/lab/tree/notebooks/00_start_here.ipynb" \
		--LabApp.extension_manager=readonly

notebook-check: notebook-kernel ## Execute every lesson from a clean temporary state.
	$(PYTHON) scripts/check_notebooks.py

mlflow-ui: ## Open the local experiment and model registry UI on port 5000.
	@mkdir -p "$(MLFLOW_DIR)/artifacts"
	$(PYTHON) -m mlflow ui --host 127.0.0.1 --port 5000 \
		--backend-store-uri "$(MLFLOW_URI)" \
		--default-artifact-root "$(MLFLOW_DIR)/artifacts"
