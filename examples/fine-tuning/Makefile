SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

UV ?= uv
UV_VERSION ?= 0.8.23
PYTHON := .venv/bin/python
KERNEL_NAME := aai-fine-tuning
# Bump this version when persisted course evidence becomes incompatible.
COURSE_ROOT := $(CURDIR)/.aai/course-v1
# HF_HUB_OFFLINE turns any accidental Hugging Face download into a loud error:
# every lesson builds models from configuration, never from pretrained weights.
COURSE_ENV := AAI_FINETUNE_PROJECT_ROOT="$(COURSE_ROOT)" \
	HF_HUB_OFFLINE=1 \
	HF_HOME="$(COURSE_ROOT)/hf"
MLFLOW_DIR := $(COURSE_ROOT)/mlflow
MLFLOW_DB := $(MLFLOW_DIR)/mlflow.db
MLFLOW_URI := sqlite:///$(MLFLOW_DB)

.PHONY: help check-uv install doctor course-reset check test format \
	format-check notebook-kernel notebook notebook-check mlflow-ui

help: ## Show the fine-tuning course commands.
	@awk 'BEGIN {FS = ":.*## "; printf "Usage: make <target>\n\nTargets:\n"} /^[a-zA-Z0-9_-]+:.*## / {printf "  %-18s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

check-uv: ## Explain how to install uv when it is unavailable.
	@command -v "$(UV)" >/dev/null 2>&1 || { \
		echo "uv is required before this course can install Python packages."; \
		echo "Follow README.md > One-time setup, then rerun make install."; \
		echo "Expected course tool version: uv $(UV_VERSION)."; \
		exit 2; \
	}

install: check-uv ## Create the exact locked local environment.
	$(UV) sync --all-extras --locked

doctor: notebook-kernel ## Verify the course environment and explain the correct kernel.
	$(COURSE_ENV) $(PYTHON) scripts/doctor.py

course-reset: ## Archive course-v1 state so the course can start fresh.
	@course_root="$(COURSE_ROOT)"; \
	expected_root="$(CURDIR)/.aai/course-v1"; \
	if [ "$$course_root" != "$$expected_root" ]; then \
		echo "Refusing reset: expected $$expected_root, got $$course_root"; \
		exit 2; \
	fi; \
	if [ ! -e "$$course_root" ]; then \
		echo "Nothing to reset: course-v1 has no saved state."; \
		exit 0; \
	fi; \
	archive_root="$(CURDIR)/.aai/archive"; \
	mkdir -p "$$archive_root"; \
	stamp=$$(date -u +%Y%m%dT%H%M%SZ); \
	destination="$$archive_root/course-v1-$$stamp"; \
	counter=0; \
	while [ -e "$$destination" ]; do \
		counter=$$((counter + 1)); \
		destination="$$archive_root/course-v1-$$stamp-$$counter"; \
	done; \
	mv "$$course_root" "$$destination"; \
	echo "Archived course-v1 state to $$destination"; \
	echo "Nothing was deleted. To restore it, move that directory back to $$course_root."

check: format-check test notebook-check ## Run the complete local quality gate.

test: check-uv ## Run fast unit and contract tests.
	$(UV) run --frozen --no-sync pytest -q

format: check-uv ## Apply project formatting.
	$(UV) run --frozen --no-sync ruff check --fix .
	$(UV) run --frozen --no-sync black .

format-check: check-uv ## Check Python and notebook formatting.
	$(UV) run --frozen --no-sync ruff check .
	$(UV) run --frozen --no-sync black --check .

notebook-kernel: ## Register this project's exact Python environment as a kernel.
	@test -x "$(PYTHON)" || { echo "Missing environment. Run 'make install'."; exit 2; }
	$(PYTHON) -m ipykernel install --prefix "$(CURDIR)/.venv" \
		--name "$(KERNEL_NAME)" --display-name "AAI Fine-Tuning"

notebook: notebook-kernel ## Open the course at notebook 00.
	PATH="$(CURDIR)/.venv/bin:$$PATH" \
		JUPYTER_PATH="$(CURDIR)/.venv/share/jupyter" \
		$(COURSE_ENV) \
		$(PYTHON) -m jupyterlab \
		--ServerApp.root_dir="$(CURDIR)" \
		--LabApp.default_url="/lab/tree/notebooks/00_start_here.ipynb" \
		--LabApp.extension_manager=readonly

notebook-check: notebook-kernel ## Execute every lesson from a clean temporary state.
	$(PYTHON) scripts/check_notebooks.py

mlflow-ui: ## Open the local experiment tracking UI on port 5000.
	@mkdir -p "$(MLFLOW_DIR)/artifacts"
	$(PYTHON) -m mlflow ui --host 127.0.0.1 --port 5000 \
		--backend-store-uri "$(MLFLOW_URI)" \
		--default-artifact-root "$(MLFLOW_DIR)/artifacts"
