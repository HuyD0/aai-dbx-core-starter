SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

UV ?= uv
PYTHON := .venv/bin/python
KERNEL_NAME := aai-local-finetuning

.PHONY: help install prepare-flight flight-check study-smoke prepare-data \
	baselines train-smoke train evaluate capstone capstone-train-smoke \
	capstone-train capstone-evaluate notebook notebook-check notebook-kernel test \
	format format-check clean-study

help: ## Show the local fine-tuning commands.
	@awk 'BEGIN {FS = ":.*## "; printf "Usage: make <target>\n\nTargets:\n"} /^[a-zA-Z0-9_-]+:.*## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install the exact plane-ready environment while online.
	$(UV) sync --all-extras --locked

prepare-flight: install ## Download and verify data/model assets before travel.
	$(UV) run --frozen --no-sync aai-finetune prepare-flight

flight-check: ## Prove the prepared environment and assets work without a network.
	$(UV) run --offline --frozen --no-sync aai-finetune --offline flight-check

study-smoke: ## Run the deterministic study path with network access blocked.
	$(UV) run --offline --frozen --no-sync aai-finetune --offline smoke

prepare-data: ## Rebuild portable Bitext splits from the immutable local download.
	$(PYTHON) -m aai_local_finetuning.cli --offline prepare-data

baselines: ## Score the majority and deterministic baselines locally.
	$(PYTHON) -m aai_local_finetuning.cli --offline baselines --track

train-smoke: ## Run a short MLX-LM LoRA training check.
	$(PYTHON) -m aai_local_finetuning.cli --offline train --iterations 10

train: ## Run the configured MLX-LM LoRA change.
	$(PYTHON) -m aai_local_finetuning.cli --offline train

evaluate: ## Evaluate local baseline and LoRA predictions on the frozen test set.
	$(PYTHON) -m aai_local_finetuning.cli --offline evaluate --track

capstone: ## Generate and verify the deterministic readiness capstone dataset.
	$(PYTHON) -m aai_local_finetuning.cli --offline capstone

capstone-train-smoke: ## Run one compact capstone LoRA iteration.
	$(PYTHON) -m aai_local_finetuning.cli --offline capstone-train --iterations 1

capstone-train: ## Train the configured compact capstone LoRA change.
	$(PYTHON) -m aai_local_finetuning.cli --offline capstone-train

capstone-evaluate: ## Compare all capstone methods on the frozen test set.
	$(PYTHON) -m aai_local_finetuning.cli --offline capstone-evaluate

notebook-kernel: ## Register the exact nested Python as the course kernel.
	@test -x "$(PYTHON)" || { \
		echo "Missing locked environment. Run 'make prepare-flight' while online."; \
		exit 2; \
	}
	$(PYTHON) -m ipykernel install \
		--prefix "$(CURDIR)/.venv" \
		--name "$(KERNEL_NAME)" \
		--display-name "AAI Local Fine-Tuning (offline)"

notebook: notebook-kernel ## Open the prepared numbered notebook course without downloads.
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DATASETS_OFFLINE=1 \
		PATH="$(CURDIR)/.venv/bin:$$PATH" \
		JUPYTER_PATH="$(CURDIR)/.venv/share/jupyter" \
		$(PYTHON) -m jupyterlab --no-browser \
		--ServerApp.root_dir="$(CURDIR)" \
		--LabApp.default_url="/lab/tree/notebooks/00_start_here.ipynb" \
		--LabApp.extension_manager=readonly

notebook-check: notebook-kernel ## Execute notebook 00 through the registered kernel.
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DATASETS_OFFLINE=1 \
		PATH="$(CURDIR)/.venv/bin:$$PATH" \
		JUPYTER_PATH="$(CURDIR)/.venv/share/jupyter" \
		$(PYTHON) -m jupyter nbconvert --execute --to notebook --stdout \
		notebooks/00_start_here.ipynb >/dev/null

test: ## Run the fast cross-platform curriculum tests.
	$(UV) run --offline --frozen --no-sync pytest -q

format: ## Format the nested project.
	$(UV) run --offline --frozen --no-sync ruff check --fix .
	$(UV) run --offline --frozen --no-sync black .

format-check: ## Check formatting without changing files.
	$(UV) run --offline --frozen --no-sync ruff check .
	$(UV) run --offline --frozen --no-sync black --check .

clean-study: ## Remove generated study outputs but retain downloads and the model.
	$(PYTHON) -m aai_local_finetuning.cli clean-study
