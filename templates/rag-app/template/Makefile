SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

PYTHON ?= .venv/bin/python
DATABRICKS ?= databricks
TARGET ?= dev

.PHONY: help install install-ci lint format format-check test offline-checks \
	check build evaluate bundle-validate bundle-deploy doctor doctor-cloud

help: ## Show the available targets.
	@awk 'BEGIN {FS = ":.*## "; printf "Usage: make <target> [TARGET=dev]\n\nTargets:\n"} /^[a-zA-Z0-9_-]+:.*## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install the pinned private SDK and project dependencies (requires auth).
	$(PYTHON) scripts/install_core.py
	$(PYTHON) -m pip install -r requirements.lock
	$(PYTHON) -m pip install -e '.[dev]'

install-ci: ## Install from the credential-free CI package source.
	$(PYTHON) -m pip install -r requirements-ci.txt
	$(PYTHON) -m pip install -r requirements.lock
	$(PYTHON) -m pip install -e '.[dev]'

lint: ## Run Ruff lint checks.
	$(PYTHON) -m ruff check .

format: ## Apply Ruff auto-fixes and Black formatting.
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m black .

format-check: ## Check Ruff and Black formatting without changing files.
	$(PYTHON) -m ruff check .
	$(PYTHON) -m black --check .

test: ## Run unit tests.
	$(PYTHON) -m pytest -q

offline-checks: ## Run the deterministic release gate.
	$(PYTHON) evals/offline_checks.py

check: format-check test offline-checks ## Run all credential-free checks.

build: ## Build the application wheel.
	$(PYTHON) scripts/build_application.py

evaluate: ## Run the full evaluation gate (requires configured providers).
	$(PYTHON) evals/evaluate.py

bundle-validate: ## Validate the Databricks bundle (requires auth).
	$(DATABRICKS) bundle validate -t "$(TARGET)"

bundle-deploy: ## Deploy the Databricks bundle (requires auth).
	$(DATABRICKS) bundle deploy -t "$(TARGET)"

doctor: ## Run safe local SDK diagnostics.
	$(PYTHON) -m aai_core.diagnostics doctor

doctor-cloud: ## Run SDK diagnostics with cloud connectivity checks.
	$(PYTHON) -m aai_core.diagnostics doctor --cloud
