SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

PYTHON ?= .venv/bin/python
DATABRICKS ?= databricks
TARGET ?= dev
PROMPT_VERSION ?=
KNOWLEDGE_VERSION ?=
EVALUATE_ARGS ?=

.PHONY: help install install-ci lint format format-check typecheck test coverage audit \
	contracts offline-checks check build evaluate bundle-validate bundle-deploy \
	doctor doctor-cloud

help: ## Show the available targets.
	@awk 'BEGIN {FS = ":.*## "; printf "Usage: make <target> [TARGET=dev] [PROMPT_VERSION=<n>] [KNOWLEDGE_VERSION=<id>]\n\nTargets:\n"} /^[a-zA-Z0-9_-]+:.*## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install the pinned private SDK and project dependencies (requires auth).
	$(PYTHON) -m pip install -r requirements.lock
	$(PYTHON) -m pip install -r requirements-dev.lock
	$(PYTHON) scripts/install_core.py
	$(PYTHON) -m pip install --no-deps -e .

install-ci: ## Install from the credential-free CI package source.
	$(PYTHON) -m pip install -r requirements-dev.lock
	$(PYTHON) -m pip install -r requirements.lock
	$(PYTHON) -m pip install --no-deps -r requirements-ci.txt
	$(PYTHON) -m pip install --no-deps -e .

lint: ## Run Ruff lint checks.
	$(PYTHON) -m ruff check .

format: ## Apply Ruff auto-fixes and Black formatting.
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m black .

format-check: ## Check Ruff and Black formatting without changing files.
	$(PYTHON) -m ruff check .
	$(PYTHON) -m black --check .

typecheck: ## Type-check generated production code.
	$(PYTHON) -m mypy --ignore-missing-imports --check-untyped-defs \
		--disallow-incomplete-defs --disallow-untyped-defs src/app

test: ## Run unit tests.
	$(PYTHON) -m pytest -q

coverage: ## Run tests with branch coverage for generated application code.
	$(PYTHON) -m pytest -q --cov=app --cov-report=term-missing

audit: ## Audit the exact runtime dependency lock.
	@while IFS= read -r lock; do \
		echo "auditing $$lock"; \
		$(PYTHON) scripts/audit_dependencies.py --policy security-audit.toml \
			--requirement "$$lock"; \
	done < <({ find . -path ./.venv -prune -o -name requirements.lock -type f -print; \
		printf '%s\n' requirements-dev.lock; } | sort -u)

contracts: ## Validate manifest, runtime configuration, and bundle consistency.
	$(PYTHON) scripts/validate_project.py

offline-checks: ## Run the deterministic release gate.
	$(PYTHON) evals/offline_checks.py

check: format-check typecheck coverage audit contracts offline-checks ## Run checks.

build: ## Build the application wheel.
	$(PYTHON) scripts/build_application.py

evaluate: ## Run the full evaluation gate (requires configured providers).
	@test -n "$(strip $(PROMPT_VERSION))" || { echo "PROMPT_VERSION=<immutable-registry-version> is required" >&2; exit 2; }
	@test -n "$(strip $(KNOWLEDGE_VERSION))" || { echo "KNOWLEDGE_VERSION=<immutable-snapshot-id> is required" >&2; exit 2; }
	$(PYTHON) evals/evaluate.py --prompt-version "$(PROMPT_VERSION)" \
		--knowledge-version "$(KNOWLEDGE_VERSION)" $(EVALUATE_ARGS)

bundle-validate: ## Validate the Databricks bundle (requires auth).
	$(DATABRICKS) bundle validate -t "$(TARGET)" --strict

bundle-deploy: ## Deploy the Databricks bundle (requires auth).
	$(DATABRICKS) bundle deploy -t "$(TARGET)"

doctor: ## Run safe local SDK diagnostics.
	$(PYTHON) -m aai_core.diagnostics doctor

doctor-cloud: ## Run SDK diagnostics with cloud connectivity checks.
	$(PYTHON) -m aai_core.diagnostics doctor --cloud
