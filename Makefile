SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

PYTHON ?= python
TERRAFORM ?= terraform
DATABRICKS ?= databricks
TARGET ?= dev

.PHONY: help install lint format format-check test build check verify \
	sync-templates check-templates terraform-format terraform-format-check \
	terraform-init terraform-validate bundle-validate validate-templates doctor \
	doctor-cloud

help: ## Show the available targets.
	@awk 'BEGIN {FS = ":.*## "; printf "Usage: make <target> [TARGET=dev]\n\nTargets:\n"} /^[a-zA-Z0-9_-]+:.*## / {printf "  %-24s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install the SDK and development dependencies.
	$(PYTHON) -m pip install -e '.[dev]'

lint: ## Run Ruff lint checks.
	$(PYTHON) -m ruff check .

format: ## Apply Ruff auto-fixes and Black formatting.
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m black .

format-check: ## Check Ruff and Black formatting without changing files.
	$(PYTHON) -m ruff check .
	$(PYTHON) -m black --check .

test: ## Run the credential-free test suite.
	$(PYTHON) -m pytest -q

build: ## Build the SDK source distribution and wheel.
	$(PYTHON) -m build

sync-templates: ## Copy the canonical shared scaffold into every template.
	$(PYTHON) scripts/sync_template_shared.py

check-templates: ## Check that generated template scaffold files are in sync.
	$(PYTHON) scripts/sync_template_shared.py --check

terraform-format: ## Format Terraform configuration.
	$(TERRAFORM) fmt -recursive infra

terraform-format-check: ## Check Terraform formatting.
	$(TERRAFORM) fmt -check -recursive infra

terraform-init: ## Initialize Terraform without the remote backend.
	$(TERRAFORM) -chdir=infra init -backend=false -input=false

terraform-validate: terraform-init ## Validate Terraform configuration.
	$(TERRAFORM) -chdir=infra validate

check: check-templates format-check test build terraform-format-check terraform-validate ## Run the standard pre-commit checks.

verify: ## Run the complete credential-free verification used by CI.
	./scripts/cloud-verify.sh

bundle-validate: ## Validate this repository's Databricks bundle (requires auth).
	$(DATABRICKS) bundle validate -t "$(TARGET)"

validate-templates: ## Render and validate all bundles against Databricks (requires auth).
	$(PYTHON) scripts/validate_templates.py

doctor: ## Run safe local SDK diagnostics.
	$(PYTHON) -m aai_core.diagnostics doctor

doctor-cloud: ## Run SDK diagnostics with cloud connectivity checks.
	$(PYTHON) -m aai_core.diagnostics doctor --cloud
