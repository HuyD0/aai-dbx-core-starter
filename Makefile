SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

PYTHON ?= .venv/bin/python
UV ?= uv
UV_VERSION ?= 0.8.23
TERRAFORM ?= terraform
DATABRICKS ?= databricks
TARGET ?= dev
EXAMPLE ?=

.PHONY: help check-uv install lint format format-check test build check verify \
	sync-templates check-templates terraform-format terraform-format-check \
	terraform-init terraform-validate bundle-validate validate-templates doctor \
	doctor-cloud quickstart examples-install examples-connect examples-list example

help: ## Show the available targets.
	@awk 'BEGIN {FS = ":.*## "; printf "Usage: make <target> [TARGET=dev]\n\nTargets:\n"} /^[a-zA-Z0-9_-]+:.*## / {printf "  %-24s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

check-uv: ## Check that the pinned environment manager is available.
	@command -v "$(UV)" >/dev/null 2>&1 || { \
		echo "uv is required. Install uv $(UV_VERSION), then rerun this target."; \
		echo "For example: pipx install 'uv==$(UV_VERSION)'"; \
		exit 2; \
	}

install: check-uv ## Create/sync the locked SDK development environment with uv.
	$(UV) sync --extra dev --locked

examples-install: check-uv ## Install the locked Databricks and GenAI example dependencies.
	$(UV) sync --extra dev --extra databricks --extra genai --locked
	@$(PYTHON) -c 'import sys; import databricks.sdk; import mlflow; print(f"Example dependencies ready in {sys.executable} (MLflow {mlflow.__version__})")'

quickstart: install ## Prove a fresh clone works without credentials.
	$(PYTHON) scripts/examples.py quickstart

examples-connect: examples-install ## Prepare and check the keyless cloud example setup.
	$(PYTHON) scripts/examples.py connect

examples-list: install ## List learning examples and their execution mode.
	$(PYTHON) scripts/examples.py list

example: examples-install ## Preflight and run one example: make example EXAMPLE=first_trace
	@test -n "$(EXAMPLE)" || { \
		echo "EXAMPLE is required. Run 'make examples-list' to see valid names."; \
		exit 2; \
	}
	$(PYTHON) scripts/examples.py run "$(EXAMPLE)"

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
