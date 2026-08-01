SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

PYTHON ?= .venv/bin/python
UV ?= uv
UV_VERSION ?= 0.8.23
DATABRICKS ?= databricks
TARGET ?= dev
EXAMPLE ?=
LOCAL_MLFLOW_DIR ?= $(CURDIR)/.aai/local
LOCAL_MLFLOW_DB ?= $(LOCAL_MLFLOW_DIR)/mlflow.db
LOCAL_MLFLOW_URI = sqlite:///$(LOCAL_MLFLOW_DB)
# Platform console (src/platform_app). APP_NAME must match the `name` of the app
# resource; it is stopped by default so a forgotten console cannot bill.
APP_PORT ?= 8000
APP_NAME ?= aai-platform-console-dev

.PHONY: help check-uv install lint format format-check test build check verify \
	sync-templates check-templates lock-templates check-template-locks \
	bundle-validate validate-templates doctor \
	doctor-cloud quickstart examples-install examples-list local-start local-example \
	local-lifecycle local-ui workspace-connect workspace-example examples-connect example \
	pre-commit pre-push hooks-install hooks-run app-run app-start app-stop app-restart \
	study-prepare-flight study-offline-check study-lab

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

pre-commit: ## Run the fast credential-free commit gate.
	./scripts/pre-commit.sh

pre-push: ## Run the complete credential-free verification gate.
	./scripts/pre-push.sh

hooks-install: install ## Install the repository's pre-commit and pre-push hooks.
	$(PYTHON) -m pre_commit install \
		--hook-type pre-commit \
		--hook-type pre-push

hooks-run: pre-commit pre-push ## Run both Git hook stages now.

examples-install: check-uv ## Install locked Databricks, GenAI, and interactive example dependencies.
	$(UV) sync --extra dev --extra databricks --extra genai --extra examples --locked
	@$(PYTHON) -c 'import sys; import databricks.sdk; import ipykernel; import jupyterlab; import mlflow; print(f"Example dependencies ready in {sys.executable} (MLflow {mlflow.__version__})")'

quickstart: install ## Prove a fresh clone works without credentials.
	$(PYTHON) scripts/examples.py quickstart

local-start: examples-install ## Write a trace to a clean, repository-local MLflow store.
	$(PYTHON) scripts/examples.py local first_trace

local-example: examples-install ## Run one local lifecycle example or lab.
	@test -n "$(EXAMPLE)" || { \
		echo "EXAMPLE is required. Run 'make examples-list' to see valid names."; \
		exit 2; \
	}
	$(PYTHON) scripts/examples.py local "$(EXAMPLE)"

local-lifecycle: examples-install ## Run the complete deterministic MLflow curriculum.
	$(PYTHON) scripts/examples.py local first_trace
	$(PYTHON) scripts/examples.py local first_experiment
	$(PYTHON) scripts/examples.py local first_prompt
	$(PYTHON) scripts/examples.py local first_evaluation

local-ui: examples-install ## Serve the isolated local MLflow store at http://127.0.0.1:5000.
	@mkdir -p "$(LOCAL_MLFLOW_DIR)/mlruns"
	$(PYTHON) -m mlflow ui \
		--backend-store-uri "$(LOCAL_MLFLOW_URI)" \
		--default-artifact-root "$(LOCAL_MLFLOW_DIR)/mlruns"

study-prepare-flight: ## Prepare the Apple-silicon fine-tuning project while online.
	$(MAKE) -C examples/local-finetuning prepare-flight

study-offline-check: ## Prove the prepared fine-tuning project is plane-ready.
	$(MAKE) -C examples/local-finetuning flight-check

study-lab: ## Run the fine-tuning project's deterministic offline study path.
	$(MAKE) -C examples/local-finetuning study-smoke

workspace-connect: examples-install ## Prepare and check keyless Databricks workspace access.
	$(PYTHON) scripts/examples.py connect

examples-connect: workspace-connect ## Backward-compatible alias for workspace-connect.

examples-list: install ## List learning examples and their execution mode.
	$(PYTHON) scripts/examples.py list

workspace-example: examples-install ## Send an example to Databricks: make workspace-example EXAMPLE=first_trace
	@test -n "$(EXAMPLE)" || { \
		echo "EXAMPLE is required. Run 'make examples-list' to see valid names."; \
		exit 2; \
	}
	$(PYTHON) scripts/examples.py workspace "$(EXAMPLE)"

example: workspace-example ## Backward-compatible alias for workspace-example.

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

lock-templates: check-uv ## Regenerate exact transitive template runtime locks.
	$(PYTHON) scripts/lock_template_dependencies.py

check-template-locks: check-uv ## Resolve and verify template runtime locks are current.
	$(PYTHON) scripts/lock_template_dependencies.py --check

resolve-upstream: ## Resolve a downstream clone's stamped-file conflicts after merging upstream.
	@# Only the files sync-templates generates are handled here: their conflicts
	@# are always "upstream changed its identifiers, we keep ours", which is
	@# mechanical. Take upstream's content so genuine template changes survive,
	@# then re-stamp this clone's identifiers over it.
	@#
	@# CAVEAT: this discards local edits to the *rest* of those files. If this
	@# clone customises a template schema beyond its identifier defaults, resolve
	@# that file by hand instead. Anything not generated is left conflicted for
	@# you to resolve deliberately.
	@conflicted=$$(git diff --name-only --diff-filter=U); \
	if [ -z "$$conflicted" ]; then echo "no conflicts to resolve"; exit 0; fi; \
	generated=$$(echo "$$conflicted" | grep -E '^(databricks\.yml|templates/[^/]+/databricks_template_schema\.json)$$' || true); \
	if [ -n "$$generated" ]; then \
		echo "$$generated" | xargs git checkout --theirs --; \
		echo "$$generated" | xargs git add --; \
		$(PYTHON) scripts/sync_template_shared.py; \
		echo "$$generated" | xargs git add --; \
		echo "re-stamped: $$generated"; \
	fi; \
	remaining=$$(git diff --name-only --diff-filter=U); \
	if [ -n "$$remaining" ]; then \
		echo "resolve these by hand, then 'git commit':"; echo "$$remaining"; exit 1; \
	fi; \
	echo "stamped-file conflicts resolved; review 'git diff --cached', then 'git commit'"

check: check-templates format-check test build ## Run the standard pre-commit checks.

verify: ## Run the complete credential-free verification used by CI.
	./scripts/cloud-verify.sh

bundle-validate: ## Validate this repository's Databricks bundle (requires auth).
	$(DATABRICKS) bundle validate -t "$(TARGET)"

validate-templates: ## Render and validate all bundles against Databricks (requires auth).
	$(PYTHON) scripts/validate_templates.py

app-run: install ## Serve the platform console locally at http://127.0.0.1:8000.
	cd src/platform_app && ../../$(PYTHON) -m uvicorn aai_console.server:app \
		--host 127.0.0.1 --port $(APP_PORT) --reload

app-start: ## Start the deployed console (it is stopped by default to avoid standing cost).
	$(DATABRICKS) apps start "$(APP_NAME)"

app-stop: ## Stop the deployed console. Stopped apps do not bill.
	$(DATABRICKS) apps stop "$(APP_NAME)"

app-restart: ## Restart the console so it picks up newly deployed code.
	$(DATABRICKS) apps stop "$(APP_NAME)"
	$(DATABRICKS) apps start "$(APP_NAME)"

doctor: ## Run safe local SDK diagnostics.
	$(PYTHON) -m aai_core.diagnostics doctor

doctor-cloud: ## Run SDK diagnostics with cloud connectivity checks.
	$(PYTHON) -m aai_core.diagnostics doctor --cloud
