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
OPS_RAG_DIR := $(CURDIR)/examples/agentic-ops-rag
OPS_RAG_KERNEL := aai-agentic-ops-rag
OPS_RAG_MLFLOW_DIR := $(OPS_RAG_DIR)/.aai
OPS_RAG_MLFLOW_URI := sqlite:///$(OPS_RAG_MLFLOW_DIR)/mlflow.db

.PHONY: help check-uv install lint format format-check typecheck test coverage audit \
	build check verify \
	sync-templates check-templates lock-templates check-template-locks \
	sync-upstream resolve-upstream bundle-validate validate-templates doctor \
	doctor-cloud quickstart examples-install examples-list local-start local-example \
	local-lifecycle local-ui workspace-connect workspace-example examples-connect example \
	pre-commit pre-push hooks-install hooks-run app-run app-start app-stop app-restart \
	study-prepare-flight study-offline-check study-lab notebook \
	classification-install classification-prepare classification-train \
	classification-doctor classification-reset classification-check \
	classification-notebook classification-ui \
	ops-rag-install ops-rag-doctor ops-rag-render ops-rag-check \
	ops-rag-notebook ops-rag-ui

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
	$(UV) sync --extra dev --extra azure-apim --extra azure-search --extra databricks --extra genai --extra examples --locked
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

notebook: ## Open the offline fine-tuning notebook course.
	$(MAKE) -C examples/local-finetuning notebook

classification-install: check-uv ## Install the locked local classification course.
	$(MAKE) -C examples/local-classification install

classification-doctor: check-uv ## Verify the local classification course setup.
	$(MAKE) -C examples/local-classification doctor

classification-reset: ## Recoverably archive local classification course-v2 state.
	$(MAKE) -C examples/local-classification course-reset

classification-prepare: check-uv ## Generate and validate the synthetic classification data.
	$(MAKE) -C examples/local-classification prepare

classification-train: check-uv ## Run the complete local classification lifecycle.
	$(MAKE) -C examples/local-classification pipeline

classification-check: check-uv ## Test code and execute every classification notebook.
	$(MAKE) -C examples/local-classification check

classification-notebook: check-uv ## Open the local classification notebook course.
	$(MAKE) -C examples/local-classification notebook

classification-ui: check-uv ## Serve the classification course's local MLflow UI.
	$(MAKE) -C examples/local-classification mlflow-ui

ops-rag-install: examples-install ## Install the locked agentic operations RAG workshop environment.
	@$(PYTHON) -c 'import azure.search.documents; import openai; print("Agentic operations RAG provider extras ready")'

ops-rag-doctor: ops-rag-install ## Check the workshop without cloud calls; add CONNECTED=1 to require filled config.
	$(PYTHON) $(OPS_RAG_DIR)/scripts/doctor.py $(if $(CONNECTED),--connected,)

ops-rag-render: ## Regenerate the workshop notebooks from reviewable Python sources.
	$(PYTHON) $(OPS_RAG_DIR)/scripts/render_notebooks.py

ops-rag-check: ops-rag-install ## Test and execute all credential-free workshop lessons.
	$(PYTHON) -m pytest -q tests/test_agentic_ops_rag.py
	$(PYTHON) $(OPS_RAG_DIR)/scripts/check_notebooks.py --execute

ops-rag-notebook: ops-rag-install ## Open the workshop at lesson 00 in JupyterLab.
	$(PYTHON) -m ipykernel install --prefix "$(CURDIR)/.venv" \
		--name "$(OPS_RAG_KERNEL)" --display-name "AAI Agentic Operations RAG"
	PATH="$(CURDIR)/.venv/bin:$$PATH" \
		JUPYTER_PATH="$(CURDIR)/.venv/share/jupyter" \
		$(PYTHON) -m jupyterlab \
		--ServerApp.root_dir="$(OPS_RAG_DIR)" \
		--LabApp.default_url="/lab/tree/notebooks/00_environment_and_stack_map.ipynb" \
		--LabApp.extension_manager=readonly

ops-rag-ui: ops-rag-install ## Serve the workshop's isolated MLflow UI at http://127.0.0.1:5001.
	@mkdir -p "$(OPS_RAG_MLFLOW_DIR)/mlruns"
	$(PYTHON) -m mlflow ui --host 127.0.0.1 --port 5001 \
		--backend-store-uri "$(OPS_RAG_MLFLOW_URI)" \
		--default-artifact-root "$(OPS_RAG_MLFLOW_DIR)/mlruns"

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

typecheck: ## Type-check the public SDK and provider boundaries.
	$(PYTHON) -m mypy --config-file pyproject.toml src/aai_core

test: ## Run the credential-free test suite.
	$(PYTHON) -m pytest -q

coverage: check-uv ## Run SDK tests with branch coverage and the release quality floor.
	$(UV) sync --extra dev --extra all --locked
	$(PYTHON) -m pytest -q \
		--cov=aai_core --cov-branch --cov-report=term-missing \
		--cov-report=xml

audit: ## Audit the locked environment for published dependency advisories.
	$(PYTHON) templates/_shared/files/scripts/audit_dependencies.py \
		--policy templates/_shared/files/security-audit.toml --uv-project .
	$(PYTHON) templates/_shared/files/scripts/audit_dependencies.py \
		--policy templates/_shared/files/security-audit.toml \
		--uv-project examples/local-classification
	$(PYTHON) templates/_shared/files/scripts/audit_dependencies.py \
		--policy templates/_shared/files/security-audit.toml \
		--uv-project examples/local-finetuning
	$(PYTHON) templates/_shared/files/scripts/audit_dependencies.py \
		--policy templates/_shared/files/security-audit.toml \
		--requirement examples/local-classification/src/aai_local_classification/model-requirements.lock

build: ## Build the SDK wheel used by downstream templates.
	$(PYTHON) -m build --wheel

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

sync-upstream: ## Merge a reviewed upstream release into a clone: make sync-upstream TAG=vX.Y.Z
	@# One-way upstream -> clone sync on a reviewed, annotated release tag. Keep
	@# the merge staged for review, mechanically resolve only generated stamped
	@# files, re-stamp the clone's identifiers, and run the credential-free gate.
	@test -n "$(TAG)" || { \
		echo "TAG is required, e.g. 'make sync-upstream TAG=v0.4.0'." >&2; \
		exit 2; \
	}
	@test -z "$$(git status --porcelain --untracked-files=normal)" || { \
		echo "sync-upstream requires a clean worktree; commit or stash local changes first." >&2; \
		exit 2; \
	}
	@test "$$(git config --get merge.keepours.driver || true)" = "true" || { \
		echo "configure the clone's keepours merge driver before syncing; see docs/enterprise-clone-runbook.md section 3a." >&2; \
		exit 2; \
	}
	git fetch upstream --tags
	@tag_ref="refs/tags/$(TAG)"; \
	if ! git rev-parse --verify "$${tag_ref}^{commit}" >/dev/null 2>&1; then \
		echo "$(TAG) is not a fetched release tag." >&2; \
		exit 2; \
	fi; \
	if [ "$$(git cat-file -t "$$tag_ref")" != "tag" ]; then \
		echo "$(TAG) must be an annotated release tag." >&2; \
		exit 2; \
	fi; \
	tag_commit=$$(git rev-parse "$${tag_ref}^{commit}"); \
	if git merge-base --is-ancestor "$$tag_commit" HEAD; then \
		echo "$(TAG) is already contained in HEAD; nothing to sync." >&2; \
		exit 2; \
	fi
	@if git merge --no-commit --no-ff "$(TAG)"; then \
		echo "merged $(TAG) cleanly (staged, not committed)"; \
	else \
		merge_status=$$?; \
		conflicted=$$(git diff --name-only --diff-filter=U); \
		if [ -z "$$conflicted" ]; then \
			echo "merge failed without file conflicts; refusing to continue." >&2; \
			exit "$$merge_status"; \
		fi; \
		if printf '%s\n' "$$conflicted" | grep -qx 'Makefile'; then \
			echo "Makefile is conflicted; resolve it by hand, then run 'make resolve-upstream'." >&2; \
			exit "$$merge_status"; \
		fi; \
		echo "merge conflicts present; resolving generated stamped files..."; \
		$(MAKE) resolve-upstream; \
	fi
	$(MAKE) sync-templates
	@# A clean preflight makes these the only possible restamp outputs. Stage the
	@# paths owned by sync-templates, then fail if anything unexpected is unstaged.
	@git add -- databricks.yml templates
	@unstaged=$$(git diff --name-only); \
	untracked=$$(git ls-files --others --exclude-standard); \
	if [ -n "$$unstaged$$untracked" ]; then \
		echo "sync left unexpected unstaged files; inspect them before continuing:" >&2; \
		[ -z "$$unstaged" ] || echo "$$unstaged" >&2; \
		[ -z "$$untracked" ] || echo "$$untracked" >&2; \
		exit 1; \
	fi
	$(MAKE) verify
	@echo "sync of $(TAG) is staged. Review 'git diff --cached' and 'git log',"
	@echo "then 'git commit' and open a PR into main."

check: check-templates format-check typecheck test build ## Run standard local checks.

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
