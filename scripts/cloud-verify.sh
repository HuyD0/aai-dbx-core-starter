#!/usr/bin/env bash
set -euo pipefail

readonly PYTHON_VERSION="3.12"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

if [[ "${AAI_CLOUD_ENV:-}" == "codex" ]]; then
  # Expected identifiers come from platform-identifiers.json — the single
  # source of truth a clone edits (see docs/enterprise-clone-runbook.md).
  identifier() {
    python3 -c "import json; print(json.load(open('platform-identifiers.json'))['$1'])"
  }
  required_values=(
    "AZURE_TENANT_ID=$(identifier azure_tenant_id)"
    "AZURE_SUBSCRIPTION_ID=$(identifier azure_subscription_id)"
    "DATABRICKS_HOST=$(identifier databricks_host)"
  )
  for required in "${required_values[@]}"; do
    name="${required%%=*}"
    expected="${required#*=}"
    if [[ "${!name:-}" != "${expected}" ]]; then
      echo "${name} is missing or does not match the documented identifier." >&2
      exit 1
    fi
  done

  if [[ -z "${AZURE_CLIENT_ID:-}" ]]; then
    echo "AZURE_CLIENT_ID must contain the dedicated non-secret client ID." >&2
    exit 1
  fi

  forbidden_credentials=(
    ARM_ACCESS_KEY
    ARM_CLIENT_SECRET
    AZURE_CLIENT_SECRET
    DATABRICKS_CLIENT_SECRET
    DATABRICKS_TOKEN
  )
  for name in "${forbidden_credentials[@]}"; do
    if [[ -n "${!name:-}" ]]; then
      echo "${name} must not be present in the Codex agent environment." >&2
      exit 1
    fi
  done
fi

uv lock --check
uv sync \
  --python "${PYTHON_VERSION}" \
  --extra dev \
  --extra all \
  --locked \
  --offline
# Generated scaffold and stamped identifier drift. CI runs this script and not
# `make check`, so without this line nothing in CI ever verified that
# databricks.yml and the template schema defaults still match
# platform-identifiers.json.
uv run --python "${PYTHON_VERSION}" python scripts/sync_template_shared.py --check
uv run --python "${PYTHON_VERSION}" ruff check .
uv run --python "${PYTHON_VERSION}" black --check .
uv run --python "${PYTHON_VERSION}" mypy --config-file pyproject.toml src/aai_core
uv run --python "${PYTHON_VERSION}" pytest -q \
  --cov=aai_core --cov-branch --cov-report=term-missing \
  --cov-report=xml
uv run --python "${PYTHON_VERSION}" python -m build --wheel --no-isolation
# Workflow security lint for this repo AND the workflows every template
# generates into team projects. --offline: no external audit calls.
# shellcheck disable=SC2086
uv run --python "${PYTHON_VERSION}" zizmor --offline .github/workflows \
  templates/*/template/.github/workflows

databricks bundle schema >/dev/null
uv run --python "${PYTHON_VERSION}" python - <<'PY'
from pathlib import Path

import yaml

for path in sorted(Path(".").glob("**/*.y*ml")):
    if any(part.startswith(".") for part in path.parts):
        continue
    with path.open(encoding="utf-8") as stream:
        yaml.safe_load(stream)
    print(f"ok: {path}")
PY

echo "cloud verification passed"
