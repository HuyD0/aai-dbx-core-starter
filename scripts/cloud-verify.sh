#!/usr/bin/env bash
set -euo pipefail

readonly PYTHON_VERSION="3.12"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

if [[ "${AAI_CLOUD_ENV:-}" == "codex" ]]; then
  required_values=(
    "AZURE_TENANT_ID=7f6a2cf9-5e4e-46ae-95d4-74016c1df1a6"
    "AZURE_SUBSCRIPTION_ID=ea936670-dda1-4884-8467-49c225bf3e83"
    "DATABRICKS_HOST=https://adb-7405609799238491.11.azuredatabricks.net"
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
uv sync --python "${PYTHON_VERSION}" --extra dev --locked --offline
uv run --python "${PYTHON_VERSION}" ruff check .
uv run --python "${PYTHON_VERSION}" black --check .
uv run --python "${PYTHON_VERSION}" pytest -q
uv run --python "${PYTHON_VERSION}" python -m build --no-isolation

terraform fmt -check -recursive infra
if [[ ! -d infra/.terraform/providers ]]; then
  terraform -chdir=infra init -backend=false -input=false
fi
terraform -chdir=infra validate

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
