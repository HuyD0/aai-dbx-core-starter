#!/usr/bin/env bash
set -euo pipefail

# Idempotent setup for the OpenAI codex-universal Ubuntu 24.04 image.
# Network access is required only while this script runs.

readonly PYTHON_VERSION="3.12"
readonly UV_VERSION="0.8.23"
readonly TERRAFORM_VERSION="1.12.2"
# KEEP IN LOCKSTEP with the databricks/setup-cli SHA pinned in
# .github/workflows/*.yml (tests/test_smoke.py enforces the match): a version
# skew makes bundle behavior diverge between Codex/local and CI. Bumping this
# requires updating the per-arch SHA-256 checksums below in the same change:
#   gh release download "v<version>" -R databricks/cli -p "*SHA256SUMS*"
# (equivalently, Databricks' official Formula/databricks.rb in
# github.com/databricks/homebrew-tap records the same zip checksums).
readonly DATABRICKS_CLI_VERSION="1.9.0"
readonly AZURE_CLI_VERSION="2.88.0"
readonly AZURE_CLI_DEB_VERSION="${AZURE_CLI_VERSION}-1~noble"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
local_bin="${HOME}/.local/bin"
tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "codex-cloud-setup.sh supports the Linux Codex Cloud runtime only." >&2
  exit 1
fi

case "$(uname -m)" in
  x86_64)
    archive_arch="amd64"
    terraform_sha256="1eaed12ca41fcfe094da3d76a7e9aa0639ad3409c43be0103ee9f5a1ff4b7437"
    databricks_sha256="10938da31db7f89e6e90f6be41d340e3387231e5da5125cfc97ecb8cb66f6394"
    ;;
  aarch64 | arm64)
    archive_arch="arm64"
    terraform_sha256="f8a0347dc5e68e6d60a9fa2db361762e7943ed084a773f28a981d988ceb6fdc9"
    databricks_sha256="a4fcc3b70ed4b26f0e8064e39287d0692218cf3d2f549042d66d04c9ad2d692a"
    ;;
  *)
    echo "Unsupported Codex Cloud architecture: $(uname -m)" >&2
    exit 1
    ;;
esac

install -d "${local_bin}"
export PATH="${local_bin}:${PATH}"

if command -v pyenv >/dev/null 2>&1; then
  pyenv global "${PYTHON_VERSION}"
fi

python_minor="$(python --version | awk '{print $2}' | cut -d. -f1-2)"
if [[ "${python_minor}" != "${PYTHON_VERSION}" ]]; then
  echo "Expected Python ${PYTHON_VERSION}; found $(python --version)." >&2
  exit 1
fi

if ! command -v pipx >/dev/null 2>&1; then
  python -m pip install --user "pipx==1.7.1"
fi
if [[ "$(uv --version 2>/dev/null | awk '{print $2}' || true)" != "${UV_VERSION}" ]]; then
  pipx install --force "uv==${UV_VERSION}"
fi

if [[ "$(terraform version -json 2>/dev/null | jq -r '.terraform_version' || true)" != "${TERRAFORM_VERSION}" ]]; then
  terraform_archive="${tmp_dir}/terraform.zip"
  curl -fsSL \
    "https://releases.hashicorp.com/terraform/${TERRAFORM_VERSION}/terraform_${TERRAFORM_VERSION}_linux_${archive_arch}.zip" \
    -o "${terraform_archive}"
  echo "${terraform_sha256}  ${terraform_archive}" | sha256sum --check -
  unzip -qo "${terraform_archive}" -d "${local_bin}"
  chmod 0755 "${local_bin}/terraform"
fi

if [[ "$(databricks version 2>/dev/null | awk '{print $3}' || true)" != "v${DATABRICKS_CLI_VERSION}" ]]; then
  # CLI 1.x releases publish zip archives; the pinned checksums above are the
  # zip checksums Databricks records in its official Homebrew formula.
  databricks_archive="${tmp_dir}/databricks.zip"
  curl -fsSL \
    "https://github.com/databricks/cli/releases/download/v${DATABRICKS_CLI_VERSION}/databricks_cli_${DATABRICKS_CLI_VERSION}_linux_${archive_arch}.zip" \
    -o "${databricks_archive}"
  echo "${databricks_sha256}  ${databricks_archive}" | sha256sum --check -
  unzip -qo "${databricks_archive}" databricks -d "${local_bin}"
  chmod 0755 "${local_bin}/databricks"
fi

current_azure_cli="$(
  az version -o json 2>/dev/null | jq -r '."azure-cli"' 2>/dev/null || true
)"
if [[ "${current_azure_cli}" != "${AZURE_CLI_VERSION}" ]]; then
  if ((EUID == 0)); then
    sudo_command=()
  else
    sudo_command=(sudo)
  fi

  "${sudo_command[@]}" install -d -m 0755 /etc/apt/keyrings
  curl -fsSL https://packages.microsoft.com/keys/microsoft.asc |
    gpg --dearmor |
    "${sudo_command[@]}" tee /etc/apt/keyrings/microsoft.gpg >/dev/null
  "${sudo_command[@]}" chmod 0644 /etc/apt/keyrings/microsoft.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/microsoft.gpg] https://packages.microsoft.com/repos/azure-cli/ noble main" |
    "${sudo_command[@]}" tee /etc/apt/sources.list.d/azure-cli.list >/dev/null
  "${sudo_command[@]}" apt-get update
  "${sudo_command[@]}" apt-get install -y --no-install-recommends \
    "azure-cli=${AZURE_CLI_DEB_VERSION}"
  "${sudo_command[@]}" rm -rf /var/lib/apt/lists/*
fi

cd "${repo_root}"
uv lock --check
uv sync --python "${PYTHON_VERSION}" --extra dev --locked
terraform -chdir=infra init -backend=false -input=false

./scripts/cloud-verify.sh
