#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

if [[ ! -x .venv/bin/python ]]; then
  echo "The locked development environment is missing; run \`make install\`." >&2
  exit 2
fi

# Keep the commit gate credential-free and fast. The pre-push hook runs the
# complete build, workflow-security, Terraform, and schema verification tier.
make check-templates format-check test
