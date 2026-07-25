#!/usr/bin/env bash
set -euo pipefail

# Cached Codex containers can resume on a newer commit. Re-running the
# idempotent setup keeps locked dependencies and tool versions synchronized.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${repo_root}/scripts/codex-cloud-setup.sh"
