#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

# This is the same credential-free verification path used by pull-request CI.
./scripts/cloud-verify.sh
