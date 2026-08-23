---
name: aai-verify-release
description: Perform a dry-run verification of an aai-core or template release candidate, including versions, compatibility, locks, candidate-wheel isolation, generated projects, examples, security checks, checksums, and immutable publication preconditions. Use before tagging, publishing, deploying, or declaring a release ready.
---

# Verify an AAI Release

Default to read-only or build-only verification. Do not tag, publish, deploy, or
change external state without explicit authorization.

## Workflow

1. Read `AGENTS.md`, release workflows, `compatibility.json`, dependency policy,
   and the release validator. Inspect the complete worktree and identify the
   candidate version and intended artifacts.
2. Fail early on inconsistent versions, unreviewed generated drift, unexpected
   credentials, or an existing immutable artifact with the same version.
3. Run the repository-owned gates rather than reproducing them:

   ```text
   make check-template-locks
   make check
   ./scripts/cloud-verify.sh
   python scripts/validate_release.py --wheel dist
   ```

4. Confirm tests installed the candidate wheel outside the checkout and that
   every affected template branch rendered and tested against it.
5. Verify SHA-pinned actions, CodeQL results, dependency findings, checksums,
   SBOM/lock evidence when available, and the absence of secrets in artifacts.
6. Treat credentialed bundle validation and live canaries as separate protected
   gates. Run them only when explicitly authorized with existing keyless
   identities; never provision resources or broaden privileges.

## Release decision

Require all mandatory evidence to pass. A skipped optional preview capability
must report `NOT_CONFIGURED`, never pass. Require the repository's declared
consecutive-green canary window before publication.

Verify that publication will:

- Reuse the exact validated wheel rather than rebuild it.
- Include an artifact checksum and available dependency/lock provenance.
- Refuse to overwrite an existing SDK or template version.
- Preserve the documented SDK/template compatibility window.
- Keep pull-request workflows credential-free.

## Handoff

Return a compact table of each gate, command or evidence source, result, and
remaining protected action. Distinguish a locally verified candidate from a
release approved for publication. Never claim CodeQL, live Databricks, or
MLflow validation without its actual result.
