# Azure Deployment Plan

> **Status:** Validated

Generated: 2026-07-25

---

## 1. Project Overview

**Goal:** Create a fully prepared Codex Cloud container environment for
`HuyD0/aai-dbx-core-starter` that can make changes and run the complete safe
pre-merge validation suite without local setup. Preserve the repository's
secretless model by delegating authenticated Azure Databricks validation and
deployment to the existing GitHub Actions OIDC path on protected `main`.

**Path:** Modernize Existing

### Current-state evidence

- Azure CLI is authenticated as `Huy.D@hotmail.com` in the intended tenant and
  subscription.
- The `dbx-dev` workspace is reachable with Azure CLI auth.
- The latest `deploy` run on `main` succeeded on 2026-07-25.
- The four required GitHub repository variables exist and contain the documented
  non-secret identifiers.
- GitHub Actions SHA pinning is already enforced.
- The Codex GitHub App does not currently have access to
  `aai-dbx-core-starter`; the environment creation page lists only `agent-eval`
  and `dbx-platform`.
- `main` protection currently has strict `lint-test`, blocks force pushes and
  deletions, and applies to admins, but incorrectly requires zero approvals and
  does not require code-owner review.
- `origin/main` contains the staged dedicated-identity migration, but the
  dedicated Entra application does not yet exist and GitHub still points to the
  legacy shared service principal.

---

## 2. Requirements

| Attribute | Value |
|-----------|-------|
| Classification | Development |
| Scale | Small |
| Budget | Cost-Optimized |
| Subscription | `practisesubscription` (`ea936670-dda1-4884-8467-49c225bf3e83`) — awaiting approval confirmation |
| Location | `eastus2` — awaiting approval confirmation |
| Secrets | None; no client secret, PAT, access key, or GitHub/Codex secret |
| Cloud runtime | Codex Cloud `universal` container |
| Python | 3.12 primary; 3.11 compatibility remains in CI |
| Terraform | 1.12.2 |
| Databricks CLI | 1.6.0 |
| Authenticated deployment | GitHub Actions OIDC from protected `main` only |

---

## 3. Components Detected

| Component | Type | Technology | Path |
|-----------|------|------------|------|
| Databricks Asset Bundle | Deployment bundle | Databricks YAML | `databricks.yml`, `resources/` |
| Sample workload | Databricks notebook/job | Python 3.11 | `src/notebooks/` |
| Credential-free PR checks | CI | GitHub Actions | `.github/workflows/ci.yml` |
| Keyless deploy | CD | GitHub Actions, Azure OIDC, Databricks CLI | `.github/workflows/deploy.yml` |
| Auth smoke test | Operations | GitHub Actions, Azure OIDC | `.github/workflows/auth-smoke.yml` |
| Identity bootstrap | Infrastructure | Terraform, AzureAD/AzureRM | `infra/` |
| Cloud task instructions | Agent configuration | `AGENTS.md` | repository root |

---

## 4. Recipe Selection

**Selected:** Terraform for the already-prepared Entra identity migration,
GitHub CLI/API for repository controls, Databricks CLI/API for workspace
registration, and Codex environment settings for the managed cloud container.

**Rationale:** This extends the repository's existing declarative identity
model and does not create an additional Azure compute platform. Codex Cloud
already supplies the ephemeral container, so Azure Container Apps, AKS, VMs,
storage, Key Vault, and Log Analytics would add cost and credentials without
improving the cloud-task workflow.

---

## 5. Architecture

```text
Codex Cloud task
  -> OpenAI-managed universal container
  -> pinned setup + maintenance scripts
  -> lint, formatting, pytest, Terraform validation, offline bundle checks
  -> proposed diff / pull request
  -> protected main (owner review + required CI)
  -> GitHub Actions OIDC
  -> dedicated Entra service principal (no secret, no ARM RBAC)
  -> dbx-dev only, CAN_USE on Job Compute policy
  -> Databricks bundle validate + deploy
```

### Service Mapping

| Component | Service | SKU / Scope |
|-----------|---------|-------------|
| Cloud development container | Codex Cloud | `universal`, cached |
| Source and CI/CD | GitHub | Existing repository and hosted runners |
| Workload deployment | Azure Databricks | Existing `dbx-dev` workspace |
| Deployment identity | Microsoft Entra ID | Dedicated app/SP/FIC; no secret or ARM role |

### Cloud-runtime trust boundary

Codex Cloud secrets are available only during setup and are removed before the
agent phase. Codex Cloud does not expose a documented GitHub Actions-compatible
OIDC token. Therefore, direct non-interactive Azure/Databricks authentication
inside the agent container cannot be both keyless and supported. Adding a PAT
or client secret would violate this repository's hard rules.

The safe design makes all pre-merge checks credential-free in the container and
keeps the real workspace validation/deployment in the already trusted GitHub
OIDC job. This is an intentional security boundary, not an unconfigured
dependency.

---

## 6. Provisioning Limit Checklist

No new regional Azure resources will be deployed.

| Resource Type | Number to Deploy | Total After Deployment | Limit/Quota | Notes |
|---------------|------------------|------------------------|-------------|-------|
| Azure regional/ARM resources | 0 | Unchanged | Not applicable | Reuses existing resource group and Databricks workspace |
| Entra applications | 1 | Existing directory count + 1 | Directory quota, not regional Azure quota | Cost-free directory object; exact Terraform plan must show only app, SP, and FIC additions |
| Codex Cloud environments | 1 | Existing Codex environments + 1 | Account entitlement, not Azure quota | Environment creation UI is available |

**Status:** ✅ No Azure regional capacity or paid compute provisioning required.

---

## 7. Planned Changes

### Repository

- Fast-forward the local checkout to `origin/main` while preserving unrelated
  untracked user files.
- Add an idempotent, pinned cloud setup script for Python 3.11, Terraform
  1.12.2, Databricks CLI 1.6.0, and the dev dependencies.
- Add an idempotent maintenance script for cached cloud containers.
- Add one cloud verification entry point shared by Codex Cloud and PR CI.
- Extend tests to enforce the cloud-runtime and keyless trust boundaries.
- Update `AGENTS.md`/README documentation so future cloud tasks run the correct
  checks and never attempt to add credentials.

### Azure and Databricks

- Review the Terraform plan and apply only the already-staged dedicated Entra
  app, service principal, and immutable-`main` federated credential.
- Register that application only in `dbx-dev`.
- Grant only `CAN_USE` on Databricks policy `0005F2031B6D2319`; do not add it to
  admins and do not grant `allow-cluster-create`.
- Confirm it is absent from UAT and has no Azure ARM role assignments.
- Update the GitHub `AZURE_CLIENT_ID` repository variable to the dedicated
  client ID.
- Run `auth-smoke` and `deploy` from `main`; roll the variable back to the
  legacy ID if either fails.
- Remove only this repository's legacy federated credential after the
  dedicated deployment succeeds. Never modify or delete the shared
  `github-actions-dbx-platform` application or its `dbx-platform` credential.

### GitHub and Codex Cloud

- Grant the ChatGPT/Codex GitHub App access to only
  `HuyD0/aai-dbx-core-starter`.
- Correct `main` protection to require one approval, code-owner review,
  dismissal of stale approvals, last-push approval, conversation resolution,
  linear history, strict `lint-test`, admin enforcement, and no force pushes or
  deletions.
- Create a Codex Cloud environment named `aai-dbx-core-starter`.
- Configure Python 3.12, caching, manual setup and maintenance scripts, and the
  four documented non-secret identifiers as environment variables.
- Configure no secrets.
- Keep agent internet access off unless the clean-container verification proves
  a narrowly allowlisted dependency is required during the agent phase.

---

## 8. Validation and Deployment

1. Run the existing local checks: `ruff check .`, `black --check .`,
   `pytest -q`, and `terraform fmt -check -recursive infra`.
2. Validate Terraform with an isolated plugin/cache directory and
   `terraform init -backend=false`.
3. Build/test the setup scripts in a clean Ubuntu 24.04 container matching the
   Codex universal base assumptions.
4. Confirm pinned tool versions and run the shared cloud verification command
   inside that clean container.
5. Invoke the repository's Azure readiness validation workflow.
6. Apply the identity migration, verify zero ARM RBAC and dev-only Databricks
   registration, then cut over the GitHub variable.
7. Run and watch `auth-smoke` and `deploy` from `main`.
8. Create the Codex environment and use its interactive terminal to run the
   same verification command.
9. Start an actual Codex Cloud task against the repository, make a harmless
   temporary validation-only change, verify checks, and discard the temporary
   diff.
10. Record command outputs, workflow URLs, and the final environment settings.

---

## 9. Rollback

- Codex environment: delete the newly created environment or remove the
  repository from its GitHub App selection.
- GitHub identity cutover: restore `AZURE_CLIENT_ID` to
  `b74a6820-d0ac-454f-8c32-02141cba3c8a`.
- Dedicated identity: remove its `dbx-dev` registration and use Terraform to
  remove only the dedicated app/SP/FIC.
- Legacy identity: preserve it until the dedicated auth smoke and deploy both
  succeed; never delete the shared application.
- Repository: changes will be isolated on a feature branch and proposed through
  a pull request; no direct push to protected `main`.

---

## 10. Execution Checklist

### Phase 1: Planning

- [x] Analyze workspace.
- [x] Gather requirements.
- [x] Detect Azure subscription and location.
- [x] Prepare resource inventory and determine quota applicability.
- [x] Scan codebase.
- [x] Select recipe.
- [x] Plan architecture.
- [x] User confirms the subscription, region, GitHub App access change,
  branch-protection change, and this plan.

### Phase 2: Execution

- [x] Fast-forward from `origin/main`.
- [x] Generate cloud-container artifacts.
- [x] Pass local cloud-container verification.
- [x] Pass exact Codex universal container verification.
- [x] Update plan status to `Ready for Validation`.

### Phase 3: Validation

- [x] Invoke Azure validation guidance.
- [x] Pass local, clean-container, Terraform, and security checks.
- [x] Update plan status to `Validated` and record proof.

### Phase 4: Deployment

- [ ] Invoke Azure deployment guidance.
- [ ] Provision and verify the dedicated identity.
- [ ] Correct GitHub protection and grant repository access to Codex.
- [ ] Create and verify the Codex Cloud environment.
- [ ] Pass authenticated GitHub OIDC smoke and Databricks deployment.
- [ ] Update plan status to `Deployed`.

---

## 11. Validation Proof

| Check | Command Run | Result | Timestamp |
|-------|-------------|--------|-----------|
| Local cloud suite | `./scripts/cloud-verify.sh` | ✅ 34 tests; lint, format, build, Terraform, Databricks schema, YAML pass | 2026-07-25T22:23:56Z |
| Exact Codex setup | `docker run ... ghcr.io/openai/codex-universal:latest ... ./scripts/codex-cloud-setup.sh` | ✅ Checksums and pinned installs pass on arm64 universal image | 2026-07-25T22:23:56Z |
| Offline agent phase | `docker network disconnect bridge aai-codex-cloud-offline` then `docker exec ... ./scripts/cloud-verify.sh` | ✅ Complete suite passes with container network disconnected | 2026-07-25T22:23:56Z |
| Terraform initialize | `terraform -chdir=infra init -input=false` | ✅ Remote backend and locked providers accessible | 2026-07-25T22:23:56Z |
| Terraform format/syntax | `terraform fmt -check -recursive infra` and `terraform -chdir=infra validate` | ✅ Pass | 2026-07-25T22:23:56Z |
| Terraform state | `terraform -chdir=infra state list` | ✅ Legacy FIC and project RG present; dedicated identity absent | 2026-07-25T22:23:56Z |
| Full Terraform preview | `terraform -chdir=infra plan -var-file=terraform.tfvars -out=/tmp/aai-dbx-cloud-env.tfplan` | ✅ 3 add, 1 unrelated RG tag update, 0 destroy | 2026-07-25T22:23:56Z |
| Scoped identity preview | `terraform -chdir=infra plan ... -target=... -out=/tmp/aai-dbx-dedicated-identity.tfplan` | ✅ Exactly 3 add, 0 change, 0 destroy | 2026-07-25T22:23:56Z |
| Static identity/RBAC | Terraform plan JSON plus `rg` review of `infra/*.tf` | ✅ Empty passwords/API permissions; no ARM role assignments | 2026-07-25T22:23:56Z |
| Azure policies | `az policy assignment list --disable-scope-strict-match` | ✅ Security Center/MFA policies compatible; West Europe block irrelevant to eastus2 | 2026-07-25T22:23:56Z |
| Template resolution | `rg '\{\{ *\.Env\.' infra --glob '*.tf' --glob '*.tfvars.json'` | ✅ No unresolved variables | 2026-07-25T22:23:56Z |

### Role Assignment Verification

- Status: Verified.
- Identity: `github-actions-aai-dbx-core-starter`.
- Azure ARM roles: none declared, as required.
- Databricks rights to add after registration: workspace presence plus
  `CAN_USE` on policy `0005F2031B6D2319`; no admin group and no
  `allow-cluster-create`.
- Unity Catalog release rights are intentionally separate and are not required
  for the deployment smoke test in this task.

**Validated by:** Azure validation workflow
**Validation timestamp:** 2026-07-25T22:23:56Z

---

## 12. Files to Generate

| File | Purpose | Status |
|------|---------|--------|
| `.azure/deployment-plan.md` | Source-of-truth plan and evidence | ✅ Planning |
| `scripts/codex-cloud-setup.sh` | Idempotent pinned setup | ⏳ |
| `scripts/codex-cloud-maintenance.sh` | Cached-container refresh | ⏳ |
| `scripts/cloud-verify.sh` | Shared credential-free verification | ⏳ |
| `.github/workflows/ci.yml` | Run shared verification in PR CI | ⏳ |
| `tests/test_smoke.py` | Enforce cloud/security invariants | ⏳ |
| `AGENTS.md` and `README.md` | Durable cloud-task operating instructions | ⏳ |

---

## 13. Next Step

Approve this plan and confirm the exact subscription/location plus the two
permission changes. Execution will then proceed without requiring local scripts
or additional setup from the user.
