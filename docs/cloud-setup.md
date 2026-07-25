# Cloud setup — every resource, exact commands, fully revocable

This is the authoritative record of what is provisioned for
`HuyD0/aai-dbx-core-starter` and how to reproduce or tear it down. Nothing here
stores a secret; auth is keyless (GitHub OIDC → Azure federated credential →
Databricks).

Legend: **[you]** = a human with `az login` + Databricks account admin, run once,
locally. **[agent]** = safe to run from the assistant's shell. **[CI]** = runs on
a GitHub runner with zero stored secrets.

---

## 0. Inventory

| # | Resource | Identifier | Created by | Reused? |
|---|---|---|---|---|
| 1 | App registration (CI OIDC identity) | `github-actions-dbx-platform` / client id `b74a6820-d0ac-454f-8c32-02141cba3c8a` | pre-existing | ✅ reused (not owned here) |
| 2 | Federated credential | `gh-aai-dbx-core-starter-main`, subject `repo:HuyD0@151226205/aai-dbx-core-starter@1311037530:ref:refs/heads/main` (immutable form) | Terraform (`infra/`) | new |
| 2a | Dedicated CI app + SP + FIC (migration target) | `github-actions-aai-dbx-core-starter` | Terraform (`infra/`) | pending human apply |
| 3 | Project resource group | `rg-aai-dbx-base-template-dev` (eastus2) | Terraform (`infra/`) | new, optional |
| 4 | Databricks SP registration | SP `b74a6820-…` in `dbx-dev` + workspace-access entitlement | Databricks CLI | verify (reuse likely) |
| 5 | GitHub repo variables | `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID`, `DATABRICKS_HOST` | `gh` | new |
| 6 | Terraform state | `rg-terraform-state` / `tfstatee18f8286` / container `tfstate` | pre-existing account, new key | reused account |

No client secrets, PATs, or access keys are created anywhere.

The shared identity is currently a workspace admin in both `dbx-dev` and
`dbx-uat`. Complete
[`dedicated-identity-migration.md`](dedicated-identity-migration.md) to replace
it for this repository without modifying the shared app used by `dbx-platform`.

---

## 1. Prerequisites

```bash
# [you] GitHub CLI — needs repo + workflow scopes to push workflows and set vars
gh auth login -h github.com -s repo,workflow
gh auth status

# [you] Azure CLI on the right subscription
az login
az account set --subscription ea936670-dda1-4884-8467-49c225bf3e83
az account show -o table

# [you] Confirm you are a Databricks ACCOUNT admin (needed to register an SP in a
# Unity-Catalog-enabled workspace). If this errors, get account-admin first.
databricks account service-principals list \
  --account-id <your-databricks-account-id> 2>/dev/null | head
```

---

## 2. Bootstrap the OIDC identity (Terraform) — resources #2, #3

Run once. This is the chicken-and-egg breaker: CI cannot authenticate until the
federated credential exists, so a human creates it.

```bash
# [you] optional: create the state container the first time
az storage container create \
  --account-name tfstatee18f8286 --name tfstate --auth-mode login

cd infra
terraform init

# Generate a MULTI-PLATFORM provider lock so CI's Linux runner accepts the same
# checksums your local (darwin_arm64) init produced. Commit the resulting
# .terraform.lock.hcl. Skipping this makes CI's `terraform init` fail on checksums.
terraform providers lock -platform=linux_amd64 -platform=darwin_arm64

terraform plan  -var-file=terraform.tfvars    # REVIEW the plan before applying
terraform apply -var-file=terraform.tfvars
```

> If `terraform plan` errors on `application_id`, your azuread provider resolved
> to a version where the attribute differs — see the fallback comments in
> `identity.tf` and adjust before applying. This is the one line that can't be
> validated offline.

For the original bootstrap, the expected new objects were one federated
credential and the optional resource group. The current migration also creates
one repository-owned Entra application, its service principal, and a second
federated credential. The legacy shared application remains a **data source** —
read-only, never modified.

Verify the credential landed:

```bash
# [agent] read-only
az ad app federated-credential list \
  --id b74a6820-d0ac-454f-8c32-02141cba3c8a \
  --query "[].{name:name, subject:subject}" -o table
```

You should see `gh-aai-dbx-core-starter-main` alongside the pre-existing
`dbx-platform` credential.

---

## 3. Register / verify the SP in Databricks — resource #4

**Azure RBAC does not grant in-workspace rights.** The CI SP must exist inside
`dbx-dev` with permission to deploy. Because we reuse `github-actions-dbx-platform`
(already used against `dbx-platform`), it is very likely already registered — so
verify first, only create if missing.

```bash
# Point the CLI at dev via Azure-CLI auth (no token/PAT):
export DATABRICKS_HOST=https://adb-7405609799238491.11.azuredatabricks.net
export DATABRICKS_AUTH_TYPE=azure-cli

# [you] VERIFY it's present (search by application id)
databricks service-principals list \
  --filter "applicationId eq b74a6820-d0ac-454f-8c32-02141cba3c8a"

# [you] CREATE only if the list above is empty
databricks service-principals create \
  --application-id b74a6820-d0ac-454f-8c32-02141cba3c8a \
  --display-name github-actions-dbx-platform \
  --active true

# [you] LEAST-PRIVILEGE entitlements so it can deploy bundles / create job
# clusters. (workspace-access is implied by presence.) Grant via the workspace
# admin UI (Settings → Identity and access → Service principals → entitlements)
# or the entitlements API. Keep it to:
#   - Workspace access
#   - If jobs use new job clusters, scope cluster creation to a CLUSTER POLICY
#     (fixed node types, NO arbitrary init scripts, capped autoscale) or a
#     pre-created constrained instance pool — do NOT grant "unrestricted cluster
#     creation". Unrestricted creation lets anyone holding the SP token run
#     arbitrary code (init scripts / node config) as the SP.
```

> Reuse does not guarantee deploy rights. `github-actions-dbx-platform` may serve
> the `dbx-platform` repo against a *different* workspace, so confirm it is
> present in **dbx-dev** *and* can create jobs there — authenticating is not the
> same as being authorized. The `deploy` job (§5) is the definitive test.

> Data-plane note: if `databricks ...` fails locally with a TLS/cert error while
> `az ...` works, your shell is blocking `*.azuredatabricks.net`. Run these on a
> GitHub runner or an unrestricted terminal. The `auth-smoke` workflow (§5) does
> exactly this check in the cloud.

---

## 4. GitHub repo variables (non-secret) — resource #5

All four are **identifiers, not secrets**, so they are repo *variables*.

```bash
# [agent] after `gh auth login`
gh variable set AZURE_CLIENT_ID       -R HuyD0/aai-dbx-core-starter -b b74a6820-d0ac-454f-8c32-02141cba3c8a
gh variable set AZURE_TENANT_ID       -R HuyD0/aai-dbx-core-starter -b 7f6a2cf9-5e4e-46ae-95d4-74016c1df1a6
gh variable set AZURE_SUBSCRIPTION_ID -R HuyD0/aai-dbx-core-starter -b ea936670-dda1-4884-8467-49c225bf3e83
gh variable set DATABRICKS_HOST       -R HuyD0/aai-dbx-core-starter -b https://adb-7405609799238491.11.azuredatabricks.net

gh variable list -R HuyD0/aai-dbx-core-starter
```

No `gh secret set` commands exist in this project — by design.

---

## 5. Verify end-to-end in the cloud

```bash
# [agent] push must have happened first; run the auth smoke test from main:
gh workflow run auth-smoke.yml -R HuyD0/aai-dbx-core-starter --ref main
gh run watch -R HuyD0/aai-dbx-core-starter
```

`auth-smoke` proves the **authentication** chain only: `azure/login` exchanged
the OIDC token (no secret), `az account show` printed the SP, and
`databricks current-user me` returned the SP. It does **not** prove deploy
*authorization* — `current-user me` succeeds for any principal with mere
workspace access.

**End-to-end is proven by the `deploy` job going green**, because that runs
`databricks bundle deploy -t dev`, which actually creates the job and writes the
bundle folder. Merge to `main` (or `gh workflow run deploy.yml --ref main`) and
confirm it succeeds. If deploy 403s while smoke passed, the SP is authenticated
but under-entitled — revisit §3.

**If `azure/login` fails with `AADSTS700213: No matching federated identity
record`,** the FIC subject doesn't match the token. Read the exact string from
the job log line `subject claim - ...` and set the FIC to it. This account mints
the **immutable** form (`repo:<owner>@<owner_id>/<repo>@<repo_id>:...`); the
readable form will not match. The numeric ids are already wired in
`infra/terraform.tfvars` (`github_owner_id`, `repo_id`).

**If `azure/login` fails with `No subscriptions found`,** the SP authenticated
but has no ARM RBAC (by design). Both workflows set `allow-no-subscriptions:
true` and omit `subscription-id`, so `az login` succeeds on the tenant token
alone. Do not grant the SP a subscription role to work around this.

---

## 6. Revoke / tear down

```bash
# Remove the federated credential (and optional RG). App registration untouched.
cd infra && terraform destroy -var-file=terraform.tfvars

# (Optional) remove the SP from the workspace — do NOT delete the shared app reg
export DATABRICKS_HOST=https://adb-7405609799238491.11.azuredatabricks.net
export DATABRICKS_AUTH_TYPE=azure-cli
databricks service-principals delete <sp-databricks-id>

# Remove GitHub variables
for v in AZURE_CLIENT_ID AZURE_TENANT_ID AZURE_SUBSCRIPTION_ID DATABRICKS_HOST; do
  gh variable delete "$v" -R HuyD0/aai-dbx-core-starter
done
```

Because the CI identity is nothing but a federated credential on a reused app,
revocation is a single `terraform destroy` — there is no secret to rotate or
leak.

---

## 7. Adding a prod target later

1. Add a target in `databricks.yml` for `prod` → `dbx-uat`
   (`https://adb-7405613180844632.12.azuredatabricks.net`).
2. Decide the gate: a GitHub `production` environment (with required reviewers).
3. **Add a matching FIC** in `infra/identity.tf` with subject
   `repo:HuyD0@151226205/aai-dbx-core-starter@1311037530:environment:production`
   (immutable form), apply it.
4. Register the SP in `dbx-uat` (repeat §3 against the uat host).
5. Add a `deploy-prod` job that sets `environment: production` and deploys `-t prod`.

---

## 8. Security model, required hardening & status

The keyless model has one load-bearing assumption: **whoever can land a commit
on `main` (or run `workflow_dispatch`) can trigger a credentialed deploy.** There
is no secret to steal, but that boundary must be enforced on the GitHub side.

### 8.1 REQUIRED — protect `main` (do this)

```bash
# Require PR + 1 review (incl. code owners), block direct/force pushes, apply to admins.
gh api -X PUT repos/HuyD0/aai-dbx-core-starter/branches/main/protection \
  -H "Accept: application/vnd.github+json" \
  -F 'required_pull_request_reviews[required_approving_review_count]=1' \
  -F 'required_pull_request_reviews[require_code_owner_reviews]=true' \
  -F 'required_pull_request_reviews[dismiss_stale_reviews]=true' \
  -F 'required_pull_request_reviews[require_last_push_approval]=true' \
  -F 'enforce_admins=true' \
  -F 'required_status_checks[strict]=true' -f 'required_status_checks[contexts][]=lint-test' \
  -F 'restrictions=null' \
  -F 'required_linear_history=true' \
  -F 'required_conversation_resolution=true' \
  -F 'allow_force_pushes=false' \
  -F 'allow_deletions=false'
```

Also restrict who can run `workflow_dispatch` (Settings → Actions, or limit repo
write access) — branch protection alone does not gate manual dispatch. The
`.github/CODEOWNERS` file makes owner review mandatory for every path once
"require code owner reviews" is on.

After the SHA-pinning hardening PR is merged, require immutable action
references repository-wide:

```bash
gh api -X PUT repos/HuyD0/aai-dbx-core-starter/actions/permissions \
  -F 'enabled=true' \
  -f 'allowed_actions=all' \
  -F 'sha_pinning_required=true'
```

Do not enable this before the pinning PR reaches `main`; otherwise the existing
tag-based workflows on `main` will stop running.

### 8.2 Shared-SP isolation (finding B — dedicated migration required)

`github-actions-dbx-platform` is reused across this repo and `dbx-platform`. An
audit on 2026-07-25 confirmed it is an admin with unrestricted cluster creation
in both `dbx-dev` and `dbx-uat`. A token minted by this repository can therefore
reach both workspaces.

Do not remove the shared app or its UAT assignment because that can break
`dbx-platform`. Follow
[`dedicated-identity-migration.md`](dedicated-identity-migration.md) to create a
repository-owned principal, register it only in `dbx-dev`, grant only `CAN_USE`
on the constrained Job Compute policy, repoint `AZURE_CLIENT_ID`, verify, and
then remove only this repository's legacy FIC.

### 8.3 `DATABRICKS_HOST` is a bearer-token sink (finding H)

It is a non-secret repo *variable*, but the `azure-cli` auth sends a live AAD
token to whatever host it names (the token is for the first-party
`AzureDatabricks` resource and is **not** host-bound). A repo **admin** could
repoint it to capture a token — no workflow edit needed. Treat repo-admin as a
trusted role; audit changes to this variable.

### 8.4 Hardening status

| Fix | Status |
|---|---|
| A — pin `databricks/setup-cli` to a commit SHA | ✅ applied — `@8b7b124…` (v1.9.0) in deploy.yml + auth-smoke.yml |
| C — CODEOWNERS + `main` branch protection | ✅ CODEOWNERS added; branch protection = §8.1 (run it) |
| D — `persist-credentials: false` on checkout | ✅ applied (deploy.yml, ci.yml) |
| G — cluster policy instead of unrestricted creation | ✅ documented (§3) |
| H — `DATABRICKS_HOST` trust assumption | ✅ documented (§8.3) |
| E — pin ALL actions to SHAs + Dependabot | ✅ applied |
| F — workflow-scoped FIC via `job_workflow_ref` sub-customization | ⬜ deferred (low; needs OIDC sub-customization + matching FIC) |
| B — dedicated per-repo SP | 🟨 Terraform prepared; human apply/cutover required |
