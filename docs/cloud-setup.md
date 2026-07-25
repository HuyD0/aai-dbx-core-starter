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
| 2 | Federated credential | `gh-aai-dbx-core-starter-main`, subject `repo:HuyD0/aai-dbx-core-starter:ref:refs/heads/main` | Terraform (`infra/`) | new |
| 3 | Project resource group | `rg-aai-dbx-base-template-dev` (eastus2) | Terraform (`infra/`) | new, optional |
| 4 | Databricks SP registration | SP `b74a6820-…` in `dbx-dev` + workspace-access entitlement | Databricks CLI | verify (reuse likely) |
| 5 | GitHub repo variables | `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID`, `DATABRICKS_HOST` | `gh` | new |
| 6 | Terraform state | `rg-terraform-state` / `tfstatee18f8286` / container `tfstate` | pre-existing account, new key | reused account |

No client secrets, PATs, or access keys are created anywhere.

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

Expected new objects: one `azuread_application_federated_identity_credential`
and (optionally) `rg-aai-dbx-base-template-dev`. The app registration itself is a
**data source** — read-only, never modified.

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

# [you] Minimal entitlements so it can deploy bundles / create job clusters.
# (workspace-access is implied by presence; add cluster-create for job clusters.)
# Grant via the workspace admin UI (Settings → Identity and access → Service
# principals → entitlements) or the entitlements API. Keep it to:
#   - Workspace access
#   - Allow unrestricted cluster creation  (only if jobs use new job clusters)
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
   `repo:HuyD0/aai-dbx-core-starter:environment:production`, apply it.
4. Register the SP in `dbx-uat` (repeat §3 against the uat host).
5. Add a `deploy-prod` job that sets `environment: production` and deploys `-t prod`.
