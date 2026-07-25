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
| 1 | Dedicated CI app + SP | `github-actions-aai-dbx-core-starter` / client id `a7e40167-d3f6-48a9-acd9-7998230cce34` | Terraform (`infra/`) | no |
| 2 | Dedicated federated credential | `gh-aai-dbx-core-starter-main`, subject `repo:HuyD0@151226205/aai-dbx-core-starter@1311037530:ref:refs/heads/main` | Terraform (`infra/`) | no |
| 3 | Project resource group | `rg-aai-dbx-base-template-dev` (eastus2) | Terraform (`infra/`) | new, optional |
| 4 | Databricks SP registration | SP `a7e40167-…` in `dbx-dev` + `CAN_USE` on policy `0005F2031B6D2319` | Databricks CLI | no |
| 5 | GitHub repo variables | Identity/workspace ids, cost tags, owner group, and SDK volume path | `gh` | new |
| 6 | Terraform state | `rg-terraform-state` / `tfstatee18f8286` / container `tfstate` | pre-existing account, new key | reused account |
| 7 | SDK artifact volume | `platform.artifacts.python_packages` | human-run Databricks SQL | new |

No client secrets, PATs, or access keys are created anywhere.

The dedicated migration completed on 2026-07-25. The legacy shared application
still serves `dbx-platform`, but this repository no longer has a federated
credential on it.

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

Expected identity objects are one repository-owned Entra application, its
service principal, and one immutable-`main` federated credential. The legacy
shared application is not referenced or managed.

Verify the credential landed:

```bash
# [agent] read-only
az ad app federated-credential list \
  --id a7e40167-d3f6-48a9-acd9-7998230cce34 \
  --query "[].{name:name, subject:subject}" -o table
```

You should see only `gh-aai-dbx-core-starter-main`.

---

## 3. Register / verify the SP in Databricks — resource #4

**Azure RBAC does not grant in-workspace rights.** The dedicated CI SP must
exist only inside `dbx-dev` with permission to use constrained job compute.

```bash
# Point the CLI at dev via Azure-CLI auth (no token/PAT):
export DATABRICKS_HOST=https://adb-7405609799238491.11.azuredatabricks.net
export DATABRICKS_AUTH_TYPE=azure-cli

# [you] VERIFY it's present (search by application id)
databricks service-principals list \
  --filter "applicationId eq a7e40167-d3f6-48a9-acd9-7998230cce34"

# [you] CREATE only if the list above is empty
databricks service-principals create \
  --application-id a7e40167-d3f6-48a9-acd9-7998230cce34 \
  --display-name github-actions-aai-dbx-core-starter \
  --active

databricks permissions update cluster-policies 0005F2031B6D2319 \
  --json '{"access_control_list":[{"service_principal_name":"a7e40167-d3f6-48a9-acd9-7998230cce34","permission_level":"CAN_USE"}]}'
```

Do not add the dedicated principal to `admins` and do not grant
`allow-cluster-create`. The `deploy` job (§5) is the definitive authorization
test.

> Data-plane note: if `databricks ...` fails locally with a TLS/cert error while
> `az ...` works, your shell is blocking `*.azuredatabricks.net`. Run these on a
> GitHub runner or an unrestricted terminal. The `auth-smoke` workflow (§5) does
> exactly this check in the cloud.

---

## 4. GitHub repo variables (non-secret) — resource #5

All values below are **identifiers or non-sensitive attribution values**, so
they are repository variables rather than secrets.

```bash
# [agent] after `gh auth login`
gh variable set AZURE_CLIENT_ID       -R HuyD0/aai-dbx-core-starter -b a7e40167-d3f6-48a9-acd9-7998230cce34
gh variable set AZURE_TENANT_ID       -R HuyD0/aai-dbx-core-starter -b 7f6a2cf9-5e4e-46ae-95d4-74016c1df1a6
gh variable set AZURE_SUBSCRIPTION_ID -R HuyD0/aai-dbx-core-starter -b ea936670-dda1-4884-8467-49c225bf3e83
gh variable set DATABRICKS_HOST       -R HuyD0/aai-dbx-core-starter -b https://adb-7405609799238491.11.azuredatabricks.net
gh variable set COST_CENTER           -R HuyD0/aai-dbx-core-starter -b CC-1234
gh variable set TEAM                  -R HuyD0/aai-dbx-core-starter -b data-platform
gh variable set OWNER_GROUP           -R HuyD0/aai-dbx-core-starter -b group:data-platform-owners
gh variable set SDK_ARTIFACT_VOLUME   -R HuyD0/aai-dbx-core-starter -b /Volumes/platform/artifacts/python_packages

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
for v in AZURE_CLIENT_ID AZURE_TENANT_ID AZURE_SUBSCRIPTION_ID DATABRICKS_HOST \
  COST_CENTER TEAM OWNER_GROUP SDK_ARTIFACT_VOLUME; do
  gh variable delete "$v" -R HuyD0/aai-dbx-core-starter
done
```

Terraform removes the repository-owned dedicated identity and both
repository-owned federated credentials while leaving the reused application
untouched. The SDK volume and its grants are Databricks objects and must be
revoked separately through the approved platform workflow. There is no secret
to rotate or leak.

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

### 8.2 Shared-SP isolation (finding B — migration complete)

`github-actions-dbx-platform` is reused across this repo and `dbx-platform`. An
audit on 2026-07-25 confirmed it is an admin with unrestricted cluster creation
in both `dbx-dev` and `dbx-uat`. A token minted by this repository can therefore
reach both workspaces.

The dedicated principal is registered only in `dbx-dev`, has only `CAN_USE` on
the Job Compute policy, and has no ARM role. This repository's legacy FIC was
removed after successful auth-smoke and deploy runs. Do not remove the shared
app or its UAT assignment because that can break `dbx-platform`.

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
| C — CODEOWNERS + `main` branch protection | ✅ enforced |
| D — `persist-credentials: false` on checkout | ✅ applied (deploy.yml, ci.yml) |
| G — cluster policy instead of unrestricted creation | ✅ documented (§3) |
| H — `DATABRICKS_HOST` trust assumption | ✅ documented (§8.3) |
| E — pin ALL actions to SHAs | ✅ applied; automated dependency PRs omitted for this POC |
| F — workflow-scoped FIC via `job_workflow_ref` sub-customization | ⬜ deferred (low; needs OIDC sub-customization + matching FIC) |
| B — dedicated per-repo SP | ✅ deployed, dev-only, and verified |
