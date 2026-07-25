# AGENTS.md — operating guide for AI coding agents

This file is the shared source of truth for **both Claude and Codex**
(`CLAUDE.md` imports it). Read it fully before acting. It documents the auth
chain, the exact provisioned identifiers, the safety model, and the rules of
engagement for this repo.

---

## 1. What this repo is

A keyless CI/CD starter for Azure Databricks. Code merged to `main` deploys a
Databricks Asset Bundle to the **dev** workspace via GitHub Actions, using
GitHub OIDC federated with Azure AD — **no secrets are stored anywhere**.

## 2. The auth chain (memorize this)

```
GitHub Actions (permissions: id-token: write)
  → OIDC token (immutable subject:
    repo:HuyD0@151226205/aai-dbx-core-starter@1311037530:ref:refs/heads/main)
  → azure/login@v2 exchanges it against a FEDERATED CREDENTIAL on the app
    registration  →  Azure AD access token (no client secret)
  → az CLI authenticated as the CI service principal
  → Databricks CLI with DATABRICKS_AUTH_TYPE=azure-cli uses that token
  → databricks bundle deploy -t dev
```

## 3. Provisioned identifiers (non-secret)

| Thing | Value |
|---|---|
| GitHub repo | `HuyD0/aai-dbx-core-starter` |
| Azure tenant | `7f6a2cf9-5e4e-46ae-95d4-74016c1df1a6` |
| Azure subscription | `ea936670-dda1-4884-8467-49c225bf3e83` (`practisesubscription`) |
| CI app registration (**reused**) | `github-actions-dbx-platform` |
| CI app **client id** (`AZURE_CLIENT_ID`) | `b74a6820-d0ac-454f-8c32-02141cba3c8a` |
| CI app SP object id | `f1ae1583-6b35-4d6c-a7c1-305034983307` |
| Dedicated CI app (**migration target**) | `github-actions-aai-dbx-core-starter` — ids assigned by Terraform apply |
| Federated credential | `gh-aai-dbx-core-starter-main` |
| FIC subject (immutable form) | `repo:HuyD0@151226205/aai-dbx-core-starter@1311037530:ref:refs/heads/main` |
| Databricks dev workspace | `dbx-dev` — `https://adb-7405609799238491.11.azuredatabricks.net` (id `7405609799238491`) |
| Optional project RG | `rg-aai-dbx-base-template-dev` (eastus2) |
| Terraform state | `rg-terraform-state` / `tfstatee18f8286` / container `tfstate` / key `aai-dbx-base-template/dev.tfstate` |

These are **identifiers, not secrets**. There are **no client secrets** in this
project. If you ever find yourself creating or pasting a client secret, PAT, or
access key, **stop** — that is a design violation here.

## 4. Safety model — hard rules

1. **No secrets in git, ever.** Auth is OIDC-only. The four repo *variables*
   (`AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID`,
   `DATABRICKS_HOST`) are non-secret ids. Do not add repo/environment *secrets*.
2. **Never put `azure/login`, `id-token: write`, or any credential step on a
   `pull_request` trigger.** PRs (including forks) must stay credential-free.
   `ci.yml` has `permissions: contents: read` and no OIDC — keep it that way.
3. **The deploy/smoke jobs must not declare a GitHub `environment:`.** The FIC
   subject is the branch-ref form; adding an environment changes the OIDC
   subject to `:environment:<name>` and breaks the exchange. To gate on an
   environment, first add a *matching* FIC in `infra/identity.tf`, then set it.
4. **Least privilege.** The dedicated CI SP must have **no ARM RBAC**, must be
   registered only in `dbx-dev`, must not be a workspace admin, and may use only
   the constrained Job Compute policy. Do not grant it subscription/RG roles or
   unrestricted cluster creation to "make something work" — solve it with
   Databricks object permissions instead. The legacy shared identity does not
   meet this boundary; complete `docs/dedicated-identity-migration.md`.
5. **The reused app is shared.** `github-actions-dbx-platform` also serves the
   `dbx-platform` repo. Do not delete it, rotate it, or change its other
   credentials. Terraform owns only this repo's legacy FIC plus the new
   dedicated app; it never owns the shared app.
6. **Bootstrap is human-run.** `infra/` (identity) and the Databricks SP
   registration are run once by a human with `az login` + Databricks account
   admin. CI never runs `terraform apply` and has no rights to.
7. **`main` protection IS the security boundary.** Anyone who can land a commit
   on `main` (or run `workflow_dispatch`) can trigger a credentialed deploy —
   there is no secret, so the only gate is who writes to `main`. Branch
   protection (PR + code-owner review, no direct/force push, enforced on admins)
   is a hard prerequisite, not optional — see `docs/cloud-setup.md §8.1`. Do not
   weaken it. `.github/CODEOWNERS` requires owner review for every path.
8. **Shared identity = shared blast radius.** The legacy shared SP is registered
   in both `dbx-dev` and `dbx-uat`; do not expand or mutate it. Migrate this repo
   to its dedicated dev-only SP, then remove only this repo's legacy FIC. Never
   delete the shared application or its UAT assignment. Every action is pinned
   to a commit SHA so mutable tags cannot exfiltrate the live token.

## 5. How to work in this repo

- **Add a Databricks job/pipeline:** create/extend a file in `resources/*.yml`;
  put code under `src/`. Validate with `databricks bundle validate -t dev`.
- **Cost attribution is mandatory.** Every job cluster (`new_cluster.custom_tags`)
  must carry the attribution keys `cost_center`, `team`, `owner`, `project`,
  `environment` — cluster `custom_tags` are what reach Azure VM billing and the
  `system.billing.usage` table for DBU chargeback. Bundle-wide `presets.tags`
  (in `databricks.yml`) applies the same keys to jobs/pipelines. Values come from
  the `cost_center`/`team`/`owner` **bundle variables**; set them per instance via
  a target override, `--var`, or `BUNDLE_VAR_*` — never hardcode a real cost
  center in a resource file, and never use a *secret* for them (they are
  non-secret ids). The infra tags in `infra/main.tf` mirror the same keys so
  Azure resources and Databricks compute roll up together. If the constrained Job
  Compute policy ever fixes/forbids one of these tag keys, align the bundle key
  with the policy rather than dropping attribution.
- **Change infra/identity:** edit `infra/*.tf`; a human runs `terraform plan`
  then `apply`. Never bypass with imperative `az ad ...` mutations.
- **Local Databricks auth** (for `bundle validate` etc.): `az login`, then
  `export DATABRICKS_HOST=https://adb-7405609799238491.11.azuredatabricks.net`
  and `DATABRICKS_AUTH_TYPE=azure-cli`. No profile/token needed.
- **Before committing:** `ruff check .`, `black .`, `pytest -q`,
  `terraform fmt -recursive infra`.
- **Verifying auth:** run the `auth-smoke` workflow (Actions tab) from `main`.

## 6. Environment quirks to know

- Some sandboxes/proxies block the Databricks **data plane**
  (`*.azuredatabricks.net`) while allowing Azure ARM. If `databricks ...` fails
  with a TLS/cert error locally but `az ...` works, that's the data plane being
  blocked — run Databricks steps on a GitHub runner or an unrestricted shell.
- `workflow_dispatch` must be run from `main` so the OIDC subject matches the
  FIC.
- **`azure/login` needs `allow-no-subscriptions: true`.** The CI SP has no ARM
  RBAC, so `az login` finds no subscription and errors `No subscriptions found`
  without this flag. Do not "fix" it by granting the SP a subscription role —
  that breaks the Databricks-only least-privilege model. The tenant token is all
  the Databricks `azure-cli` auth needs; `subscription-id` is not passed to login.
- GitHub mints the **immutable** OIDC subject here
  (`repo:<owner>@<owner_id>/<repo>@<repo_id>:...`). If `azure/login` ever errors
  `AADSTS700213`, the FIC subject and the token's `sub` claim have diverged —
  read the exact `subject claim` from the job log and match it.

## 7. Reproduce / revoke

Everything is in [`docs/cloud-setup.md`](docs/cloud-setup.md) with exact
commands, including how to tear it all down.
