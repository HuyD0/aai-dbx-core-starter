# aai-dbx-core-starter

A forkable starter template for **AI/data solutions on Azure Databricks** with
**fully automated, keyless CI/CD**. Merge to `main` → GitHub Actions deploys a
Databricks Asset Bundle to the workspace, authenticating with **zero stored
secrets** via GitHub OIDC → Azure federated credentials → Databricks.

It is designed to be handed to AI coding agents (Claude, Codex) — see
[`AGENTS.md`](AGENTS.md) / [`CLAUDE.md`](CLAUDE.md).

## How auth works (no secrets, anywhere)

```
GitHub Actions job (id-token: write)
  │  requests a short-lived OIDC token
  ▼
azure/login@v2  ──exchanges OIDC token via a FEDERATED CREDENTIAL──▶  Azure AD token
  │                     (no client secret involved)
  ▼
az CLI is now authenticated as the CI service principal
  │
  ▼
Databricks CLI  (DATABRICKS_AUTH_TYPE=azure-cli)  ──uses the Azure token──▶  workspace
```

The only things stored in GitHub are **non-secret identifiers** (client id,
tenant id, subscription id, workspace host) as **repo variables** — not secrets.

## Layout

```
.github/workflows/
  ci.yml           PR checks — credential-free (no OIDC, no azure/login)
  auth-smoke.yml   manual — proves the OIDC→Azure→Databricks chain end to end
  deploy.yml       push to main / dispatch — keyless bundle deploy
infra/             Terraform: one-time OIDC identity bootstrap (run locally)
databricks.yml     Databricks Asset Bundle (target: dev → dbx-dev)
resources/         bundle resources (example job)
src/notebooks/     notebook source
docs/cloud-setup.md   every provisioned resource + exact, revocable commands
AGENTS.md / CLAUDE.md  operating guide for AI agents
```

## First-time setup

1. **Bootstrap the identity** (once, locally): [`infra/README.md`](infra/README.md).
2. **Register / verify the SP in Databricks**: [`docs/cloud-setup.md`](docs/cloud-setup.md).
3. **Set repo variables** (`AZURE_CLIENT_ID`, `AZURE_TENANT_ID`,
   `AZURE_SUBSCRIPTION_ID`, `DATABRICKS_HOST`) — see `docs/cloud-setup.md`.
4. **Verify**: run the `auth-smoke` workflow from `main`, then merge to `main`
   and watch `deploy`.

## Day-to-day

Open a PR → `ci` runs (lint/test, no cloud access). Merge to `main` → `deploy`
ships the bundle to `dbx-dev`. That's it.
