# infra/ — identity & OIDC bootstrap (Terraform)

This is the **one-time, human-run bootstrap**. It provisions the keyless CI
identity so that afterward GitHub Actions can authenticate to Azure and
Databricks with **no stored secrets**.

## What it manages

| Resource | Owned by TF? | Notes |
|---|---|---|
| Federated credential `gh-aai-dbx-core-starter-main` | ✅ yes | Attached to the **reused** app `github-actions-dbx-platform`. Subject: `repo:HuyD0/aai-dbx-core-starter:ref:refs/heads/main`. |
| App registration `github-actions-dbx-platform` | ❌ no (data source) | Reused, read-only. Not created or destroyed here. |
| `rg-aai-dbx-base-template-dev` | ✅ yes | Optional empty landing zone (`create_project_rg`). |

It deliberately does **not** manage the Databricks-side service-principal
registration — see [`../docs/cloud-setup.md`](../docs/cloud-setup.md).

## Why this runs locally, not in CI

Chicken-and-egg: CI cannot authenticate to Azure until this federated
credential exists. So a human with `az login` runs it once. After that, CI is
fully keyless.

## Run it

```bash
az account set --subscription ea936670-dda1-4884-8467-49c225bf3e83
cd infra

# If using the remote backend (backend.tf), create the container once:
#   az storage container create --account-name tfstatee18f8286 \
#     --name tfstate --auth-mode login

terraform init

# Multi-platform provider lock so the Linux CI runner accepts the same checksums.
# Commit .terraform.lock.hcl afterward.
terraform providers lock -platform=linux_amd64 -platform=darwin_arm64

terraform plan   -var-file=terraform.tfvars    # review carefully
terraform apply  -var-file=terraform.tfvars
```

## Revoke everything

```bash
terraform destroy -var-file=terraform.tfvars
```

This removes the federated credential (and the optional RG). The reused app
registration is untouched.
