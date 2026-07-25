# Migrate CI to a dedicated dev-only identity

The repository currently deploys successfully through the shared
`github-actions-dbx-platform` identity. That principal is a workspace admin in
both `dbx-dev` and `dbx-uat`, so a token minted for this repository has a larger
blast radius than required.

The Terraform configuration now defines a second, repository-owned application
and service principal with the same immutable GitHub OIDC subject. It creates no
secret, requests no API permissions, and receives no Azure RBAC.

The migration is deliberately staged. Do not remove or modify the shared
application because the `dbx-platform` repository also uses it.

## 1. Create the dedicated Entra identity

Run as a human authenticated to the documented tenant and subscription:

```bash
cd infra
terraform init
terraform plan -var-file=terraform.tfvars
terraform apply -var-file=terraform.tfvars

export DEDICATED_CLIENT_ID="$(
  terraform output -raw dedicated_cicd_client_id
)"
export DEDICATED_SP_OBJECT_ID="$(
  terraform output -raw dedicated_cicd_sp_object_id
)"
```

Review the plan before applying. Expected identity changes are:

- one Entra application named `github-actions-aai-dbx-core-starter`;
- its enterprise/service principal;
- one federated identity credential for this repository's immutable `main`
  subject.

There must be no password, client secret, certificate credential, API
permission, or Azure role assignment.

## 2. Register it only in `dbx-dev`

```bash
export DATABRICKS_HOST=https://adb-7405609799238491.11.azuredatabricks.net
export DATABRICKS_AUTH_TYPE=azure-cli

databricks service-principals create \
  --application-id "$DEDICATED_CLIENT_ID" \
  --display-name github-actions-aai-dbx-core-starter \
  --active true
```

Do not add this principal to the `admins` group and do not grant the
`allow-cluster-create` entitlement. Presence in the workspace supplies
workspace access.

Grant only `CAN_USE` on the existing constrained **Job Compute** policy used by
`resources/sample_job.yml`:

```bash
databricks permissions update cluster-policies 0005F2031B6D2319 \
  --json "{
    \"access_control_list\": [{
      \"service_principal_name\": \"$DEDICATED_CLIENT_ID\",
      \"permission_level\": \"CAN_USE\"
    }]
  }"
```

Verify the principal is not registered in UAT:

```bash
DATABRICKS_HOST=https://adb-7405613180844632.12.azuredatabricks.net \
  databricks service-principals list \
  --filter "applicationId eq '$DEDICATED_CLIENT_ID'"
```

The result must be an empty list.

## 3. Cut GitHub over and verify

```bash
gh variable set AZURE_CLIENT_ID \
  -R HuyD0/aai-dbx-core-starter \
  -b "$DEDICATED_CLIENT_ID"

gh workflow run auth-smoke.yml \
  -R HuyD0/aai-dbx-core-starter \
  --ref main

gh workflow run deploy.yml \
  -R HuyD0/aai-dbx-core-starter \
  --ref main
```

Both workflows must succeed. The deploy run is the authorization proof because
it updates the bundle using the constrained service principal.

Also verify that the dedicated principal still has no ARM roles:

```bash
az role assignment list \
  --assignee "$DEDICATED_SP_OBJECT_ID" \
  --all -o table
```

The result must be empty.

## 4. Remove this repository's legacy FIC

After successful dedicated-identity deployment, remove these legacy declarations
from `infra/identity.tf` and apply again:

- `data.azuread_application.cicd`;
- `azuread_application_federated_identity_credential.gha_main`.

Then remove `cicd_app_client_id` from `variables.tf` and `terraform.tfvars`.
Never delete the shared `github-actions-dbx-platform` application or service
principal, and do not remove its `dbx-platform` federated credential or UAT
assignment.
