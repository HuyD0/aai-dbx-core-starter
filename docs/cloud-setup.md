# Cloud setup — connect externally provisioned resources

This repository configures and verifies the keyless connection used by
`HuyD0/aai-dbx-core-starter`. It does not provision Azure, Entra, Databricks,
or GitHub infrastructure.

The required authentication chain is:

```text
GitHub Actions OIDC
  -> externally managed Entra application and federated credential
  -> Azure CLI authentication
  -> Databricks unified authentication
```

No client secret, PAT, storage key, or API key is required.

## 1. External prerequisites

Before configuring this repository, the platform or identity owner must
provide:

- an Entra application and service principal dedicated to this repository;
- a federated credential for the immutable `main` subject recorded in
  `AGENTS.md`;
- registration of that principal in `dbx-dev`;
- `CAN_USE` on the constrained job-compute policy;
- `USE CATALOG`, `USE SCHEMA`, `READ VOLUME`, and `WRITE VOLUME` on the SDK
  artifact path, with no catalog or workspace administration rights.

These resources are deliberately out of scope for repository setup and CI.
Create, change, or revoke them through the organization's approved platform
process.

The current non-secret identifiers are recorded in `platform-identifiers.json`
and `AGENTS.md`.

## 2. Verify the supplied identity

Use read-only commands from an authenticated administrative shell:

```bash
az ad app federated-credential list \
  --id a7e40167-d3f6-48a9-acd9-7998230cce34 \
  --query "[].{name:name, subject:subject}" -o table

export DATABRICKS_HOST=https://adb-7405609799238491.11.azuredatabricks.net
export DATABRICKS_AUTH_TYPE=azure-cli

databricks service-principals list \
  --filter "applicationId eq a7e40167-d3f6-48a9-acd9-7998230cce34"
```

The federated credential should be named
`gh-aai-dbx-core-starter-main`. The Databricks principal must not be a
workspace admin and must not have unrestricted cluster creation.

## 3. Configure GitHub repository variables

All values are identifiers or non-sensitive attribution values, so use
repository variables rather than secrets:

```bash
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

Do not add a `gh secret set` step.

## 4. Verify end-to-end

The authentication smoke test proves the OIDC exchange. The deploy workflow
proves the principal also has the required Databricks authorization:

```bash
gh workflow run auth-smoke.yml -R HuyD0/aai-dbx-core-starter --ref main
gh workflow run deploy.yml -R HuyD0/aai-dbx-core-starter --ref main
gh run watch -R HuyD0/aai-dbx-core-starter
```

If `azure/login` reports `AADSTS700213`, compare the job's `subject claim`
with the externally managed federated credential. This account uses the
immutable GitHub owner and repository IDs, not only their readable names.

If login reports `No subscriptions found`, do not grant ARM rights as a
workaround. The workflows intentionally use `allow-no-subscriptions: true`
because the principal needs only a tenant token for Databricks authentication.

## 5. Revoke access

Ask the platform or identity owner to remove the repository's federated
credential and Databricks registration through the same external process that
created them. Never delete or mutate the shared legacy
`github-actions-dbx-platform` application.

Remove the repository variables separately:

```bash
for v in AZURE_CLIENT_ID AZURE_TENANT_ID AZURE_SUBSCRIPTION_ID DATABRICKS_HOST \
  COST_CENTER TEAM OWNER_GROUP SDK_ARTIFACT_VOLUME; do
  gh variable delete "$v" -R HuyD0/aai-dbx-core-starter
done
```

The SDK volume and its grants are also external platform resources and must be
revoked through the approved platform workflow.

## 6. Add another deployment target

Before adding a protected GitHub environment or another Databricks target:

1. Ask the platform identity owner for a federated credential whose subject
   matches that environment.
2. Register the principal in the target workspace with least privilege.
3. Add the GitHub protected environment and repository variables.
4. Add the target and deployment job to this repository.

The external identity must exist first because adding `environment:` changes
the GitHub OIDC subject.

## 7. Security boundary

Anyone who can land a commit on `main` or dispatch a credentialed workflow can
trigger a deployment. Protect `main`, require code-owner review, restrict
workflow dispatch, and keep all GitHub Actions pinned to full commit SHAs.
