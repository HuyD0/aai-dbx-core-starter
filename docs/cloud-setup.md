# Cloud setup — connect externally provisioned resources

This repository configures and verifies its own keyless connection. It does
not provision Azure, Entra, Databricks, or GitHub infrastructure.

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
: "${CI_CLIENT_ID:?Set CI_CLIENT_ID to the application ID issued by the identity owner}"
: "${FIC_NAME:?Set FIC_NAME to the issued federated-credential name}"

az ad app federated-credential list \
  --id "$CI_CLIENT_ID" \
  --query "[].{name:name, subject:subject}" -o table

source scripts/platform-env.sh

databricks service-principals list \
  --filter "applicationId eq $CI_CLIENT_ID"
```

Confirm the returned credential name equals `$FIC_NAME`. The Databricks
principal must not be a workspace admin and must not have unrestricted cluster
creation.

## 3. Configure GitHub repository variables

All values are identifiers or non-sensitive attribution values, so use
repository variables rather than secrets. They come from
`platform-identifiers.json`, so this block is correct in a clone the moment
that file is, with no editing here:

```bash
source scripts/platform-env.sh
: "${CI_CLIENT_ID:?Set CI_CLIENT_ID to the application ID issued by the identity owner}"
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
VOLUME=$(python3 -c \
  'import json;print(json.load(open("platform-identifiers.json"))["sdk_artifact_volume"])')

# Identity of the CI service principal (from your platform identity owner).
gh variable set AZURE_CLIENT_ID       -R "$REPO" -b "$CI_CLIENT_ID"

gh variable set AZURE_TENANT_ID       -R "$REPO" -b "$AZURE_TENANT_ID"
gh variable set AZURE_SUBSCRIPTION_ID -R "$REPO" -b "$AZURE_SUBSCRIPTION_ID"
gh variable set DATABRICKS_HOST       -R "$REPO" -b "$DATABRICKS_HOST"
gh variable set SDK_ARTIFACT_VOLUME   -R "$REPO" -b "$VOLUME"

gh variable list -R "$REPO"
```

Ownership, team, cost-center, and approved compute-policy values are rendered
into the generated project and cross-checked before deployment. They are not
mutable GitHub or Databricks bundle variables; change them through a reviewed
project/template update so the manifest, runtime context, and resource tags
move together.

`AZURE_CLIENT_ID` is the one value not in the fixture: it identifies an
externally provisioned Entra application, and a clone is issued a different one
(see `docs/enterprise-clone-runbook.md`).

Do not add a `gh secret set` step.

## 4. Verify end-to-end

The authentication smoke test proves the OIDC exchange. The deploy workflow
proves the principal also has the required Databricks authorization:

```bash
gh workflow run auth-smoke.yml --ref main
gh workflow run deploy.yml --ref main
gh run watch
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
  SDK_ARTIFACT_VOLUME; do
  gh variable delete "$v"
done
```

The SDK volume and its grants are also external platform resources and must be
revoked through the approved platform workflow.

## 6. UAT promotion

The supported delivery path now ends at UAT; production remains deliberately
absent. UAT uses the existing protected-`main` branch-ref OIDC subject, lifecycle
`validation`, a manual dispatch, and the same immutable artifact that passed the
dev gate. Complete the external workspace registration and least-privilege
authorization before enabling it. The exact checklist and dispatch command are in
[`docs/uat-promotion.md`](uat-promotion.md).

Credentialed jobs intentionally have no GitHub `environment:`. Add such a gate
only after the identity owner provisions and verifies its different
environment-subject federated credential; do not copy the UAT job and rename it
ad hoc.

## 7. Security boundary

Anyone who can land a commit on `main` or dispatch a credentialed workflow can
trigger a deployment. Protect `main`, require code-owner review, restrict
workflow dispatch, and keep all GitHub Actions pinned to full commit SHAs.
