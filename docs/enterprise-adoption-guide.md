# Enterprise adoption guide

Use this guide to bring the repository into an enterprise GitHub, Azure, and
Databricks environment. The detailed, command-level source of truth remains
the [enterprise clone runbook](enterprise-clone-runbook.md); use the
[cloud setup guide](cloud-setup.md) for identity verification, repository
variables, and revocation.

## Recommended approach

Create an independent repository in the enterprise organization, connect it to
externally provisioned platform resources, prove the complete path in a
non-production workspace, and only then add staging and production targets.
Do not provision Azure, Entra, Databricks, or GitHub infrastructure from this
repository.

## 1. Create an enterprise clone

Prefer a clone over a cross-organization fork. Record the new repository's
immutable GitHub organization and repository IDs:

```bash
gh api users/<enterprise-org> --jq .id
gh api repos/<enterprise-org>/<repo> --jq .id
```

Configure the source repository as a read-only upstream so enterprise-specific
configuration cannot be pushed back accidentally:

```bash
git remote add upstream <upstream-repo-url>
git remote set-url --push upstream DISABLED
git config merge.keepours.driver true
git config merge.keepours.name "always keep this clone's value"
```

Consume reviewed upstream release tags rather than tracking upstream `main`:

```bash
git fetch upstream --tags
git merge <reviewed-release-tag>
make sync-templates
make verify
```

## 2. Request external platform prerequisites

Ask the enterprise identity and platform owners to provide:

- a repository-specific Microsoft Entra application and service principal;
- a `main` branch federated credential using the clone's immutable GitHub IDs;
- registration of the principal in the target Databricks workspace;
- `CAN_USE` on an approved, constrained job-compute policy;
- the required Unity Catalog catalog and schema access; and
- least-privilege `READ VOLUME` and `WRITE VOLUME` access to the SDK artifact
  volume.

The identity should have no client secret, no workspace-administrator role, no
unrestricted cluster-creation capability, and no Azure ARM role unless a
separately reviewed workload requires one. Provision, change, and revoke these
resources through approved external platform processes.

The intended keyless authentication chain is:

```text
GitHub Actions OIDC
  -> enterprise Entra federated credential
  -> Azure CLI authentication
  -> Databricks unified authentication
```

Do not add client secrets, personal access tokens, storage keys, or Databricks
tokens to the repository.

## 3. Configure the clone

Update `platform-identifiers.json`, the single source for environment-specific
platform identifiers. Set the enterprise tenant, subscription, Databricks host,
compute policy, SDK artifact volume, template repository, and SDK package
source. Prefer an internal package index for credential-free generated-project
CI when one is available.

Regenerate derived template defaults rather than editing them by hand:

```bash
make sync-templates
uv run pytest -q tests/test_smoke.py
```

Configure these as GitHub **repository variables**, not secrets:

- `AZURE_CLIENT_ID`
- `AZURE_TENANT_ID`
- `AZURE_SUBSCRIPTION_ID`
- `DATABRICKS_HOST`
- `SDK_ARTIFACT_VOLUME`

The team, non-personal owner group, cost center, and constrained compute policy
are rendered, non-overridable project contract values. Update them through a
reviewed template/project change rather than repository variables. Keep prompts,
user content, and secret material out of tags and variables.

## 4. Integrate enterprise model access

Route enterprise LLM access through Azure API Management or Databricks AI
Gateway. Keep deployment and endpoint names in environment configuration while
application code uses logical resource names. If a provider requires a value
that cannot be replaced by identity, store only an approved secret reference in
configuration, never the raw value.

## 5. Protect the deployment boundary

Before enabling credentialed workflows:

1. Protect `main` and require pull requests and code-owner review.
2. Block direct and force pushes, including for administrators.
3. Restrict manual dispatch of credentialed workflows.
4. Keep pull-request workflows credential-free.
5. Keep every GitHub Action pinned to a full commit SHA.

Anyone able to land a commit on `main` or dispatch a credentialed workflow can
exercise the short-lived deployment identity, so repository protection is part
of the security boundary.

## 6. Validate in a development workspace

Run the checks in layers:

```bash
./scripts/cloud-verify.sh
gh workflow run auth-smoke.yml --ref main
gh run watch
gh workflow run deploy.yml --ref main
gh run watch
```

The authentication smoke test proves the GitHub-to-Entra OIDC exchange. The
deployment proves the identity also has the required Databricks authorization.
Do not broaden permissions to work around an authentication or authorization
failure; correct the federated subject or the specific platform grant instead.

Pilot one template after the platform path is green. Confirm its offline tests,
cost attribution, compute policy, artifact installation, model gateway, and
deployment behavior before onboarding additional teams.

## 7. Add higher environments deliberately

For each staging or production target:

1. Provision the matching federated credential externally.
2. Register the principal in the target Databricks workspace with least
   privilege.
3. Add the protected GitHub environment and its non-secret variables.
4. Add the Databricks bundle target and deployment job only after the identity
   exists.

Adding a GitHub `environment:` changes the OIDC subject. Never reuse the
branch-ref credential without first arranging a matching environment
credential with the enterprise identity owner.

## Optional platform console

Leave the Databricks Apps platform console disabled during initial adoption.
Enable it only when Databricks Apps is available, the platform owner has created
the app out of band, the CI principal has the narrow management grant, and a
serverless usage policy is available for cost attribution. The console bills
continuously while running and is stopped by default.

## Operational follow-up

- Define owners for identity, Databricks platform controls, SDK publication,
  model gateways, cost review, and incident response.
- Rehearse access revocation through the same external process that provisioned
  the identity and platform grants.
- Merge upstream release tags, not arbitrary upstream commits.
- Run `make verify` after each upstream merge and confirm the clone's identifiers
  remain intact.
- Follow [platform operations](platform-operations.md) for SDK artifact and
  release controls.
