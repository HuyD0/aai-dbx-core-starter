# UAT promotion

The supported delivery path is `dev -> UAT`. UAT carries lifecycle
`validation`; this repository intentionally declares no production target.

The workflow never provisions identity or infrastructure. Before setting
`UAT_DEPLOYMENT_ENABLED=true`, the platform and identity owners must complete
all of the following outside this repository:

1. Register this repository's dedicated CI service principal in the UAT
   workspace. Grant only the approved compute policy, artifact volume, and
   workload-specific resource permissions. Do not reuse or mutate the shared
   legacy application.
2. Verify the existing `main` branch-ref federated credential still has the
   immutable repository subject documented in `AGENTS.md`. The credentialed
   jobs deliberately have no GitHub `environment:` because no matching
   environment-subject credential has been verified.
3. Protect `main`, restrict workflow dispatch to trusted maintainers, and set
   the non-secret repository variable `DATABRICKS_UAT_HOST` to the exact
   `databricks_uat_host` value in `platform-identifiers.json`. Keep the existing
   repository-level `AZURE_CLIENT_ID` and `AZURE_TENANT_ID` values for the
   dedicated CI identity registered in both workspaces. The fixture records that workspace as
   `databricks_uat_host`, and `dbx-uat` is only its human-readable name in
   AGENTS.md section 3; the platform owner must reconfirm that mapping before
   enablement. CI's exact-match check prevents
   drift but does not claim live workspace verification.
4. Set the non-secret repository variables `HUB_LAKEBASE_BRANCH`,
   `HUB_LAKEBASE_DATABASE`, and `HUB_LAKEBASE_SCHEMA` to reviewed references
   for the existing Lakebase resource. The prerequisite gate rejects missing
   or placeholder values before requesting UAT credentials and never guesses a
   Lakebase path.
5. Externally create any optional Databricks App deployed in UAT and grant the
   CI principal `CAN MANAGE`. CI may bind and update that App; it must never
   create the App or its service principal.
6. Set the repository variable `UAT_DEPLOYMENT_ENABLED` to `true` only after
   the preceding controls have been reviewed.

Start a promotion with:

```bash
gh workflow run deploy.yml --ref main -f target=uat
```

If an opted-in console App needs its one-time bundle binding, include the
externally created name:

```bash
gh workflow run deploy.yml --ref main -f target=uat \
  -f bind_app=aai-platform-console-uat
```

One run builds and validates a single wheel before cloud login, records its
SHA-256 digest and source commit, deploys it to dev, and then applies the manual
UAT prerequisite gate. UAT re-verifies the same wheel evidence before
deployment. There is no rebuild between environments. The optional console
resource remains outside the default bundle include: these settings prepare its
existing Lakebase binding but do not enable the console or create any resource.

`UAT_DEPLOYMENT_ENABLED` is a reviewed enablement flag, not a substitute for
branch protection. Do not add a GitHub environment gate until its matching
environment-subject federated credential has been provisioned and verified
externally.

For a credentialed, read-only bundle check from an approved shell, first export
the three reviewed `HUB_LAKEBASE_*` values named above, then pass those exact
existing-resource references through the bundle variables:

```bash
source scripts/platform-env.sh
DATABRICKS_HOST="$DATABRICKS_UAT_HOST" \
BUNDLE_VAR_deployment_release="$(git rev-parse HEAD)" \
BUNDLE_VAR_hub_lakebase_branch="$HUB_LAKEBASE_BRANCH" \
BUNDLE_VAR_hub_lakebase_database="$HUB_LAKEBASE_DATABASE" \
BUNDLE_VAR_hub_lakebase_schema="$HUB_LAKEBASE_SCHEMA" \
  databricks bundle validate -t uat --strict
```

Rollback is a new promotion of a reviewed known-good commit, normally by
reverting `main` and dispatching UAT again. Do not use `bundle destroy` as a
rollback mechanism.
