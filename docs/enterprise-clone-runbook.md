# Enterprise clone runbook

Use this checklist to connect a clone of this repository to resources managed
by another GitHub organization, Azure tenant, and Databricks workspace. The
repository does not provision those resources.

## 1. Capture the clone's immutable GitHub IDs

```bash
gh api users/<org> --jq .id
gh api repos/<org>/<repo> --jq .id
```

The federated credential subject embeds these numeric owner and repository
IDs. A clone cannot reuse the current repository's subject.

## 2. Request the external platform prerequisites

Ask the enterprise identity and platform owners for:

- a repository-specific Entra application and service principal;
- a `main` branch federated credential using the new immutable IDs;
- registration in the target Databricks workspace;
- `CAN_USE` on the approved job-compute policy;
- least-privilege access to the SDK artifact volume;
- for the cost anomaly watch: `USE CATALOG` on `system`, `USE SCHEMA` on
  `system.billing`, and `SELECT` on `system.billing.usage` and
  `system.billing.list_prices`.

The identity must have no client secret and should have no Azure ARM role
unless an independently reviewed workload requirement needs one. Do not add
infrastructure provisioning to this repository or its CI.

## 3. Update repository identifiers

Edit `platform-identifiers.json`, then run the sync. That covers every value
the build consumes — `databricks.yml`, the template schema defaults, and the
package URLs — and tests fail on any copy that disagrees:

```bash
$EDITOR platform-identifiers.json
make sync-templates
```

The keys:

| Key | Notes |
|---|---|
| `azure_tenant_id` | |
| `azure_subscription_id` | |
| `databricks_host` | |
| `databricks_uat_host` | UAT workspace used only by the protected promotion path |
| `job_compute_policy_id` | |
| `sdk_artifact_volume` | `/Volumes/<catalog>/<schema>/<volume>`; the dotted form used by app resource bindings is derived from it |
| `app_usage_policy_id` | Serverless usage policy for the optional console app; stamp the clone's policy even when the app remains disabled |
| `project` | Cost-attribution `project` tag for every job cluster and preset. The cost anomaly watch buckets spend by this tag, so a clone that leaves it attributes its own usage to this repository |
| `template_repo` | The clone's own Git URL. Left pointing upstream, the platform console generates `bundle init` commands that initialise projects from the upstream repository |
| `sdk_pip_source` | PEP 508 direct URL from which a generated project's **credential-free CI** installs `aai-core`. Left pointing upstream, every generated project's CI depends on that repository over the public internet. Prefer an immutable URL in the internal artifact service, such as `https://packages.example/aai_core-{{.aai_core_version}}-py3-none-any.whl`. The sync step projects repository tag placeholders through the reviewed source ref in `compatibility.json`. |

`make sync-templates` stamps the derived copies — `databricks.yml`'s variable
defaults and both workspace hosts, and the platform-controlled defaults in
each `templates/*/databricks_template_schema.json`.

Verify:

```bash
pytest -q tests/test_smoke.py
```

These checks fail on any value that still disagrees, on any `*.md` that
restates an identifier, and on a fixture missing a key this version requires.

`AZURE_CLIENT_ID` is deliberately *not* in the fixture: it identifies the
externally provisioned Entra application from section 2, and is set as a GitHub
repository variable in section 4.

Four things sit outside the fixture and no sync or test can set them for you.
Do them now, while the fixture edit is fresh:

1. **Record the new identity in prose.** The AGENTS.md section 3 table (app
   name, client id, service-principal object id, federated-credential name) and
   the copy-paste grant request in `docs/platform-console.md` still describe the
   upstream tenant's objects. They are identity objects, not environment
   fixtures, which is why they live in prose — and why editing them is manual.
2. **Replace `.github/CODEOWNERS`.** Section 7 covers this; it is listed here
   because it is an upstream username in a file merges will keep offering back.
3. **Create the Unity Catalog objects.** Section 2 asks for *access to* the SDK
   artifact volume; creating the catalog, schema, and volume is a separate step
   with its SQL in
   [`docs/platform-operations.md`](platform-operations.md#bootstrap-the-artifact-volume).
4. **Delete or ignore `docs/platform-audit.md`.** It is a dated historical
   snapshot, deliberately exempt from the identifier-drift tests, so it keeps
   quoting the upstream tenant's values forever. Nothing depends on it.

## 3a. Track upstream without re-resolving the same conflicts

Updates flow one way: upstream → clone. Merging the other direction would push
this tenant's subscription, workspace host, compute policy, and volume paths
into the upstream repository, so make the wrong direction fail rather than
relying on discipline:

```bash
git remote add upstream <upstream-repo-url>
git remote set-url --push upstream DISABLED
```

Prefer a GitHub *clone* over a *fork*: cross-organisation forks are unreliable
under enterprise SSO/EMU, and a fork relationship advertises a pull-request path
back upstream that must not exist.

Before using either merge path, enable automatic resolution for the two
clone-owned files that are meant to differ forever. Git will not run a merge
driver a repository defines for itself, so each clone sets this once, locally
— `.gitattributes` is already committed:

```bash
git config merge.keepours.driver true
git config merge.keepours.name "always keep this clone's value"
```

Merge rather than rebase: rebasing this clone's commits re-applies the same
identifier resolution on every sync, while a merge settles it once per release.

Sync on release tags rather than `main`, so what you merge is a reviewed, tested
point rather than whatever is mid-flight. A clone cannot use `sync-upstream`
until it has synced the release that introduces that target. Bootstrap that
first release manually, leaving the merge staged for review rather than
committing it automatically:

```bash
git fetch upstream --tags
git merge --no-commit --no-ff vX.Y.Z
make sync-templates   # no-op unless upstream changed what is stamped
make verify
```

If that merge reports conflicts, resolve them as described below before running
the remaining commands. Review `git diff --cached` and `git log`, then commit and
open a pull request into `main` only after the credential-free verification is
green.

Once a clone has synced the release that introduced the target, the whole
sequence is a single command. Start from a clean worktree:

```bash
make sync-upstream TAG=vX.Y.Z
```

It fetches tags, merges the tag with `--no-commit --no-ff`, resolves the
generated stamped files with `make resolve-upstream`, re-stamps, and runs
`make verify`. It never commits: on success, review `git diff --cached` and
`git log`, then commit and open a pull request into `main`. Any non-generated
conflict — for example, a clone that has locally modified the `Makefile` — is
left for deliberate hand-resolution, and the target stops before verification
rather than hiding it. The one-time bootstrap merge can expose the same expected
manual conflicts.

Personal-account maintainers: see `docs/upstream-release-prompt.md` for the
release-cutting checklist that keeps each tag mergeable.

### When a sync does conflict

Ordinary upstream changes — new template properties, SDK work, documentation —
merge cleanly, because the values this clone changed live in one file that the
merge driver keeps.

The exception is upstream changing *its own* identifiers. Upstream then edits
the same stamped lines this clone did, so `databricks.yml` and the generated
`databricks_template_schema.json` files conflict. That resolution is mechanical
— take upstream's content so its template changes survive, then re-stamp this
clone's identifiers over it:

```bash
make resolve-upstream
make verify
git commit
```

The target only touches generated files and leaves anything else conflicted for
you to resolve deliberately. It takes upstream's version of those files
wholesale, so if this clone customises a template schema beyond its identifier
defaults, resolve that file by hand instead.

Rehearse it before it matters — change every value in the fixture on a scratch
branch, merge an upstream tag, and confirm no conflict prompt and a green
`make verify` with your values intact.

## 3b. Configure the Codex Cloud environment

`scripts/cloud-verify.sh` is the credential-free gate. Its identity and
forbidden-credential checks run **only** when `AAI_CLOUD_ENV=codex`, and when
that variable is unset the whole block is skipped without a word — so a clone
that never sets it loses the check that proves no cloud credential reached the
agent environment.

In the Codex environment configuration set:

| Variable | Value |
|---|---|
| `AAI_CLOUD_ENV` | `codex` |
| `AZURE_TENANT_ID` | `azure_tenant_id` from the fixture |
| `AZURE_SUBSCRIPTION_ID` | `azure_subscription_id` from the fixture |
| `DATABRICKS_HOST` | `databricks_host` from the fixture |
| `AZURE_CLIENT_ID` | the clone's own client id from section 2 |

The first four must match the fixture exactly; the script compares them and
fails on any mismatch. Use `scripts/codex-cloud-setup.sh` as the environment's
setup script and `scripts/codex-cloud-maintenance.sh` as its maintenance
script. Never add a client secret, PAT, or Databricks token to that
environment: the same block fails if one is present, and that is deliberate —
Codex Cloud is offline by design and hands authenticated work to GitHub
Actions.

## 4. Configure repository variables

Follow `docs/cloud-setup.md` with the enterprise client ID, tenant, workspace,
and artifact path. Use GitHub repository variables, never secrets.

Do not skip the cost-attribution variables there — `COST_CENTER`, `TEAM`,
`OWNER_GROUP`, and `COST_ALERT_EMAIL`. `deploy.yml` falls back to placeholders
when they are unset, so the first deployment succeeds while charging this
tenant's usage to another organisation's cost center, and the cost-anomaly
watch — which CI unpauses on that same first deploy — mails its alerts to an
undeliverable placeholder. Nothing fails; the attribution is simply wrong.

## 5. Configure model access

Enterprise LLM access should flow through Azure API Management or Databricks AI
Gateway. This is application configuration in `aai-platform.yml`; application
code continues to use logical resource names.

- For Databricks AI Gateway, configure the logical model's
  `provider: databricks` deployment as the gateway-enabled serving endpoint.
- For Azure API Management, use `provider: azure_apim`, the enterprise
  `base_url`, and `token_scope`. If a subscription key is mandatory, store
  only a secret reference such as `keyvault://...` or
  `databricks-secret://...`.

## 6. Decide whether to run the platform console

The guided console (`src/platform_app`) is **optional and off by default in a
clone**. Its bundle resource lives at `resources/optional/platform_console.yml`,
deliberately outside the `resources/*.yml` glob in `databricks.yml`, so a clone
deploys nothing app-related until you opt in with an explicit `include:`.

Leave it off unless all of the following hold in the target workspace:

- Databricks Apps is enabled.
- The workspace setting *"Only allow app deployments from Git"* is **off**
  (otherwise a workspace-folder `source_code_path` deploy fails and the resource
  needs `git_source` instead).
- Your platform identity owner has created the app out-of-band and granted the
  CI principal `CAN MANAGE` on it. Creating an app auto-provisions a workspace
  service principal, which AGENTS.md section 4 rule 8 reserves for the human-run
  platform process — CI must only ever update an existing app.
- You have a serverless usage policy id for cost attribution. An app resource
  has no `tags` field, so `app_usage_policy_id` is how its spend is attributed.

The exact grant request to send is in `docs/platform-console.md`. Note the
console bills continuously while running and is stopped by default; use
`make app-start` / `make app-stop`.

## 7. Protect and verify

1. Replace the upstream entry in `.github/CODEOWNERS` with a non-personal
   enterprise team that has at least two eligible reviewers, then verify that
   GitHub recognizes the team as a code owner.
2. Protect `main`; require at least one approval, code-owner review, approval
   of the latest push, conversation resolution, and the repository's required
   quality/security checks. Enforce the rule for administrators and block
   direct, force, and deletion pushes.
3. Run `gh workflow run auth-smoke.yml --ref main`.
4. Run or merge into `deploy.yml`; a green deployment is the definitive
   authorization test.
5. Run `./scripts/cloud-verify.sh` for the credential-free local checks.

The included UAT target reuses the protected-`main` branch-ref credential and
needs the dedicated CI principal registered with least privilege in its UAT
workspace before `UAT_DEPLOYMENT_ENABLED` is set. Its credentialed job has no
GitHub `environment:` until a matching environment-subject credential is
separately provisioned and verified. See `docs/uat-promotion.md`.
