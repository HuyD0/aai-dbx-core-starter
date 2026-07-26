# AGENTS.md — operating guide for AI coding agents

This file is the shared source of truth for Claude and Codex (`CLAUDE.md`
imports it). Read it fully before acting.

## 1. What this repository is

This is the monorepo for the AAI AI/ML developer platform:

- `aai-core`, an installable Python SDK.
- Databricks project templates that import the SDK.
- Guided examples and lifecycle documentation.
- A keyless CI/CD foundation for Azure Databricks.

The SDK helps development teams use Azure Databricks, Unity Catalog, MLflow 3,
Microsoft Foundry, Azure AI Search, Databricks AI Search, experiments, prompts,
tracing, evaluation, RAG, and agent lifecycle practices consistently.

The contract is:

```text
templates teach the lifecycle
  -> the platform console guides developers into them
  -> aai-core supplies stable runtime contracts
  -> native provider clients remain available
  -> platform infrastructure enforces identity, policy, governance, and cost
```

Version 1 focuses on platform foundations and GenAI/RAG. Classical ML and
feature-engineering templates are future work.

## 2. Authentication chain

```text
GitHub Actions (permissions: id-token: write)
  -> OIDC token (immutable subject:
     repo:HuyD0@151226205/aai-dbx-core-starter@1311037530:ref:refs/heads/main)
  -> azure/login exchanges it against an Entra federated credential
  -> Azure CLI authenticated as the CI service principal
  -> Databricks CLI uses DATABRICKS_AUTH_TYPE=azure-cli
  -> bundle deploy or Unity Catalog volume wheel publication
```

There are no stored credentials in this chain.

## 3. Provisioned identifiers

| Thing | Value |
|---|---|
| GitHub repo | `HuyD0/aai-dbx-core-starter` |
| Azure tenant | `7f6a2cf9-5e4e-46ae-95d4-74016c1df1a6` |
| Azure subscription | `ea936670-dda1-4884-8467-49c225bf3e83` (`practisesubscription`) |
| Legacy CI app | `github-actions-dbx-platform` |
| Legacy client id | `b74a6820-d0ac-454f-8c32-02141cba3c8a` |
| Legacy SP object id | `f1ae1583-6b35-4d6c-a7c1-305034983307` |
| Dedicated CI app | `github-actions-aai-dbx-core-starter` |
| Dedicated client id | `a7e40167-d3f6-48a9-acd9-7998230cce34` |
| Dedicated SP object id | `4539bb3b-b4ff-4f63-9da5-5873ececace6` |
| Federated credential | `gh-aai-dbx-core-starter-main` |
| FIC subject | `repo:HuyD0@151226205/aai-dbx-core-starter@1311037530:ref:refs/heads/main` |
| Dev workspace | `dbx-dev` / `https://adb-7405609799238491.11.azuredatabricks.net` / `7405609799238491` |
| SDK artifact path | `/Volumes/platform/artifacts/python_packages/aai_core/<version>/` |

These are non-secret identifiers. Do not classify them as secrets.

`platform-identifiers.json` at the repo root is the machine-readable copy of
the environment-specific values above; tests and `scripts/cloud-verify.sh`
cross-check every other occurrence against it. When cloning this repo into a
different tenant/workspace, edit that file first and follow
`docs/enterprise-clone-runbook.md` — the smoke tests then point at each
remaining file that must agree (this table included).

## 4. Hard security rules

1. **No secrets in Git.** Never add a client secret, PAT, storage key, API key,
   or raw Key Vault value. `aai-platform.yml` contains identifiers and secret
   references only.
2. **Prefer identity over secrets.** Use managed/workload identity, GitHub OIDC,
   Databricks unified authentication, and Unity Catalog service credentials.
   Key Vault is for credentials that cannot be replaced by identity.
3. **Pull requests remain credential-free.** Never add `azure/login`,
   `id-token: write`, a real credential, or cloud integration calls to the
   `pull_request` workflow.
4. **Credentialed jobs have no GitHub `environment:`.** The FIC uses a
   branch-ref subject. Have the platform identity owner add a matching
   environment FIC before introducing an environment gate.
5. **Least privilege.** The dedicated CI principal has no ARM RBAC, is
   registered only in `dbx-dev`, is not workspace admin, and uses constrained
   compute. Wheel publication adds only `READ VOLUME` and `WRITE VOLUME` on the
   SDK artifact volume.
6. **Never grant broad rights to fix authentication.** Solve failures with the
   correct Databricks object permission, Unity Catalog privilege, compute
   policy, or FIC.
7. **The legacy app is shared.** Do not delete, rotate, or mutate
   `github-actions-dbx-platform` or its other credentials/assignments.
8. **Bootstrap is external and human-run.** Identity changes, Databricks
   principal registration, catalogs, volumes, permissions, model endpoints,
   search services, and indexes are provisioned through approved platform
   processes, not this repository, CI, or application code.
9. **`main` protection is the security boundary.** Require PR and code-owner
   review, block direct/force push, and enforce protection on administrators.
10. **Pin every GitHub Action to a full commit SHA.** Credentialed workflows
    run third-party code next to a live short-lived identity.
11. **Tags contain no sensitive data.** Use a non-personal `owner_group`, not
    an individual email. Never place prompts, user content, or secret material
    in tags.
12. **Releases are immutable.** Never overwrite an existing SDK version in the
    Unity Catalog volume.

## 5. SDK design rules

- Keep the public entry point small: `bootstrap()`, `PlatformSettings`,
  `PlatformContext`, and domain modules.
- Use strict Pydantic v2 models (`extra="forbid"`, frozen where evidence must
  be immutable) at configuration, untrusted-input, persisted-evidence, tool,
  and structured-output boundaries. Use small `StrEnum` vocabularies for
  platform-owned policy choices. Do not add Pydantic models for transient
  internal state or mirror native provider response objects.
- Use logical resource names in application code; endpoint/deployment/index
  names belong in environment configuration.
- Provider abstractions cover capabilities, not administration.
- Expose `native_client` for provider-specific functionality.
- Keep SDK terms close to native APIs. Experiments use
  `baseline -> change -> result -> decision`; decisions are `adopt`, `reject`,
  or `inconclusive`. `candidate` is deprecated platform terminology, not a
  lifecycle stage or new SDK object.
- Fail capability and configuration checks before making an expensive request.
- Do not create a proprietary authentication protocol or token cache.
- `SecretValue` must never reveal its value through `str`, `repr`, logs,
  exceptions, traces, tags, or MLflow parameters.
- Use one `ResourceContext` and project it to MLflow, traces, Databricks, Azure,
  and structured logs.
- Do not permit applications to override controlled ownership/cost fields.
- Keep MLflow classic evaluation and `mlflow.genai.evaluate()` concepts
  separate.
- RAG retriever spans must emit MLflow document fields `page_content`,
  `doc_uri`, `chunk_id`, metadata, and optional id.
- Treat code, model, prompt, tool, index, embedding, and chunking changes as
  application releases.
- Pin supported dependency ranges in `pyproject.toml` and resolve exact
  versions in `uv.lock`.

## 6. Template rules

- Generated production logic is packaged Python under `src/`; notebooks are
  for exploration and teaching.
- Every template imports a pinned `aai-core` version.
- Every template contains unit tests, evaluation data, an evaluation gate,
  bundle resources, cost tags, and keyless setup instructions.
- Agent templates use MLflow Agent Server on Databricks Apps as the primary
  HTTP serving path. Models-from-code Model Serving is a compatibility path.
  LangGraph stays an optional, native application recipe with durable state,
  interrupts before side effects, and idempotency.
- Every job cluster carries `application`, `project`, `environment`, `team`,
  `owner_group`, `cost_center`, `data_classification`, `lifecycle`, and
  `tag_schema_version`.
- Bundle presets apply the same tags to supported jobs and pipelines.
- Templates may select providers by configuration without generating
  incompatible application architectures.
- Render each template in credential-free CI and run the generated unit tests.

## 7. Platform console (Databricks App) rules

The guided console lives at `src/platform_app` and is deployed as a Databricks
App. It renders the lifecycle for a developer, generates the exact commands they
run on their own machine, and reports platform state. Rules:

- **Platform-level only. Never templated.** Templates render through Go
  `text/template`, whose `{{ }}` delimiters collide with the console's Jinja
  syntax, and `templates/_shared/versions.json` would have to pin a web stack
  into every generated project. Generated projects link to the console's URL.
- **The console never verifies a developer's own access.** On-behalf-of-user
  authorization is not used: consent is irrevocable, and its scopes do not cover
  compute policies, Unity Catalog volumes or catalog grants. Every check runs as
  the app service principal and is labelled platform state. This is enforced by
  `assert_platform_state`, not by convention.
- **An app cannot carry the nine platform tags.** `resources.App` is
  `additionalProperties: false` and has no `tags` field, and the CLI applies no
  presets to apps (`// Apps: No presets`). Cost attributes through
  `usage_policy_id`, which section 9 already prescribes for serverless. Do not
  attempt to add tags to an app resource — it is a hard schema rejection.
- **The app identity is provisioned externally.** Creating an app auto-provisions
  a service principal, and section 4 rule 8 reserves principal registration for
  the human-run platform process. CI only ever *updates* an existing app; the
  platform owner runs `databricks apps create` and the app SP is recorded in
  section 3.
- **The app resource stays out of `resources/*.yml`.** `databricks.yml` globs that
  directory, so an app resource there would make every clone require Apps to be
  enabled. It lives in `resources/optional/` behind an explicit opt-in `include`.
- **Never add a workflow to deploy it.** Extend `.github/workflows/deploy.yml`.
  A new workflow file would need its own SHA-pinned action to satisfy the pinning
  test and would silently escape the credentialed-workflow tuple in
  `tests/test_smoke.py`.
- **No environment identifier may appear under `src/platform_app`.** The app's
  `source_code_path` uploads only that directory, so it cannot read
  `platform-identifiers.json`; values arrive as bundle-supplied environment
  configuration. A test enforces this because it is what keeps a clone portable.
- **The runtime environment holds a live OAuth secret.** `DATABRICKS_CLIENT_SECRET`
  is injected into the app process and app logs are readable by anyone with
  `CAN MANAGE`. Run with `--no-access-log`, never echo the environment, never
  serialise an SDK object wholesale (`dataclasses.asdict()` reaches
  `PlatformSettings.raw`), and scrub credential values out of any provider error
  before rendering it.
- **An exception handler alone does not keep a message out of the log.** Starlette's
  `ServerErrorMiddleware` sends the handler's response and then deliberately
  re-raises so the server can log it, at which point uvicorn prints the traceback
  *and the exception message*. It is the outermost layer Starlette builds, so
  `add_middleware` cannot get outside it — the app object is wrapped instead
  (`ContainExceptions`). Keep the test client strict: an earlier version swallowed
  that re-raise and hid the leak entirely.
- **The console is stopped by default.** A running app bills continuously with no
  scale-to-zero. `deploy.yml` deploys code only; `make app-start` / `app-stop`
  control the running state. A running app does not pick up newly deployed code —
  `make app-restart` does.
- **The container's `requirements.txt` pins the whole dependency closure.** It is a
  second dependency channel that `uv lock --check` never sees, and
  `pip install -r requirements.txt` runs at *deploy* time — so pinning only the
  direct requirements would let a newer transitive reach the container long after
  CI tested the locked set. The file is generated from `uv.lock` with markers
  pre-evaluated for the runtime (Ubuntu 22.04, CPython 3.11), and a test
  recomputes it rather than trusting it.
- **No node, no bundler, no `package.json`.** `scripts/cloud-verify.sh` performs an
  offline `uv sync --locked`; an npm lockfile ecosystem is a change of security
  posture, not a dependency. The console's web dependencies live in the `dev`
  extra, never in a public `app` extra that would ship a web server to SDK
  consumers.

## 8. Development workflow

Install:

```bash
python -m pip install -e '.[dev]'
```

The root `Makefile` provides the same workflow as discoverable shortcuts:

```bash
make help
make install
make hooks-install
make check
make verify
```

Before committing:

```bash
ruff check .
black --check .
pytest -q
python -m build
python scripts/validate_release.py --wheel dist
```

When changing a runtime dependency, update its supported/certified entry in
`dependency-policy.toml`, the exact `uv.lock`, regenerate every affected
template transitive lock with
`python scripts/lock_template_dependencies.py`, and update
`compatibility.json` in the same change. PRs test the certified locks; the
scheduled dependency canary must continue to pass both lower and latest
supported resolutions on Python 3.11 and 3.12.

### Codex Cloud

The repository has one supported credential-free cloud verification path:

```bash
./scripts/cloud-verify.sh
```

The Codex environment uses `scripts/codex-cloud-setup.sh` as its setup script
and `scripts/codex-cloud-maintenance.sh` as its maintenance script. They pin
Python 3.12, uv, Databricks CLI, and Azure CLI, then cache all dependencies
required by `cloud-verify.sh`. Agent-phase internet access and cloud credentials
are intentionally absent.

Codex Cloud cannot use the GitHub Actions OIDC identity. It runs offline checks,
opens a proposed change, and relies on protected `main` to hand authenticated
bundle validation and deployment to GitHub Actions. Never add a PAT, client
secret, Databricks token, or other long-lived credential to a Codex environment
to bypass this boundary.

For a Databricks bundle change:

```bash
az login
export DATABRICKS_HOST=https://adb-7405609799238491.11.azuredatabricks.net
export DATABRICKS_AUTH_TYPE=azure-cli
databricks bundle validate -t dev
```

Some sandboxes block `*.azuredatabricks.net`. A TLS failure with successful
Azure ARM access may be a data-plane network restriction; use a GitHub runner
or unrestricted shell rather than weakening identity.

## 9. Tagging and cost attribution

The canonical tag fields are documented in `docs/tagging-standard.md`. Bundle
variables provide `cost_center`, `team`, and `owner_group`; never hardcode a
real team's values into reusable template resources.

Classic compute uses cluster `custom_tags`, which propagate to Azure VM billing
and `system.billing.usage`. Serverless workloads use serverless usage policies.
Unity Catalog governed tags control supported securables. If a compute policy
fixes or forbids a tag, align the template and policy rather than dropping cost
attribution.

## 10. External infrastructure and release changes

- This repository does not own or provision cloud infrastructure. Do not add
  infrastructure-as-code tooling to its setup, CI, verification, or templates.
- Identity and platform-resource changes go through an approved external
  platform process. Do not use repository scripts or CI for imperative
  identity mutations.
- The `publish-sdk` workflow runs from `main`, verifies the requested version,
  builds the wheel, creates a checksum, and refuses to overwrite an existing
  artifact.
- The non-secret repo variable `SDK_ARTIFACT_VOLUME` points to
  `/Volumes/platform/artifacts/python_packages`.
- Generated projects download and checksum the exact pinned wheel locally and
  install the same volume path in Databricks jobs.

## 11. Reproduce, migrate, and revoke

Use:

- `docs/cloud-setup.md` for connecting to externally provisioned resources and
  requesting revocation.
- `docs/enterprise-clone-runbook.md` to stand this repository up in another
  GitHub org and Azure tenant (the identity must be re-minted — the FIC
  subject embeds immutable repo/owner ids).
- `docs/platform-operations.md` for the SDK volume and platform controls.
- `docs/archive/` for completed one-time migrations (historical record).

Never delete the shared legacy application or its UAT assignment.
