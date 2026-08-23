# AI Platform Hub

AI Platform Hub is the platform console's Databricks App. It gives developers one guided
surface for getting started: it renders the lifecycle, generates the exact commands they
run on their own machine, and reports platform state.

It lives at `src/platform_app` and is **not** part of the published `aai-core` wheel.
The Hub architecture, capability gates, external prerequisites, cost controls, and
failure runbooks are documented in
[`docs/ai-platform-hub.md`](ai-platform-hub.md).

## What it does, and what it deliberately does not

| | |
|---|---|
| **Guide** | Renders the ladder in `docs/developer-onboarding.md` and `docs/developer-guide.md`. A test asserts every command block is verbatim from the document it cites, so the console cannot drift from the docs. |
| **Generate** | Builds the `az login` → export → `databricks bundle init` sequence for the chosen template, with this workspace's identifiers already substituted. This is the one thing `scripts/setup_dev.py` cannot do. |
| **Platform state** | Reports what the *app's own service principal* can reach: its identity, the constrained compute policy, the SDK artifact volume. |
| **Hub** | Publishes the `ai-platform/v1` schema and registration/workflow API, and renders portfolio, readiness, application detail, cost-optimization, and action-queue surfaces. Durable UAT registry state is available when the optional existing-Lakebase binding is configured; observability and remote workflow capabilities remain gated. |
| **Estimate** | A cost estimator ([`docs/cost-estimation.md`](cost-estimation.md)): composes multi-workload monthly estimates from a bundled Azure list-price snapshot, entirely stateless and clearly labelled as estimates — never observed billing. |

**The console never verifies your personal access.** On-behalf-of-user authorization is
not used, for two reasons that are not going to change soon:

1. Consent is **irrevocable** — *"After granting consent, users can't revoke it."* This
   repository's story is revocation (`docs/cloud-setup.md`).
2. The documented scopes (`sql`, `genie`, `files`, `iam.access-control:read`,
   `iam.current-user:read`) do not cover compute-policy visibility, `READ VOLUME`, or
   catalog grants — precisely the rungs onboarding cares about.

So every workspace row is labelled **platform state**. To check *your* access, run the
preflight that already does it properly, on your machine, as you:

```bash
python3.12 scripts/setup_dev.py --check-only
```

## Run it locally

```bash
make app-run
```

Serves on <http://127.0.0.1:8000>. With no workspace credentials the platform-state rows
report `skip`, which is the expected local result. Identifiers come from
`platform-identifiers.json` when running from a checkout; a hosted app has no checkout, so
it reads them from bundle-supplied environment configuration instead and a missing value
stays a loud failure.

## Cost posture: stopped by default

A running app bills continuously (`MEDIUM` ≈ 0.5 DBU/hour, 2 vCPU / 6 GB) and has no
documented scale-to-zero. Nothing in CI can enforce a ceiling, so the default state is the
only real control.

```bash
make app-start
make app-stop
```

`deploy.yml` deploys code only — it never starts the app.

**Sharp edge:** `databricks bundle deploy` does *not* restart a running app, so a running
console keeps serving the previous code while CI goes green. A *stopped* app picks up the
latest code when started. If you leave it running, use `make app-restart` after a deploy.

## Deploying it (requires external grants first)

The app resource lives at `resources/optional/platform_console.yml`, deliberately outside
the `resources/*.yml` glob in `databricks.yml`, so a clone into a tenant where Apps is
disabled is unaffected. Enable it with an explicit `include`.

Before any of that can work, the base prerequisites must be provisioned externally. Creating an app
auto-provisions a service principal, and AGENTS.md section 4 rule 8 reserves principal
registration for the human-run platform process — so **CI must never create the app**, only
update one that already exists.

### Grant request to the platform identity owner

> **Request: onboarding console app for `aai-dbx-core-starter`.**
>
> 1. **Create the app out-of-band.** Please run `databricks apps create` for an app named
>    `aai-platform-console-dev` in the `dbx-dev` workspace. Creating it through CI would
>    mint a workspace service principal from a repository pipeline, which our operating
>    contract reserves for your process. We then bind to that existing app so CI only ever
>    updates it — see "Binding" below.
> 2. **Grant the CI principal update rights on that app only.** Principal:
>    `github-actions-aai-dbx-core-starter`, client id
>    `a7e40167-d3f6-48a9-acd9-7998230cce34`, service principal object id
>    `4539bb3b-b4ff-4f63-9da5-5873ececace6`, registered in `dbx-dev` only. It needs
>    `CAN MANAGE` on this one app — not app-create rights, and not workspace admin.
> 3. **Provide a serverless usage policy id for cost attribution.** A Databricks App
>    resource has no `tags` field (the bundle schema is `additionalProperties: false` and
>    the CLI applies no presets to apps), so the nine platform tags cannot be attached the
>    way they are to a job cluster. Our tagging standard already routes serverless spend
>    through usage policies. Please create (or name an existing) usage policy carrying the
>    standard tag set, and give us its id to record in `platform-identifiers.json`.
>
> 4. **Grant the app's own service principal exactly two read privileges.** The app
>    auto-provisions its own principal, distinct from the CI one, and the console's
>    platform-state panel reports what *it* can reach. It needs:
>    - `READ VOLUME` on the SDK artifact volume (`sdk_artifact_volume` in
>      `platform-identifiers.json`), so it can report whether the platform can
>      read the published SDK wheel; and
>    - `CAN_USE` on the constrained job compute policy
>      (`job_compute_policy_id`), because the panel calls
>      `cluster_policies.get()` and without it that row reports failure even when
>      everything else is correctly provisioned.
>
>    Nothing further is required for the original guide and platform-state checks. Hub
>    registry, serving-view, and Jobs grants are separate opt-ins; request them only
>    through the ordered enablement checklist in `docs/ai-platform-hub.md`.

### Optional durable Hub state on the existing Lakebase resource

The App resource uses the current Lakebase Autoscaling `postgres` binding. It does not
create or select a project, branch, database, endpoint, or identity. Before opting the
resource in, the platform owner must provide these non-secret values:

- `hub_lakebase_branch`: the full
  `projects/<project>/branches/<branch>` resource path;
- `hub_lakebase_database`: the full
  `projects/<project>/branches/<branch>/databases/<database>` resource path; and
- `hub_lakebase_schema`: a dedicated lowercase PostgreSQL schema name that does not
  already belong to a developer or another application.

Set `hub_state_mode` to `lakebase`. The resource grants `CAN_CONNECT_AND_CREATE`, and the
Apps runtime injects the endpoint/host/database/user coordinates. The process uses the
app service principal to generate OAuth database credentials; no database password or
connection URL is configured.

Deploy the App before connecting to that schema locally. On first start, the app service
principal creates and owns the schema, then applies forward-only transactional migrations
under an advisory lock. If a developer creates it first, the app cannot take ownership
and intentionally fails closed. Do not drop or reassign an existing schema without a
reviewed backup and explicit approval: `DROP SCHEMA ... CASCADE` destroys Hub state.

The exact branch and database paths are intentionally required and have no repository
default. Confirm them with the platform owner; never silently assume a branch named
`production`. The binding is:

```yaml
- name: postgres
  postgres:
    branch: ${var.hub_lakebase_branch}
    database: ${var.hub_lakebase_database}
    permission: CAN_CONNECT_AND_CREATE
```

Startup fails closed if the binding variables are absent, TLS is not required, the
endpoint path is not an Autoscaling endpoint path, the app does not own its schema, or
migration history is inconsistent. Keep `hub_job_mode=unavailable` until durable job
reconciliation is separately approved.

### Binding — required, and easy to get wrong

`databricks bundle deploy` does **not** adopt an existing workspace app just because the
name matches. Without a binding in the deployment state it plans a *new* app, and then
either fails on the duplicate name or attempts the creation that rule 8 forbids. The
one-time bind, after the platform owner has created the app:

```bash
databricks bundle deployment bind platform_console aai-platform-console-dev --auto-approve
```

The CLI's own help is explicit that this is what causes the existing workspace resource to
be updated on the next deployment. `databricks bundle generate app --existing-app-name
<name> --key platform_console --bind` does the generate-and-bind in one step instead.

**The bind must be recorded in the state CI uses.** With `mode: development`, bundle
deployment state lives under the *deploying principal's* home, so a bind performed from a
developer's laptop is invisible to the CI service principal and the first CI deploy would
still plan a new app. The dedicated principal exists only inside the OIDC job, so the bind
has to happen there — `deploy.yml` takes a `bind_app` dispatch input for exactly this, and
runs the bind immediately before `bundle deploy` in the same job:

```bash
gh workflow run deploy.yml --ref main -f bind_app=aai-platform-console-dev
```

Leave `bind_app` empty for every normal deploy; it is a one-time step.

### Ordering, and the one rough edge

`bundle deployment bind` needs the resource key to exist in the bundle configuration, so
the include must be merged *before* the bind can run. But merging also triggers a
push deploy. So expect this sequence:

1. Platform owner creates the app and grants the privileges above.
2. Set `app_usage_policy_id` and merge the `include` line.
3. **That push-triggered deploy will fail** — the app is not yet bound, so the bundle
   plans a new one. This is expected and recoverable; nothing is destroyed.
4. Dispatch the workflow with `bind_app` set. It binds and then deploys in one run.
5. Subsequent pushes deploy normally.

If a red run on `main` between steps 3 and 4 is unacceptable, do the merge and the
dispatch in the same maintenance window, or temporarily pause the push trigger. There is
no way to bind before the configuration exists, so this ordering is inherent rather than
an oversight — it is called out here so it is not discovered during the deploy.

Also confirm two workspace settings, either of which silently breaks deployment:

- **Apps enabled** for the workspace.
- **"Only allow app deployments from Git"** — if on, workspace-folder
  `source_code_path` deploys fail and the resource needs `git_source` instead.

## Adding another onboarding track

The shell remains generic; onboarding content arrives through `aai_console.registry`.
A `TrackSource` is anything with an `id` and a `tracks()` method. Operational Hub data
does not use this content registry: it goes through the versioned Hub API, repository,
and governed serving adapters described in `docs/ai-platform-hub.md`.
