# Platform console

The platform console is a Databricks App that gives a developer one guided surface for
getting started: it renders the lifecycle, generates the exact commands they run on their
own machine, and reports platform state.

It lives at `src/platform_app` and is **not** part of the published `aai-core` wheel.

## What it does, and what it deliberately does not

| | |
|---|---|
| **Guide** | Renders the ladder in `docs/developer-onboarding.md` and `docs/developer-guide.md`. A test asserts every command block is verbatim from the document it cites, so the console cannot drift from the docs. |
| **Generate** | Builds the `az login` → export → `databricks bundle init` sequence for the chosen template, with this workspace's identifiers already substituted. This is the one thing `scripts/setup_dev.py` cannot do. |
| **Platform state** | Reports what the *app's own service principal* can reach: its identity, the constrained compute policy, the SDK artifact volume. |

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

Before any of that can work, three things must be provisioned externally. Creating an app
auto-provisions a service principal, and AGENTS.md section 4 rule 8 reserves principal
registration for the human-run platform process — so **CI must never create the app**, only
update one that already exists.

### Grant request to the platform identity owner

> **Request: onboarding console app for `aai-dbx-core-starter`.**
>
> 1. **Create the app out-of-band.** Please run `databricks apps create` for an app named
>    `aai-platform-console-dev` in the `dbx-dev` workspace. Creating it through CI would
>    mint a workspace service principal from a repository pipeline, which our operating
>    contract reserves for your process. Once it exists, we bind to it with
>    `databricks bundle deploy`, which only ever updates it.
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
> The app's own auto-provisioned service principal additionally needs `READ VOLUME` on the
> SDK artifact volume so the console can report whether the platform can read the published
> wheel. It needs no other privilege — it reads no application data.

Also confirm two workspace settings, either of which silently breaks deployment:

- **Apps enabled** for the workspace.
- **"Only allow app deployments from Git"** — if on, workspace-folder
  `source_code_path` deploys fail and the resource needs `git_source` instead.

## Adding a second track set

The shell is generic; content arrives through `aai_console.registry`. A `TrackSource` is
anything with an `id` and a `tracks()` method. A lifecycle-and-cost dashboard — MLflow runs
grouped by the `application` tag, gate outcomes, and `system.billing.usage` sliced by the
nine tags — is intended to be a second registration rather than a rewrite.
