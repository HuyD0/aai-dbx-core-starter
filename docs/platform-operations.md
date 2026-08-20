# Platform operations

## Bootstrap the artifact volume

Create the following Unity Catalog objects through the platform's approved
administrative workflow:

The catalog, schema, and volume names are the three dotted components of
`sdk_artifact_volume` in `platform-identifiers.json`:

```sql
CREATE CATALOG IF NOT EXISTS <catalog>;
CREATE SCHEMA IF NOT EXISTS <catalog>.<schema>;
CREATE VOLUME IF NOT EXISTS <catalog>.<schema>.<volume>;
```

Grant development groups `READ VOLUME`. Grant the dedicated release service
principal `READ VOLUME` and `WRITE VOLUME` only on that volume. Do not grant
Azure ARM RBAC or workspace admin to publish a wheel.

Set the non-secret GitHub repository variable:

```bash
gh variable set SDK_ARTIFACT_VOLUME -b "$(python3 -c \
  'import json;print(json.load(open("platform-identifiers.json"))["sdk_artifact_volume"])')"
```

Run `publish-sdk` from `main` with the exact `pyproject.toml` version. Existing
versions cannot be overwritten.

## SDK wheel lifecycle

There are three related but distinct wheel paths:

1. **SDK source development.** A developer working in this repository uses
   `pip install -e '.[dev]'` or a provider-specific editable install. The
   checkout is the source of truth, so downloading the released wheel would
   make local SDK edits invisible.
2. **SDK release.** The `publish-sdk` workflow builds
   `aai_core-<version>-py3-none-any.whl` before cloud login, creates a SHA-256
   checksum, and publishes both files immutably under
   `<sdk_artifact_volume>/aai_core/<version>/`.
3. **Application consumption.** A generated application's
   `scripts/install_core.py` downloads that exact SDK wheel and checksum through
   Databricks unified authentication, verifies the checksum, and installs it
   locally. The generated Databricks job installs the same pinned SDK wheel
   directly from the volume.

The generated application's own code is packaged separately as an application
wheel from `src/app`. A job therefore receives two first-party artifacts: the
application wheel produced by that repository and the immutable `aai-core`
wheel stored in the platform volume.

This repository's sample bundle is intentionally different: it builds the
current SDK checkout into `dist/` and deploys that wheel to its smoke job. That
path validates a proposed unreleased SDK wheel. It does not replace the
`publish-sdk` workflow or mutate an existing released version.

### Credential-free PR boundary

Version `0.1` renders and tests generated projects inside this monorepo, where
the SDK source is already present. A separate consuming repository cannot read
the Unity Catalog volume from an untrusted pull request without cloud
credentials.

Before templates are distributed into independent repositories, provide a
credential-free, read-only SDK mirror or an approved internal runner image
containing the pinned wheel. Do not solve this by adding OIDC, Databricks login,
PATs, or package credentials to pull-request workflows.

## Provider catalog

The platform team maintains approved logical resources:

- General, reasoning, and low-cost chat models.
- Embedding profiles with dimensions and normalization.
- Evaluation judge deployments.
- Search services and indexes.

For each resource publish provider, environment, data-residency classification,
capabilities, quotas, cost ownership, SLO, and support owner.

## Cost anomaly detection

The bundle deploys a scheduled watch job (`resources/cost_anomaly_job.yml`)
that evaluates the previous day's observed spend in `system.billing.usage`,
list-priced via `system.billing.list_prices`, against a per-series robust
baseline (median plus a MAD multiple over a trailing 28-day window). Series
are the account total, each workspace, each `billing_origin_product`, and
each `custom_tags['project']` value — spend with no `project` tag lands in an
explicit `untagged` bucket, which is itself a governance signal. A series
never seen before is flagged when its first day exceeds the new-spend floor.
Amounts are list prices, not negotiated rates; the bias is identical across
baseline and observation, so detection stays relative, and the delta floors
are list-price currency units.

- **Exit codes are the alert.** `0` pass, `2` anomaly detected, `1`
  configuration or runtime error — including an evaluation day with no
  billing rows, because unknown cost is not zero. Both non-zero codes fail
  the run, and `email_notifications.on_failure` mails the `COST_ALERT_EMAIL`
  group alias (never an individual), so a broken or blind monitor alerts
  exactly like an anomaly.
- **One live schedule.** `system.billing` is account-wide from any workspace.
  The schedule's pause state is the bundle variable
  `cost_anomaly_pause_status` (default `PAUSED`); only CI's dev deployment
  sets `UNPAUSED`. Laptop deploys and the dispatch-gated UAT target never add
  a duplicate daily scan or duplicate alerts.
- **Reading `system.billing` is an external grant.** The job runs as the
  deploying CI principal, which needs `USE CATALOG` on `system`, `USE SCHEMA`
  on `system.billing`, and `SELECT` on the two tables — requested and revoked
  through the approved platform process (`docs/cloud-setup.md`). Until the
  grant lands, the daily run exits 1 and the failure email proves the alert
  path end to end.
- **Tuning is a reviewed change.** Thresholds are task `parameters` in the
  resource file (`--sensitivity`, `--min-delta`, `--lag-days`, ...), not
  runtime knobs. Run on demand with
  `databricks bundle run aai_dbx_base_template_cost_anomaly -t dev`; the
  human-readable report is in the run output, and `--json` emits one
  machine-readable document. Offline, the same CLI runs against a JSON
  fixture: `python -m aai_core.billing.cli detect --input rows.json`.
- **Complements, not replacements.** Account-console Budgets (static monthly
  thresholds with email alerts) are an external, admin-managed complement.
  Materialized aggregates, serving views, and a console cost page follow the
  governed-telemetry architecture in `docs/ai-platform-hub.md` and remain
  future work; the console never reads raw system tables.

## Operational controls

- Keyless identities and least-privilege grants.
- Private endpoints and egress policy.
- Compute and serverless usage policies.
- Governed tag definitions and allowed values.
- Provider quota/rate-limit dashboards.
- Trace ingestion, quality, latency, failure, and spend dashboards.
- Governed automatic-evaluation scorers, sampling rates, filters, and judge
  cost budgets for development and production traces.
- Feedback and evaluation-dataset retention, privacy, and access policies.
- SDK compatibility and deprecation policy.
- Incident, rollback, and provider-outage procedures.

## Release readiness

Promote `aai-core` from `0.x` to `1.0` after a pilot team can generate a
project, authenticate keylessly, run an experiment, inspect a trace, execute
the evaluation gate, deploy to dev, and diagnose common failures using only
the documented paved road.
