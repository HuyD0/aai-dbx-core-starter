# Platform operations

## Bootstrap the artifact volume

Create the following Unity Catalog objects through the platform's approved
administrative workflow:

```sql
CREATE CATALOG IF NOT EXISTS platform;
CREATE SCHEMA IF NOT EXISTS platform.artifacts;
CREATE VOLUME IF NOT EXISTS platform.artifacts.python_packages;
```

Grant development groups `READ VOLUME`. Grant the dedicated release service
principal `READ VOLUME` and `WRITE VOLUME` only on
`platform.artifacts.python_packages`. Do not grant Azure ARM RBAC or workspace
admin to publish a wheel.

Set the non-secret GitHub repository variable:

```text
SDK_ARTIFACT_VOLUME=/Volumes/platform/artifacts/python_packages
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
   `/Volumes/platform/artifacts/python_packages/aai_core/<version>/`.
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
path validates an unreleased SDK candidate. It does not replace the
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
