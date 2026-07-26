# SDK versioning and deprecation policy

`aai-core` is the platform's stable contract layer — templates pin it, and
fleet-wide upgrades should usually be a one-line version bump in consuming
projects. That only works if versioning is predictable.

## What is public

The public surface is:

- The top-level exports in `aai_core.__all__` (snapshotted by
  `tests/test_public_api.py` — changing it is a deliberate act).
- Documented classes/functions of the domain modules (`providers`, `secrets`,
  `tags`, `tracing`, `experiments`, `prompts`, `evaluation`, `rag`, `agents`,
  `deployment`, `testing`) that do not start with an underscore.
- The `aai-platform.yml` configuration schema and the `aai-core` CLI.

Underscore-prefixed names, module internals, and anything imported from a
provider SDK are not covered.

## Semver, pre-1.0

Releases follow semver. While the SDK is `0.x`:

- **Minor** (`0.X` → `0.X+1`) may contain breaking changes; they are called
  out in `CHANGELOG.md` with a migration note.
- **Patch** releases never break the public surface.

Starting with `0.3.0`, fleet-facing APIs should follow the post-1.0 discipline
even though semver technically permits a breaking minor: keep the old and new
surface working for at least two minor releases and 90 calendar days. Emit
`DeprecationWarning` with `stacklevel=2`, name the planned removal version, and
document the migration in the changelog. An urgent pre-1.0 break must have an
explicit migration note and platform-owner approval.

From `1.0.0`, removal is major-only. The two-minor/90-day minimum also applies
to configuration fields and supported Python-version removal. Stable error
codes are never reused for a different condition.

## Error contract

Every SDK-raised error derives from
`aai_core.exceptions.AaiCoreError` and carries a stable `code`. Codes are part
of the public surface: they may be added, never renamed within a major.

## Dependencies

`dependency-policy.toml` distinguishes three claims:

- **minimum** — the lower bound of every declared supported range;
- **certified** — the exact graph in `uv.lock` and the exact universal
  transitive runtime locks generated into templates;
- **latest-compatible** — the newest versions within the declared ranges,
  tested on a scheduled canary.

A supported range is real only when its minimum and latest-compatible lanes
pass. Pre-1.0 or preview provider packages are capped at the next minor until a
wider range has passed those lanes. Templates deploy generated
`universal-transitive-v1` locks: every public runtime dependency is pinned
across Python 3.11/3.12, including environment markers. Regenerate them with
`python scripts/lock_template_dependencies.py` whenever a direct dependency or
certified version changes. They intentionally do not enable pip's global hash
mode because Databricks Apps installs the private checksum-verified
`aai-core` wheel and the local application in the same requirements
transaction; exact versions prevent resolver drift, while the private SDK
artifact retains manifest and SHA-256 verification.

The `requires-python` window (currently `>=3.11,<3.13`) is widened deliberately
when Databricks Runtime adopts a new Python. Treat that as a minor release with
base-wheel, provider-extra, and generated-template coverage for the new
version.

## Feature maturity

`compatibility.json` is the machine-readable support statement. Stable
features follow the SDK compatibility policy. Preview features use certified
dependency pins, capability checks, and actionable failures; they do not enter
the stable top-level API by implication. Native-framework features such as
LangGraph graphs and state remain application-owned even when MLflow
autologging supports them. The dependency canary also exercises native OpenAI
sync, async, complete streaming, and cancellation paths; task-local trace
context; Agent Server schemas; async LangGraph persistence; and strict Pydantic
contracts. Dependency ranges are widened only after those behaviors pass on
Python 3.11 and 3.12 at both supported bounds.

## Template versions

Each template follows semver independently. Changes to generated behavior,
dependencies, runtime, gates, or shared scaffold require a template version
bump. `compatibility.json` binds every template version to its required
`aai-core` version. Credential-free CI renders the template, installs the
candidate SDK wheel rather than importing `src/`, and runs the generated
project against that wheel.

## Releases

Wheels are published immutably to the Unity Catalog volume by the
`publish-sdk` workflow; a version is never overwritten. The workflow builds
once from the locked build environment, validates that exact wheel, and
publishes its checksum and `release-manifest.json`. The manifest contains the
source commit, wheel digest, compatibility-policy digests, and template
versions, and is written last as the release-completion marker. A matching
partial upload may be resumed; different bytes are never overwritten.
Consuming projects upgrade by changing their pinned version and re-running
their release gate.

Each published version MUST also be git-tagged `v<version>` on the release
commit: generated projects' credential-free CI installs aai-core from that
tag (their `aai_core_pip_source` default), so tag and volume wheel must
describe the same code. Never move a release tag.

A release is accepted only when:

1. the versioned changelog, compatibility manifest, template versions/defaults,
   and dependency pins agree;
2. Python 3.11 and 3.12 base/provider lanes pass on the built wheel;
3. every template renders and passes its offline gate using that wheel;
4. the protected annotated tag points at the release commit; and
5. the bounded keyless dev-workspace validation succeeds.

Rollback means repinning the previous immutable release or publishing a new
patch; it never means replacing release bytes or moving a tag.
