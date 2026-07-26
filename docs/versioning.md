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

From `1.0.0`: breaking changes only in majors, with a deprecation period of at
least one minor release during which the old surface emits
`DeprecationWarning` and both surfaces work.

## Error contract

Every SDK-raised error derives from `aai_core.AaiCoreError` and carries a
stable `code`. Codes are part of the public surface: they may be added, never
renamed within a major.

## Dependencies

`pyproject.toml` pins supported ranges; `uv.lock` resolves exact versions
(AGENTS.md §5). The `requires-python` window (currently `>=3.11,<3.13`) is
widened deliberately when Databricks Runtime adopts a new Python — treat that
as a minor release with CI coverage for the new version.

## Releases

Wheels are published immutably to the Unity Catalog volume by the
`publish-sdk` workflow; a version is never overwritten. Consuming projects
upgrade by changing their pinned version and re-running their release gate.
