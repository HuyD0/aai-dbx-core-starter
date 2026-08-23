# Cutting a release a downstream clone can merge cleanly

Downstream enterprise clones merge this repository's **release tags**, not
`main`. Follow this checklist so each tag is a reviewed, self-consistent point
that a clone can pull with `make sync-upstream TAG=vX.Y.Z`.

## 1. Prepare the release change

Start from current `main`, choose the semantic version `X.Y.Z`, and prepare the
release on a branch. Do not push a release commit directly to protected `main`.

Align every SDK-version surface explicitly:

- Set `[project].version` in `pyproject.toml`.
- Set `sdk.version` in `compatibility.json` for the checkout under development.
- Set `sdk.generated_project_default.version`, its reviewed full commit ref and
  content digest, and every `templates.<name>.aai_core` value to the SDK that new
  projects may actually consume. A candidate is commit-pinned; do not claim a
  release tag or volume publication yet.
- Set the source-checkout fallback `__version__` in
  `src/aai_core/__init__.py`.
- Run `make sync-templates` to stamp `aai_core_version`,
  `aai_core_source_ref`, and the projected pip source into every template
  schema from the canonical metadata.

Run `uv lock` so the root package entry in `uv.lock` records the same version.

Calculate the reviewed commit's canonical SDK digest locally; this command does
not contact the remote repository:

```bash
python scripts/validate_release.py \
  --print-sdk-content-sha256 <full-candidate-commit>
```

The release validator permits the checkout version and generated-project
default to differ, but requires every generated surface to agree with the latter.

Add a matching section to `CHANGELOG.md`. The release tooling requires a
heading that starts with the exact version:

```markdown
## X.Y.Z
```

## 2. Keep template and dependency metadata consistent

If the release changes generated behavior, dependencies, runtime, gates, or
shared scaffold, bump each affected template's own semantic version in both:

- `compatibility.json` at `templates.<name>.version`; and
- `templates/<name>/template/.aai-template.json.tmpl` at `template_version`.

If a runtime dependency changed, update its supported and certified entry in
`dependency-policy.toml`, update the exact `uv.lock`, regenerate every affected
template transitive lock with `python scripts/lock_template_dependencies.py`,
and update `compatibility.json` in the same change. If the Databricks App's
dependency closure changed, regenerate the exact pins in
`src/platform_app/requirements.txt` from `uv.lock` for its documented runtime.

Re-stamp the shared template scaffold and run the complete credential-free
gate:

```bash
make sync-templates
make verify
python scripts/validate_release.py --wheel dist
```

Manually dispatch `dependency-canary.yml` for the frozen candidate and require
all Python 3.11/3.12 `lowest-direct` and `highest` jobs to pass. Record the run
with the release review; scheduled or earlier-version success is not sufficient.

Review all generated changes. Commit them on the release branch, open a pull
request into `main`, and merge only after the required checks and code-owner
review pass.

## 3. Tag the exact merged commit

Update the local `main` branch to the merged release commit and confirm the
working tree is clean. Then create an **annotated** tag on that exact commit:

```bash
git switch main
git pull --ff-only origin main
test -z "$(git status --porcelain)"
git tag -a vX.Y.Z -m "Release X.Y.Z"
git push origin vX.Y.Z
```

Do not create a lightweight tag, move an existing release tag, or tag the
pre-merge branch commit. The publish path verifies that `vX.Y.Z` is an annotated
tag pointing at the checked-out release commit.

## 4. Publish from `main`

Manually dispatch the `publish-sdk` workflow from `main` with the version input
`X.Y.Z` (without the `v` prefix), either in GitHub Actions or with:

```bash
gh workflow run publish-sdk.yml --ref main -f version=X.Y.Z
```

The workflow validates the tag and wheel, then publishes the wheel, checksum,
and release manifest immutably. Never overwrite an existing SDK version; use a
new patch release to correct one.

Only after the completion manifest is present in the volume should a follow-up
change advance the checkout to the next unreleased version and switch
`sdk.generated_project_default` from the candidate commit to the exact annotated
`vX.Y.Z` tag with `status: published`. Until then, UAT/runtime deployment remains
blocked even though generated-project PR CI can install the commit-pinned source.

## Do not

- Do **not** copy environment-specific identifier values outside
  `platform-identifiers.json` or restate them in Markdown. Each clone keeps its
  own fixture, and its values win on merge through the configured `keepours`
  driver.
- Do **not** hardcode a repository URL in a documented
  `databricks bundle init` command. Resolve it from
  `platform-identifiers.json` through the repository's existing helper.
- Do **not** add credentials to pull-request checks or bypass the protected
  `main` release path.
