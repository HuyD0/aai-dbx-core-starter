"""Sync canonical shared scaffold files and identifier defaults into every template.

Three canonical sources feed every bundle template:

1. templates/_shared/ holds byte-for-byte scaffold copies. The Databricks template
   renderer only sees one template root, so files cannot be shared natively.
2. platform-identifiers.json holds the environment-specific values. Their schema
   defaults were previously hand-copied into every template, which meant a
   clone had to edit the same value in several places and every upstream merge
   conflicted on all of them. They are now stamped from the fixture.
3. compatibility.json owns the SDK version and immutable source ref offered by
   generated projects. This is deliberately separate from the SDK version under
   development in pyproject.toml.

A template is any templates/<dir>/ containing databricks_template_schema.json.

Usage:
    python scripts/sync_template_shared.py          # write copies and defaults
    python scripts/sync_template_shared.py --check  # CI drift check, exit 1 on drift
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATES_DIR = REPO_ROOT / "templates"
SHARED_DIR = TEMPLATES_DIR / "_shared"
SCHEMA_FILE = "databricks_template_schema.json"
IDENTIFIERS_FILE = REPO_ROOT / "platform-identifiers.json"
COMPATIBILITY_FILE = REPO_ROOT / "compatibility.json"

#: Schema property -> platform-identifiers.json key. Every one of these is an
#: environment-specific value that a clone must change, and none of them varies
#: between templates, so the fixture owns them and this script stamps them.
#: `aai_core_pip_source` is included deliberately: it is the only default that
#: reaches out to a *repository* rather than a workspace, so a clone that misses
#: it silently makes every generated project's CI depend on the upstream repo.
IDENTIFIER_DEFAULTS = {
    "workspace_host": "databricks_host",
    "uat_workspace_host": "databricks_uat_host",
    "compute_policy_id": "job_compute_policy_id",
    "aai_core_volume": "sdk_artifact_volume",
    "aai_core_pip_source": "sdk_pip_source",
}
RELEASE_DEFAULTS = frozenset({"aai_core_version", "aai_core_source_ref"})
SCHEMA_DEFAULTS = frozenset(IDENTIFIER_DEFAULTS) | RELEASE_DEFAULTS


def load_identifiers() -> dict[str, str]:
    raw = json.loads(IDENTIFIERS_FILE.read_text(encoding="utf-8"))
    missing = sorted(
        key for key in IDENTIFIER_DEFAULTS.values() if not str(raw.get(key, "")).strip()
    )
    if missing:
        raise SystemExit(
            f"{IDENTIFIERS_FILE.name} is missing required identifier(s): "
            + ", ".join(missing)
        )
    return raw


def load_generated_project_default() -> dict[str, str]:
    """Return the clone-independent SDK defaults recorded for new projects."""

    raw = json.loads(COMPATIBILITY_FILE.read_text(encoding="utf-8"))
    try:
        generated_default = raw["sdk"]["generated_project_default"]
        version = str(generated_default["version"])
        source_ref = str(generated_default["source"]["ref"])
    except (KeyError, TypeError) as error:
        raise SystemExit(
            "compatibility.json is missing sdk.generated_project_default metadata"
        ) from error
    if not version.strip() or not source_ref.strip():
        raise SystemExit(
            "compatibility.json generated-project SDK version/ref cannot be empty"
        )
    return {
        "aai_core_version": version,
        "aai_core_source_ref": source_ref,
    }


def projected_pip_source(value: str) -> str:
    """Bind a repository source to the reviewed release ref schema property.

    Enterprise clones may replace the repository source with their own direct
    URL. A Git source keeps its clone-owned repository location but always uses
    the release-metadata ref; non-Git artifact URLs remain authoritative.
    """

    if not value.startswith("git+"):
        return value
    repository, separator, _ = value.rpartition("@")
    if not separator or not repository:
        raise SystemExit("sdk_pip_source Git URL must contain an immutable ref")
    return f"{repository}@{{{{.aai_core_source_ref}}}}"


def planned_schema_defaults() -> list[tuple[Path, str, str]]:
    """(schema path, property name, expected default) across all templates."""

    identifiers = load_identifiers()
    release_defaults = load_generated_project_default()
    planned: list[tuple[Path, str, str]] = []
    for template in discover_templates():
        schema_path = template / SCHEMA_FILE
        properties = json.loads(schema_path.read_text(encoding="utf-8"))["properties"]
        for prop, identifier_key in IDENTIFIER_DEFAULTS.items():
            if prop not in properties:
                continue
            value = identifiers[identifier_key]
            if prop == "aai_core_pip_source":
                value = projected_pip_source(value)
            planned.append((schema_path, prop, value))
        for prop, value in release_defaults.items():
            if prop in properties:
                planned.append((schema_path, prop, value))
    return planned


#: Bundle variable name -> identifier key. `sdk_artifact_volume_full_name` is the
#: same volume in dotted Unity Catalog form, which is what an app resource binding
#: requires, so it is derived rather than stored twice.
BUNDLE_VARIABLE_DEFAULTS = {
    "app_usage_policy_id": "app_usage_policy_id",
    "job_compute_policy_id": "job_compute_policy_id",
    "project": "project",
    "sdk_artifact_volume": "sdk_artifact_volume",
    "template_repo": "template_repo",
}
BUNDLE_FILE = REPO_ROOT / "databricks.yml"
PROJECT_FILE = REPO_ROOT / "pyproject.toml"


def volume_full_name(volume_path: str) -> str:
    """`/Volumes/a/b/c` -> `a.b.c`, the dotted form an app resource binds to."""

    parts = [part for part in volume_path.split("/") if part]
    if len(parts) != 4 or parts[0] != "Volumes":
        raise SystemExit(
            f"sdk_artifact_volume must look like /Volumes/<catalog>/<schema>/<volume>; "
            f"got {volume_path!r}"
        )
    return ".".join(parts[1:])


def _expected_bundle_values() -> dict[str, str]:
    identifiers = load_identifiers()
    expected = {
        variable: identifiers[key] for variable, key in BUNDLE_VARIABLE_DEFAULTS.items()
    }
    expected["sdk_artifact_volume_full_name"] = volume_full_name(
        identifiers["sdk_artifact_volume"]
    )
    return expected


def _apply_bundle_identifiers(check: bool) -> list[str]:
    """Stamp (or verify) databricks.yml's identifier literals.

    Rewritten line-wise rather than through a YAML round-trip, which would strip
    the file's explanatory comments. `workspace.host` must stay a literal because
    the Databricks CLI forbids variable interpolation in authentication fields.
    """

    identifiers = load_identifiers()
    original = BUNDLE_FILE.read_text(encoding="utf-8")
    lines = original.splitlines(keepends=True)
    drift: list[str] = []

    def replace_scalar(index: int, value: str, label: str) -> None:
        line = lines[index]
        prefix, _, current = line.partition(":")
        current_value = current.strip()
        quote = '"' if current_value.startswith('"') else ""
        rendered = f"{prefix}: {quote}{value}{quote}\n"
        if line == rendered:
            return
        if check:
            # Name the setting, not just the line: this message is what a clone
            # acts on when `make sync-templates` has not been run.
            drift.append(
                f"databricks.yml:{index + 1} {label} is "
                f"{current_value} != platform-identifiers.json {quote}{value}{quote}"
            )
        else:
            lines[index] = rendered

    expected = _expected_bundle_values()
    section: str | None = None
    current_key: str | None = None
    for index, line in enumerate(lines):
        top_level = re.match(r"^([a-z_]+):\s*$", line)
        if top_level:
            section, current_key = top_level.group(1), None
            continue
        nested = re.match(r"^  ([a-z_]+):\s*$", line)
        if nested:
            current_key = nested.group(1)
            continue
        if (
            section == "variables"
            and re.match(r"^    default:", line)
            and current_key in expected
        ):
            replace_scalar(
                index, expected[current_key], f"variables.{current_key}.default"
            )
            current_key = None
        elif section == "targets" and re.match(r"^      host:", line):
            identifier_key = {
                "dev": "databricks_host",
                "uat": "databricks_uat_host",
            }.get(current_key or "")
            if identifier_key is not None:
                # Authentication fields cannot use bundle-variable interpolation,
                # so both governed workspace hosts remain literal values stamped
                # from the clone-owned identifier fixture.
                replace_scalar(
                    index,
                    identifiers[identifier_key],
                    f"targets.{current_key}.workspace.host",
                )

    if not check and "".join(lines) != original:
        BUNDLE_FILE.write_text("".join(lines), encoding="utf-8")
        print("stamped identifiers into databricks.yml")
    return drift


def bundle_identifier_drift() -> list[str]:
    """databricks.yml literals that disagree with the fixture. Writes nothing."""

    return _apply_bundle_identifiers(check=True)


def _apply_project_urls(check: bool) -> list[str]:
    """Stamp package support links from the clone-owned repository URL."""

    repository = str(load_identifiers()["template_repo"]).rstrip("/")
    if repository.endswith(".git"):
        repository = repository[:-4]
    if not repository.startswith("https://"):
        raise SystemExit("template_repo must be an HTTPS repository URL")
    expected = {
        "Documentation": f"{repository}/tree/main/docs",
        "Issues": f"{repository}/issues",
        "Repository": repository,
    }
    original = PROJECT_FILE.read_text(encoding="utf-8")
    lines = original.splitlines(keepends=True)
    section: str | None = None
    seen: set[str] = set()
    drift: list[str] = []
    for index, line in enumerate(lines):
        header = re.match(r"^\[([^]]+)]\s*$", line)
        if header:
            section = header.group(1)
            continue
        if section != "project.urls":
            continue
        assignment = re.match(r'^([A-Za-z]+)\s*=\s*"[^"]*"\s*$', line)
        if assignment is None or assignment.group(1) not in expected:
            continue
        key = assignment.group(1)
        seen.add(key)
        rendered = f'{key} = "{expected[key]}"\n'
        if line == rendered:
            continue
        if check:
            drift.append(f"pyproject.toml project URL {key} differs from template_repo")
        else:
            lines[index] = rendered
    missing = sorted(set(expected) - seen)
    if missing:
        raise SystemExit(
            "pyproject.toml [project.urls] is missing: " + ", ".join(missing)
        )
    if not check and "".join(lines) != original:
        PROJECT_FILE.write_text("".join(lines), encoding="utf-8")
        print("stamped package support URLs into pyproject.toml")
    return drift


def schema_default_drift() -> list[str]:
    """Identifier defaults that disagree with the fixture. Writes nothing."""

    return _apply_schema_defaults(check=True)


def _apply_schema_defaults(check: bool) -> list[str]:
    """Stamp (or verify) identifier defaults. Returns drift descriptions."""

    drift: list[str] = []
    by_schema: dict[Path, list[tuple[str, str]]] = {}
    for schema_path, prop, expected in planned_schema_defaults():
        by_schema.setdefault(schema_path, []).append((prop, expected))

    for schema_path, expectations in by_schema.items():
        # Round-trips at indent 2 with a trailing newline, so rewriting the whole
        # document keeps the diff to the lines that actually changed.
        document = json.loads(schema_path.read_text(encoding="utf-8"))
        changed = False
        for prop, expected in expectations:
            current = document["properties"][prop].get("default")
            if current == expected:
                continue
            if check:
                drift.append(
                    f"{schema_path.relative_to(REPO_ROOT)}: {prop} default "
                    f"{current!r} != canonical schema default {expected!r}"
                )
            else:
                document["properties"][prop]["default"] = expected
                changed = True
        if changed:
            schema_path.write_text(
                # ensure_ascii matches the committed files: unescaping \u2014
                # would rewrite unrelated description lines and reintroduce the
                # very merge noise this stamping exists to remove.
                json.dumps(document, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"stamped identifiers into {schema_path.relative_to(REPO_ROOT)}")
    return drift


def discover_templates() -> list[Path]:
    return sorted(
        entry
        for entry in TEMPLATES_DIR.iterdir()
        if entry.is_dir() and (entry / SCHEMA_FILE).is_file()
    )


def planned_copies() -> list[tuple[Path, Path]]:
    """(canonical source, destination) pairs across all discovered templates."""

    manifest = json.loads((SHARED_DIR / "manifest.json").read_text(encoding="utf-8"))
    opt_out: dict[str, list[str]] = manifest.get("opt_out", {})
    pairs: list[tuple[Path, Path]] = []
    for template in discover_templates():
        for relative in manifest["files"]:
            if template.name in opt_out.get(relative, []):
                continue
            pairs.append(
                (SHARED_DIR / "files" / relative, template / "template" / relative)
            )
        for library_file in sorted((SHARED_DIR / "library").glob("*.tmpl")):
            pairs.append((library_file, template / "library" / library_file.name))
    return pairs


def fork_review_drift(
    shared_dir: Path = SHARED_DIR, templates_dir: Path = TEMPLATES_DIR
) -> list[str]:
    """Opted-out forks whose canonical source changed since their last review.

    An opt_out entry turns a shared file into a per-template fork that the
    byte-for-byte sync no longer sees, so a fix applied to the canonical copy
    can silently never reach the fork (that happened: a canonical
    databricks.yml.tmpl comment missed the agent-app fork). fork_reviews in
    the manifest records the sha256 of the canonical file each fork was last
    reviewed against; when the canonical moves, this check fails until someone
    reviews the change against each fork, ports what applies, and runs
    `python scripts/sync_template_shared.py --acknowledge-forks`.

    The hash pins the canonical, not the fork: a fork is free to evolve on its
    own, but a canonical change must be consciously dispositioned for every
    fork it does not reach.
    """

    manifest = json.loads((shared_dir / "manifest.json").read_text(encoding="utf-8"))
    opt_out: dict[str, list[str]] = manifest.get("opt_out", {})
    reviews: dict[str, dict[str, str]] = manifest.get("fork_reviews", {})
    acknowledge = "python scripts/sync_template_shared.py --acknowledge-forks"
    drift: list[str] = []
    for relative, templates in opt_out.items():
        canonical = shared_dir / "files" / relative
        canonical_digest = (
            hashlib.sha256(canonical.read_bytes()).hexdigest()
            if canonical.is_file()
            else None
        )
        for template in templates:
            fork = templates_dir / template / "template" / relative
            recorded = reviews.get(relative, {}).get(template)
            if not fork.is_file():
                if recorded is not None:
                    drift.append(
                        f"fork_reviews records {relative} for {template}, but "
                        f"{template} opts out without a fork; run `{acknowledge}` "
                        "to drop the stale entry"
                    )
                continue
            if canonical_digest is None:
                drift.append(
                    f"templates/_shared/files/{relative} is missing but "
                    f"{template} still opts out and forks it; either restore "
                    "the canonical file or remove the opt_out entry and adopt "
                    "the fork as template-owned"
                )
            elif recorded is None:
                drift.append(
                    f"templates/{template}/template/{relative} is an opted-out "
                    f"fork with no recorded review; review it against "
                    f"templates/_shared/files/{relative}, port what applies, "
                    f"then run `{acknowledge}`"
                )
            elif recorded != canonical_digest:
                drift.append(
                    f"templates/_shared/files/{relative} changed since "
                    f"{template}'s opted-out fork was last reviewed against it; "
                    f"port what applies to the fork, then run `{acknowledge}`"
                )
    for relative, templates in reviews.items():
        for template in templates:
            if template not in opt_out.get(relative, []):
                drift.append(
                    f"fork_reviews records {relative} for {template}, which "
                    f"opt_out does not list; run `{acknowledge}` to drop the "
                    "orphaned entry"
                )
    return sorted(drift)


def acknowledge_forks(
    shared_dir: Path = SHARED_DIR, templates_dir: Path = TEMPLATES_DIR
) -> None:
    """Re-pin every existing opted-out fork to the current canonical hash.

    Run this only after actually reviewing the canonical change against each
    fork — it is the mechanical tail of the review, not a substitute for it.
    """

    manifest_path = shared_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reviews: dict[str, dict[str, str]] = {}
    for relative, templates in manifest.get("opt_out", {}).items():
        canonical = shared_dir / "files" / relative
        if not canonical.is_file():
            # A retired canonical cannot be reviewed against; fork_review_drift
            # reports it, and pinning nothing keeps that report alive.
            continue
        for template in templates:
            if not (templates_dir / template / "template" / relative).is_file():
                continue
            reviews.setdefault(relative, {})[template] = hashlib.sha256(
                canonical.read_bytes()
            ).hexdigest()
    manifest["fork_reviews"] = {
        relative: dict(sorted(templates.items()))
        for relative, templates in sorted(reviews.items())
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    try:
        displayed = manifest_path.relative_to(REPO_ROOT)
    except ValueError:
        displayed = manifest_path
    print(f"acknowledged fork reviews in {displayed}")


def unmanaged_duplicate_sources() -> list[str]:
    """Report byte-identical template source files outside shared ownership.

    Rendered locks are generated independently and intentionally excluded. A
    duplicate application/scaffold source must either be manifest-owned or be
    made workload-specific so future security fixes cannot silently drift.
    """

    managed = {destination.resolve() for _, destination in planned_copies()}
    groups: dict[str, list[Path]] = {}
    for template in discover_templates():
        for path in (template / "template").rglob("*"):
            if not path.is_file() or path.stat().st_size <= 200:
                continue
            if path.name.endswith(".lock"):
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            groups.setdefault(digest, []).append(path)

    drift: list[str] = []
    for paths in groups.values():
        if len(paths) < 2 or all(path.resolve() in managed for path in paths):
            continue
        rendered = ", ".join(str(path.relative_to(REPO_ROOT)) for path in sorted(paths))
        drift.append(
            "unmanaged byte-identical template source exceeds 200 bytes: " + rendered
        )
    return sorted(drift)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report drift between canonical and template copies; exit 1 on any.",
    )
    parser.add_argument(
        "--acknowledge-forks",
        action="store_true",
        help="After reviewing a canonical change against every opted-out fork, "
        "re-pin the recorded fork-review hashes to the current canonical files.",
    )
    args = parser.parse_args()

    if args.acknowledge_forks:
        acknowledge_forks()
        return 0

    drift: list[str] = []
    for source, destination in planned_copies():
        if not source.is_file():
            raise SystemExit(f"canonical file missing: {source}")
        in_sync = (
            destination.is_file() and destination.read_bytes() == source.read_bytes()
        )
        if in_sync:
            continue
        if args.check:
            state = "differs from" if destination.is_file() else "missing vs"
            drift.append(
                f"{destination.relative_to(REPO_ROOT)} {state} canonical "
                f"{source.relative_to(REPO_ROOT)}"
            )
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            print(f"synced {destination.relative_to(REPO_ROOT)}")

    drift.extend(_apply_schema_defaults(check=args.check))
    drift.extend(_apply_bundle_identifiers(check=args.check))
    drift.extend(_apply_project_urls(check=args.check))
    drift.extend(unmanaged_duplicate_sources())
    drift.extend(fork_review_drift())

    if drift:
        for line in drift:
            print(f"DRIFT: {line}", file=sys.stderr)
        print(
            "Run `python scripts/sync_template_shared.py` after editing the "
            "canonical copy under templates/_shared/, platform-identifiers.json, "
            "or compatibility.json.",
            file=sys.stderr,
        )
        return 1
    print(
        f"shared scaffold and generated defaults in sync across "
        f"{len(discover_templates())} template(s)"
        if args.check
        else "sync complete"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
