"""Cross-template shared-scaffold guarantees: sync drift and pin agreement."""

import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "templates" / "_shared"

_spec = importlib.util.spec_from_file_location(
    "sync_template_shared", ROOT / "scripts" / "sync_template_shared.py"
)
sync_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sync_module)

_release_spec = importlib.util.spec_from_file_location(
    "validate_release", ROOT / "scripts" / "validate_release.py"
)
release_module = importlib.util.module_from_spec(_release_spec)
_release_spec.loader.exec_module(release_module)


def test_shared_files_are_in_sync():
    drift = [
        f"{destination.relative_to(ROOT)} vs {source.relative_to(ROOT)}"
        for source, destination in sync_module.planned_copies()
        if not destination.is_file() or destination.read_bytes() != source.read_bytes()
    ]
    assert not drift, (
        "shared scaffold drift (edit the canonical copy under templates/_shared "
        "and run scripts/sync_template_shared.py): " + "; ".join(drift)
    )


def test_byte_identical_template_sources_have_shared_ownership():
    assert not sync_module.unmanaged_duplicate_sources()


def test_generated_tag_contract_uses_schema_two():
    for template in sync_module.discover_templates():
        for path in (template / "template").rglob("*.yml.tmpl"):
            text = path.read_text(encoding="utf-8")
            if "tag_schema_version:" not in text:
                continue
            assert 'tag_schema_version: "1"' not in text, path
            assert 'tag_schema_version: "2"' in text, path


def test_package_support_urls_follow_clone_repository_identifier():
    assert not sync_module._apply_project_urls(check=True)


def test_git_sdk_source_keeps_the_clone_repository_and_uses_release_metadata():
    source = "git+https://example.invalid/platform/aai-core@v0.3.0"

    assert sync_module.projected_pip_source(source) == (
        "git+https://example.invalid/platform/aai-core@{{.aai_core_source_ref}}"
    )
    artifact = "https://packages.example/aai_core-{{.aai_core_version}}.whl"
    assert sync_module.projected_pip_source(artifact) == artifact


def test_common_dependency_pins_agree():
    pins = json.loads((SHARED / "versions.json").read_text())["pins"]
    pin_pattern = re.compile(r"^([A-Za-z0-9_.\[\]-]+)==([A-Za-z0-9_.]+)\s*$")
    mismatches = []
    for template in sync_module.discover_templates():
        lock = template / "template" / "requirements.lock"
        if not lock.is_file():
            continue
        for line in lock.read_text().splitlines():
            match = pin_pattern.match(line.strip())
            if not match:
                continue
            package, version = match.groups()
            if package in pins and version != pins[package]:
                mismatches.append(
                    f"{template.name}: {package}=={version} "
                    f"(canonical {pins[package]})"
                )
    assert not mismatches, "pin drift vs templates/_shared/versions.json: " + "; ".join(
        mismatches
    )


def test_template_runtime_locks_are_exact_and_transitive():
    for template in sync_module.discover_templates():
        lock = template / "template" / "requirements.lock"
        text = lock.read_text(encoding="utf-8")
        pins = release_module.requirement_pins(lock)
        assert "Certified universal transitive runtime lock" in text
        assert not release_module.unpinned_requirement_lines(lock)
        assert len(pins) >= 20, f"{template.name} regressed to a direct-only lock"


def test_opted_out_forks_are_review_pinned():
    """Every opted-out fork carries a current fork_reviews pin: a canonical
    change cannot silently skip the forks it no longer reaches (that happened
    once — a canonical databricks.yml.tmpl comment missed the agent-app
    fork)."""

    assert not sync_module.fork_review_drift()


def _fork_review_fixture(tmp_path, *, fork_reviews):
    shared = tmp_path / "_shared"
    (shared / "files").mkdir(parents=True)
    (shared / "files" / "Makefile").write_text("canonical\n", encoding="utf-8")
    manifest = {
        "files": ["Makefile"],
        "opt_out": {"Makefile": ["agent-app", "prompt-app"]},
        "fork_reviews": fork_reviews,
    }
    (shared / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    templates = tmp_path / "templates"
    fork = templates / "agent-app" / "template" / "Makefile"
    fork.parent.mkdir(parents=True)
    fork.write_text("forked\n", encoding="utf-8")
    # prompt-app opts out without a fork: a legitimate absence.
    return shared, templates


def test_fork_review_detects_a_canonical_change_and_a_missing_pin(tmp_path):
    import hashlib

    canonical_digest = hashlib.sha256(b"canonical\n").hexdigest()

    shared, templates = _fork_review_fixture(
        tmp_path, fork_reviews={"Makefile": {"agent-app": canonical_digest}}
    )
    assert sync_module.fork_review_drift(shared, templates) == []

    (shared / "files" / "Makefile").write_text("canonical v2\n", encoding="utf-8")
    stale = sync_module.fork_review_drift(shared, templates)
    assert len(stale) == 1 and "changed since" in stale[0]

    shared, templates = _fork_review_fixture(tmp_path / "unpinned", fork_reviews={})
    unpinned = sync_module.fork_review_drift(shared, templates)
    assert len(unpinned) == 1 and "no recorded review" in unpinned[0]


def test_fork_review_rejects_entries_for_absent_or_unlisted_forks(tmp_path):
    shared, templates = _fork_review_fixture(
        tmp_path,
        fork_reviews={
            "Makefile": {"agent-app": "0" * 64, "prompt-app": "0" * 64},
            "unlisted.py": {"agent-app": "0" * 64},
        },
    )
    drift = sync_module.fork_review_drift(shared, templates)
    assert any("opts out without a fork" in line for line in drift)
    assert any("opt_out does not list" in line for line in drift)


def test_missing_canonical_is_reported_and_acknowledge_does_not_crash(tmp_path):
    shared, templates = _fork_review_fixture(
        tmp_path, fork_reviews={"Makefile": {"agent-app": "0" * 64}}
    )
    (shared / "files" / "Makefile").unlink()

    drift = sync_module.fork_review_drift(shared, templates)
    assert len(drift) == 1 and "is missing" in drift[0]

    sync_module.acknowledge_forks(shared, templates)

    # A retired canonical cannot be rubber-stamped: the report stays alive.
    still = sync_module.fork_review_drift(shared, templates)
    assert len(still) == 1 and "is missing" in still[0]


def test_acknowledge_forks_repins_and_clears_drift(tmp_path):
    shared, templates = _fork_review_fixture(tmp_path, fork_reviews={})
    assert sync_module.fork_review_drift(shared, templates)

    sync_module.acknowledge_forks(shared, templates)

    assert sync_module.fork_review_drift(shared, templates) == []
    reviews = json.loads((shared / "manifest.json").read_text())["fork_reviews"]
    assert set(reviews) == {"Makefile"}
    assert set(reviews["Makefile"]) == {"agent-app"}


def test_manifest_files_exist_and_nothing_orphaned():
    manifest = json.loads((SHARED / "manifest.json").read_text())
    for relative in manifest["files"]:
        assert (SHARED / "files" / relative).is_file(), relative
    declared = {str(Path(relative)) for relative in manifest["files"]}
    on_disk = {
        str(path.relative_to(SHARED / "files"))
        for path in (SHARED / "files").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    assert declared == on_disk, (
        f"manifest/files mismatch: only-declared={sorted(declared - on_disk)} "
        f"only-on-disk={sorted(on_disk - declared)}"
    )


def test_sdk_template_and_dependency_manifests_are_consistent():
    """A template must never advertise an SDK or certified stack that its
    repository release metadata does not describe."""

    release_module.validate_repository()
