"""Behavioral tests for the downstream release-tag synchronization target.

The real Makefile recipes are copied into disposable Git repositories. The
expensive ``verify`` target and the repository-specific template synchronizer
are replaced with deterministic local stubs, so these tests exercise Git and
Make behavior without credentials, network access, or the development venv.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = ROOT / "Makefile"


@dataclass(frozen=True)
class Repositories:
    upstream: Path
    clone: Path
    base_commit: str


def run(
    cwd: Path,
    *command: str,
    check: bool = True,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        input=input_text,
        text=True,
    )
    if check and completed.returncode != 0:
        pytest.fail(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed


def git(
    cwd: Path,
    *arguments: str,
    check: bool = True,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return run(cwd, "git", *arguments, check=check, input_text=input_text)


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def replace(path: Path, old: str, new: str) -> None:
    original = path.read_text(encoding="utf-8")
    assert old in original, f"{old!r} is not present in {path}"
    path.write_text(original.replace(old, new), encoding="utf-8")


def target_block(source: str, target: str) -> str:
    """Return one target and its recipe from the repository Makefile."""

    lines = source.splitlines()
    try:
        start = next(
            index for index, line in enumerate(lines) if line.startswith(f"{target}:")
        )
    except StopIteration:
        pytest.fail(f"root Makefile has no {target!r} target")

    end = len(lines)
    header = re.compile(r"^[A-Za-z0-9_.%/-]+(?:\s+[A-Za-z0-9_.%/-]+)*:")
    for index in range(start + 1, len(lines)):
        if header.match(lines[index]):
            end = index
            break
    return "\n".join(lines[start:end]).rstrip() + "\n"


def harness_makefile() -> str:
    source = MAKEFILE.read_text(encoding="utf-8")
    return "\n".join(
        (
            "SHELL := /bin/bash",
            ".SHELLFLAGS := -eu -o pipefail -c",
            f"PYTHON := {sys.executable}",
            "SYNC_SENTINEL := base",
            ".PHONY: sync-upstream resolve-upstream sync-templates verify",
            "",
            "sync-templates:",
            "\t@$(PYTHON) scripts/sync_template_shared.py",
            "",
            "verify:",
            '\t@test -z "$$(git diff --name-only --diff-filter=U)"',
            "\t@echo verify >> .calls",
            "",
            target_block(source, "resolve-upstream"),
            target_block(source, "sync-upstream"),
        )
    )


SYNC_STUB = """\
import json
from pathlib import Path

fixture = json.loads(Path("platform-identifiers.json").read_text(encoding="utf-8"))
identifier = fixture["identifier"]

bundle = Path("databricks.yml")
bundle_lines = bundle.read_text(encoding="utf-8").splitlines()
bundle.write_text(
    "\\n".join(
        f"identifier: {identifier}" if line.startswith("identifier:") else line
        for line in bundle_lines
    )
    + "\\n",
    encoding="utf-8",
)

schema_path = Path("templates/demo/databricks_template_schema.json")
schema = json.loads(schema_path.read_text(encoding="utf-8"))
schema["identifier"] = identifier
schema_path.write_text(json.dumps(schema, indent=2) + "\\n", encoding="utf-8")

with Path(".calls").open("a", encoding="utf-8") as stream:
    stream.write("sync\\n")
"""


def commit_all(repository: Path, message: str) -> str:
    git(repository, "add", "--all")
    git(repository, "commit", "--quiet", "-m", message)
    return git(repository, "rev-parse", "HEAD").stdout.strip()


def tag_release(repository: Path, tag: str, *, annotated: bool = True) -> str:
    if annotated:
        git(repository, "tag", "--annotate", tag, "--message", f"Release {tag}")
    else:
        git(repository, "tag", tag)
    return git(repository, "rev-parse", "HEAD").stdout.strip()


def create_repositories(tmp_path: Path) -> Repositories:
    upstream = tmp_path / "upstream"
    clone = tmp_path / "clone"
    upstream.mkdir()
    git(upstream, "init", "--quiet", "--initial-branch=main")
    git(upstream, "config", "user.name", "Upstream Test")
    git(upstream, "config", "user.email", "upstream@example.invalid")

    write(upstream / "Makefile", harness_makefile())
    write(upstream / ".gitignore", ".calls\n")
    write(upstream / ".gitattributes", "platform-identifiers.json merge=keepours\n")
    write(upstream / "platform-identifiers.json", '{"identifier": "upstream"}\n')
    write(upstream / "databricks.yml", "identifier: upstream\nstructural: base\n")
    write(
        upstream / "templates/demo/databricks_template_schema.json",
        json.dumps({"identifier": "upstream", "structural": "base"}, indent=2) + "\n",
    )
    write(upstream / "scripts/sync_template_shared.py", SYNC_STUB)
    write(upstream / "tracked.txt", "base\n")
    write(upstream / "feature.txt", "base\n")
    base_commit = commit_all(upstream, "initial release")
    tag_release(upstream, "v1.0.0")

    run(tmp_path, "git", "clone", "--quiet", str(upstream), str(clone))
    git(clone, "config", "user.name", "Clone Test")
    git(clone, "config", "user.email", "clone@example.invalid")
    git(clone, "remote", "rename", "origin", "upstream")
    git(clone, "config", "merge.keepours.driver", "true")
    git(clone, "config", "merge.keepours.name", "keep clone fixture")
    return Repositories(upstream=upstream, clone=clone, base_commit=base_commit)


@pytest.fixture
def repositories(tmp_path: Path) -> Repositories:
    return create_repositories(tmp_path)


def make_sync(clone: Path, tag: str | None = None) -> subprocess.CompletedProcess[str]:
    command = ["make", "--silent", "sync-upstream"]
    if tag is not None:
        command.append(f"TAG={tag}")
    return run(clone, *command, check=False)


def calls(clone: Path) -> list[str]:
    path = clone / ".calls"
    return path.read_text(encoding="utf-8").splitlines() if path.exists() else []


def unmerged_paths(repository: Path) -> list[str]:
    output = git(
        repository, "diff", "--name-only", "--diff-filter=U"
    ).stdout.splitlines()
    return sorted(output)


def assert_merge_is_staged(repository: Path) -> None:
    assert git(repository, "rev-parse", "--verify", "MERGE_HEAD").returncode == 0
    assert git(repository, "diff", "--name-only").stdout == ""


def test_sync_upstream_is_phony_and_visible_in_help() -> None:
    source = MAKEFILE.read_text(encoding="utf-8")
    match = re.search(r"(?ms)^\.PHONY:(.*?)(?=^[^\t\s#])", source)
    assert match, "root Makefile has no .PHONY declaration"
    phony_targets = set(match.group(1).replace("\\", " ").split())
    assert {"sync-upstream", "resolve-upstream"}.issubset(phony_targets)

    completed = run(ROOT, "make", "--silent", "help")
    assert re.search(r"^\s+sync-upstream\s+", completed.stdout, re.MULTILINE)


def test_sync_upstream_requires_a_tag(repositories: Repositories) -> None:
    completed = make_sync(repositories.clone)

    assert completed.returncode == 2
    assert "TAG is required" in completed.stdout + completed.stderr
    assert calls(repositories.clone) == []
    assert git(repositories.clone, "rev-parse", "HEAD").stdout.strip() == (
        repositories.base_commit
    )


def test_sync_upstream_rejects_a_missing_tag_without_running_gates(
    repositories: Repositories,
) -> None:
    completed = make_sync(repositories.clone, "v9.9.9")

    assert completed.returncode != 0
    assert calls(repositories.clone) == []
    assert unmerged_paths(repositories.clone) == []
    assert (
        git(
            repositories.clone, "rev-parse", "--verify", "MERGE_HEAD", check=False
        ).returncode
        != 0
    )


def test_sync_upstream_rejects_a_lightweight_tag(
    repositories: Repositories,
) -> None:
    write(repositories.upstream / "feature.txt", "lightweight release\n")
    commit_all(repositories.upstream, "lightweight release")
    tag_release(repositories.upstream, "v1.1.0", annotated=False)

    completed = make_sync(repositories.clone, "v1.1.0")

    assert completed.returncode != 0
    assert calls(repositories.clone) == []
    assert git(repositories.clone, "rev-parse", "HEAD").stdout.strip() == (
        repositories.base_commit
    )
    assert (
        git(
            repositories.clone, "rev-parse", "--verify", "MERGE_HEAD", check=False
        ).returncode
        != 0
    )


def test_clean_annotated_merge_stays_uncommitted_and_staged(
    repositories: Repositories,
) -> None:
    write(repositories.upstream / "feature.txt", "annotated release\n")
    release_commit = commit_all(repositories.upstream, "annotated release")
    tag_release(repositories.upstream, "v1.1.0")

    completed = make_sync(repositories.clone, "v1.1.0")

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert git(repositories.clone, "rev-parse", "HEAD").stdout.strip() == (
        repositories.base_commit
    )
    assert (
        git(repositories.clone, "rev-parse", "MERGE_HEAD^{commit}").stdout.strip()
        == release_commit
    )
    assert (
        "feature.txt"
        in git(
            repositories.clone, "diff", "--cached", "--name-only"
        ).stdout.splitlines()
    )
    assert calls(repositories.clone)[-1] == "verify"
    assert_merge_is_staged(repositories.clone)


def test_dirty_worktree_failure_does_not_run_sync_or_verify(
    repositories: Repositories,
) -> None:
    write(repositories.upstream / "tracked.txt", "release\n")
    commit_all(repositories.upstream, "change tracked file")
    tag_release(repositories.upstream, "v1.1.0")
    write(repositories.clone / "tracked.txt", "uncommitted clone work\n")

    completed = make_sync(repositories.clone, "v1.1.0")

    assert completed.returncode != 0
    assert calls(repositories.clone) == []
    assert (repositories.clone / "tracked.txt").read_text(encoding="utf-8") == (
        "uncommitted clone work\n"
    )
    assert unmerged_paths(repositories.clone) == []


def test_missing_keepours_driver_stops_before_fetch_or_gates(
    repositories: Repositories,
) -> None:
    git(repositories.clone, "config", "--unset", "merge.keepours.driver")

    completed = make_sync(repositories.clone, "v1.1.0")

    assert completed.returncode != 0
    assert "configure the clone's keepours merge driver" in (
        completed.stdout + completed.stderr
    )
    assert calls(repositories.clone) == []
    assert unmerged_paths(repositories.clone) == []


def test_non_conflict_merge_failure_does_not_fall_through_to_gates(
    repositories: Repositories,
) -> None:
    tree = git(repositories.upstream, "write-tree").stdout.strip()
    unrelated_commit = git(
        repositories.upstream,
        "commit-tree",
        tree,
        input_text="unrelated release\n",
    ).stdout.strip()
    git(
        repositories.upstream,
        "tag",
        "--annotate",
        "v9.0.0",
        "--message",
        "Unrelated release",
        unrelated_commit,
    )

    completed = make_sync(repositories.clone, "v9.0.0")

    assert completed.returncode != 0
    assert calls(repositories.clone) == []
    assert unmerged_paths(repositories.clone) == []
    assert git(repositories.clone, "rev-parse", "HEAD").stdout.strip() == (
        repositories.base_commit
    )


def test_generated_conflicts_are_resolved_with_clone_identifiers(
    repositories: Repositories,
) -> None:
    write(repositories.clone / "platform-identifiers.json", '{"identifier": "clone"}\n')
    write(
        repositories.clone / "databricks.yml", "identifier: clone\nstructural: base\n"
    )
    write(
        repositories.clone / "templates/demo/databricks_template_schema.json",
        json.dumps({"identifier": "clone", "structural": "base"}, indent=2) + "\n",
    )
    commit_all(repositories.clone, "configure clone identifiers")
    clone_head = git(repositories.clone, "rev-parse", "HEAD").stdout.strip()

    write(
        repositories.upstream / "platform-identifiers.json",
        '{"identifier": "upstream-v2"}\n',
    )
    write(
        repositories.upstream / "databricks.yml",
        "identifier: upstream-v2\nstructural: release\n",
    )
    write(
        repositories.upstream / "templates/demo/databricks_template_schema.json",
        json.dumps({"identifier": "upstream-v2", "structural": "release"}, indent=2)
        + "\n",
    )
    commit_all(repositories.upstream, "change upstream identifiers")
    tag_release(repositories.upstream, "v1.1.0")

    completed = make_sync(repositories.clone, "v1.1.0")

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert git(repositories.clone, "rev-parse", "HEAD").stdout.strip() == clone_head
    assert unmerged_paths(repositories.clone) == []
    bundle = (repositories.clone / "databricks.yml").read_text(encoding="utf-8")
    schema = json.loads(
        (
            repositories.clone / "templates/demo/databricks_template_schema.json"
        ).read_text(encoding="utf-8")
    )
    assert "identifier: clone" in bundle
    assert "structural: release" in bundle
    assert schema == {"identifier": "clone", "structural": "release"}
    assert git(repositories.clone, "show", ":databricks.yml").stdout == bundle
    assert calls(repositories.clone)[-1] == "verify"
    assert_merge_is_staged(repositories.clone)


def test_makefile_conflict_stops_before_verify(repositories: Repositories) -> None:
    replace(
        repositories.clone / "Makefile",
        "SYNC_SENTINEL := base",
        "SYNC_SENTINEL := clone",
    )
    commit_all(repositories.clone, "customize clone Makefile")

    replace(
        repositories.upstream / "Makefile",
        "SYNC_SENTINEL := base",
        "SYNC_SENTINEL := upstream",
    )
    commit_all(repositories.upstream, "change upstream Makefile")
    tag_release(repositories.upstream, "v1.1.0")

    completed = make_sync(repositories.clone, "v1.1.0")

    assert completed.returncode != 0
    assert "verify" not in calls(repositories.clone)
    assert "Makefile" in unmerged_paths(repositories.clone)


def test_successful_restamp_changes_are_in_the_index(
    repositories: Repositories,
) -> None:
    write(repositories.clone / "platform-identifiers.json", '{"identifier": "clone"}\n')
    clone_head = commit_all(repositories.clone, "configure clone fixture")

    write(
        repositories.upstream / "databricks.yml",
        "identifier: upstream\nstructural: release\n",
    )
    write(
        repositories.upstream / "templates/demo/databricks_template_schema.json",
        json.dumps({"identifier": "upstream", "structural": "release"}, indent=2)
        + "\n",
    )
    commit_all(repositories.upstream, "change generated structure")
    tag_release(repositories.upstream, "v1.1.0")

    completed = make_sync(repositories.clone, "v1.1.0")

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert git(repositories.clone, "rev-parse", "HEAD").stdout.strip() == clone_head
    assert_merge_is_staged(repositories.clone)
    staged_bundle = git(repositories.clone, "show", ":databricks.yml").stdout
    staged_schema = json.loads(
        git(
            repositories.clone,
            "show",
            ":templates/demo/databricks_template_schema.json",
        ).stdout
    )
    assert "identifier: clone" in staged_bundle
    assert "structural: release" in staged_bundle
    assert staged_schema == {"identifier": "clone", "structural": "release"}
    assert calls(repositories.clone)[-1] == "verify"
