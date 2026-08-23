"""Create or verify immutable wheel evidence for one CI delivery run."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

_OBJECT_ID = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_SCHEMA_VERSION = "1"
_EVIDENCE_NAME = "release-evidence.json"


def _source_commit(value: str) -> str:
    normalized = value.strip().lower()
    if not _OBJECT_ID.fullmatch(normalized):
        raise ValueError("source commit must be a full 40- or 64-character object id")
    return normalized


def _artifact(artifact_dir: Path) -> Path:
    root = artifact_dir.resolve(strict=True)
    matches = sorted(root.glob("*.whl"))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one wheel in {root}, found {len(matches)}")
    artifact = matches[0]
    if artifact.is_symlink() or not artifact.is_file():
        raise ValueError("release artifact must be one regular, non-symlink wheel")
    if artifact.resolve(strict=True).parent != root:
        raise ValueError("release artifact must stay inside the artifact directory")
    return artifact


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_evidence(artifact_dir: Path, source_commit: str) -> Path:
    """Write canonical evidence beside the single wheel and return its path."""

    artifact = _artifact(artifact_dir)
    commit = _source_commit(source_commit)
    evidence = {
        "schema_version": _SCHEMA_VERSION,
        "source_commit": commit,
        "artifact": {
            "filename": artifact.name,
            "sha256": _sha256(artifact),
        },
    }
    output = artifact.parent / _EVIDENCE_NAME
    output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def verify_evidence(artifact_dir: Path, source_commit: str) -> Path:
    """Fail unless the staged wheel and checkout match the recorded evidence."""

    artifact = _artifact(artifact_dir)
    expected_commit = _source_commit(source_commit)
    evidence_path = artifact.resolve(strict=True).parent / _EVIDENCE_NAME
    document: Any = json.loads(evidence_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or set(document) != {
        "schema_version",
        "source_commit",
        "artifact",
    }:
        raise ValueError("release evidence has an invalid top-level contract")
    if document["schema_version"] != _SCHEMA_VERSION:
        raise ValueError("release evidence schema version is unsupported")
    if document["source_commit"] != expected_commit:
        raise ValueError("release evidence source commit does not match the checkout")
    recorded = document["artifact"]
    if not isinstance(recorded, dict) or set(recorded) != {"filename", "sha256"}:
        raise ValueError("release evidence artifact contract is invalid")
    if recorded["filename"] != artifact.name:
        raise ValueError("release evidence names a different wheel")
    digest = recorded["sha256"]
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("release evidence contains an invalid SHA-256 digest")
    if digest != _sha256(artifact):
        raise ValueError("release artifact digest does not match immutable evidence")
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("create", "verify"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("dist"))
    parser.add_argument("--source-commit", required=True)
    arguments = parser.parse_args()
    operation = create_evidence if arguments.command == "create" else verify_evidence
    try:
        result = operation(arguments.artifact_dir, arguments.source_commit)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    else:
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
