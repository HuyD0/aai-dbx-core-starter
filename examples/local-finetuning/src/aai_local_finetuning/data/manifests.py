"""SHA-256 dataset manifests and offline artifact verification."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

from .leakage import load_chat_jsonl
from .schemas import DatasetManifest, FileDigest, ManifestVerification


class DatasetIntegrityError(RuntimeError):
    """Raised when prepared data no longer matches its content manifest."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_digest(path: str | Path, *, display_path: str | None = None) -> FileDigest:
    source = Path(path)
    return FileDigest(
        path=display_path or source.name,
        sha256=sha256_file(source),
        size_bytes=source.stat().st_size,
    )


def zip_member_digest(path: str | Path, member: str) -> FileDigest:
    """Hash one ZIP member without extracting or mutating the raw archive."""

    archive_path = Path(path)
    digest = hashlib.sha256()
    with zipfile.ZipFile(archive_path) as archive:
        info = archive.getinfo(member)
        with archive.open(info, "r") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return FileDigest(
        path=f"{archive_path.name}!/{member}",
        sha256=digest.hexdigest(),
        size_bytes=info.file_size,
    )


def verify_manifest(
    manifest_or_directory: str | Path,
    *,
    source_path: str | Path | None = None,
) -> ManifestVerification:
    """Verify generated artifacts, split IDs, and optionally the immutable source."""

    supplied = Path(manifest_or_directory)
    manifest_path = supplied / "manifest.json" if supplied.is_dir() else supplied
    if not manifest_path.is_file():
        raise FileNotFoundError(f"dataset manifest does not exist: {manifest_path}")
    manifest = DatasetManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    root = manifest_path.parent.resolve()
    mismatches: list[str] = []
    checked = 0

    for artifact_name, expected in sorted(manifest.artifacts.items()):
        artifact_path = (root / expected.path).resolve()
        if not artifact_path.is_relative_to(root):
            mismatches.append(f"{artifact_name}: path escapes manifest directory")
            continue
        if not artifact_path.is_file():
            mismatches.append(f"{artifact_name}: file is missing")
            continue
        checked += 1
        actual_size = artifact_path.stat().st_size
        actual_hash = sha256_file(artifact_path)
        if actual_size != expected.size_bytes:
            mismatches.append(f"{artifact_name}: size mismatch")
        if actual_hash != expected.sha256:
            mismatches.append(f"{artifact_name}: SHA-256 mismatch")

    for split_name, descriptor in sorted(manifest.splits.items()):
        split_path = (root / descriptor.path).resolve()
        if not split_path.is_relative_to(root) or not split_path.is_file():
            continue
        try:
            examples = load_chat_jsonl(split_path)
        except (OSError, ValueError) as error:
            mismatches.append(f"{split_name}: invalid JSONL ({type(error).__name__})")
            continue
        actual_ids = tuple(example.example_id for example in examples)
        if len(examples) != descriptor.record_count:
            mismatches.append(f"{split_name}: record count mismatch")
        if actual_ids != descriptor.record_ids:
            mismatches.append(f"{split_name}: record ID manifest mismatch")
        if sha256_file(split_path) != descriptor.sha256:
            mismatches.append(f"{split_name}: split SHA-256 mismatch")

    if source_path is not None:
        source = Path(source_path)
        if not source.is_file():
            mismatches.append("source: file is missing")
        else:
            checked += 1
            if source.stat().st_size != manifest.source.size_bytes:
                mismatches.append("source: size mismatch")
            if sha256_file(source) != manifest.source.sha256:
                mismatches.append("source: SHA-256 mismatch")

    return ManifestVerification(
        valid=not mismatches,
        checked_files=checked,
        mismatches=tuple(mismatches),
    )


def require_valid_manifest(
    manifest_or_directory: str | Path,
    *,
    source_path: str | Path | None = None,
) -> ManifestVerification:
    """Return verified evidence or fail closed with actionable mismatch details."""

    try:
        verification = verify_manifest(
            manifest_or_directory,
            source_path=source_path,
        )
    except (OSError, ValueError) as error:
        raise DatasetIntegrityError(
            "prepared dataset manifest could not be verified "
            f"({type(error).__name__})"
        ) from error
    if verification.valid:
        return verification
    detail = "\n".join(f"- {mismatch}" for mismatch in verification.mismatches)
    raise DatasetIntegrityError("prepared dataset integrity mismatch:\n" + detail)
