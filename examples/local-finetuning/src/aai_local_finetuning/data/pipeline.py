"""End-to-end local preparation of the Bitext fine-tuning sample."""

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .bitext import load_bitext
from .duplicates import group_related_records, merge_record_groups
from .leakage import assert_no_leakage, check_split_leakage
from .manifests import file_digest, zip_member_digest
from .policies import policy_versions
from .processing import canonicalize_bitext, deduplicate_exact
from .quality import build_quality_report
from .schemas import (
    ChatExample,
    DatasetManifest,
    DatasetProvenance,
    FileDigest,
    LeakageReport,
    PreparationConfig,
    PreparationResult,
    QualityReport,
    SplitDescriptor,
)
from .splitting import curate_splits, to_chat_examples


@dataclass(frozen=True)
class _PreparedInMemory:
    train: tuple[ChatExample, ...]
    validation: tuple[ChatExample, ...]
    test: tuple[ChatExample, ...]
    quality: QualityReport
    leakage: LeakageReport
    source_member: str | None
    all_unique_ids: tuple[str, ...]


def prepare_dataset(
    input_path: str | Path,
    output_dir: str | Path,
    config: PreparationConfig | None = None,
    *,
    related_raw_paths: Sequence[str | Path] = (),
) -> PreparationResult:
    """Prepare deterministic chat splits from a local Bitext CSV or Kaggle ZIP."""

    source = Path(input_path).expanduser()
    settings = config or PreparationConfig()
    prepared = _prepare_in_memory(source, settings)
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)

    train_path = destination / "train.jsonl"
    validation_path = destination / "valid.jsonl"
    test_path = destination / "test.jsonl"
    quality_path = destination / "quality_report.json"
    leakage_path = destination / "leakage_report.json"
    manifest_path = destination / "manifest.json"

    _write_jsonl(train_path, prepared.train)
    _write_jsonl(validation_path, prepared.validation)
    _write_jsonl(test_path, prepared.test)
    _write_text(quality_path, prepared.quality.model_dump_json(indent=2) + "\n")
    _write_text(leakage_path, prepared.leakage.model_dump_json(indent=2) + "\n")

    artifacts = {
        "train": file_digest(train_path),
        "validation": file_digest(validation_path),
        "test": file_digest(test_path),
        "quality_report": file_digest(quality_path),
        "leakage_report": file_digest(leakage_path),
    }
    split_examples = {
        "train": prepared.train,
        "validation": prepared.validation,
        "test": prepared.test,
    }
    split_paths = {
        "train": train_path,
        "validation": validation_path,
        "test": test_path,
    }
    splits = {
        name: SplitDescriptor(
            logical_name=name,
            path=split_paths[name].name,
            record_count=len(examples),
            record_ids=tuple(example.example_id for example in examples),
            sha256=artifacts[name].sha256,
            frozen=name == "test",
        )
        for name, examples in split_examples.items()
    }
    source_resolved = source.resolve()
    raw_files = _raw_file_digests(
        source_resolved,
        source_member=prepared.source_member,
        related_paths=related_raw_paths,
    )
    manifest = DatasetManifest(
        schema_version="1.0.0",
        dataset_fingerprint=_dataset_fingerprint(prepared.all_unique_ids),
        dataset=DatasetProvenance(
            name=settings.source_dataset,
            source=settings.source_provider,
            owner=settings.source_owner,
            url=settings.source_url,
            version=settings.source_version,
            license=settings.source_license,
            date_accessed=settings.date_accessed,
        ),
        source=file_digest(source_resolved, display_path=source_resolved.name),
        raw_files=raw_files,
        source_member=prepared.source_member,
        processing=settings,
        code_revision=_code_revision(),
        processing_source_sha256=_processing_source_sha256(),
        processing_config_path=settings.processing_config_path,
        output_version=settings.output_version,
        split_strategy=settings.split_strategy,
        split_seed=settings.seed,
        policy_versions=policy_versions(),
        artifacts=artifacts,
        splits=splits,
    )
    _write_text(manifest_path, manifest.model_dump_json(indent=2) + "\n")
    return PreparationResult(
        output_dir=destination,
        train_path=train_path,
        validation_path=validation_path,
        test_path=test_path,
        quality_report_path=quality_path,
        leakage_report_path=leakage_path,
        manifest_path=manifest_path,
        manifest=manifest,
        quality_report=prepared.quality,
        leakage_report=prepared.leakage,
    )


def audit_dataset(
    input_path: str | Path,
    config: PreparationConfig | None = None,
) -> QualityReport:
    """Audit and simulate splitting without writing any generated artifacts."""

    settings = config or PreparationConfig()
    return _prepare_in_memory(Path(input_path).expanduser(), settings).quality


def _prepare_in_memory(source: Path, config: PreparationConfig) -> _PreparedInMemory:
    loaded = load_bitext(source)
    canonicalized = canonicalize_bitext(loaded.records)
    unique, exact_duplicate_count = deduplicate_exact(canonicalized.records)
    if not unique:
        raise ValueError("Bitext input contains no valid records after sanitization")
    grouping = group_related_records(
        unique,
        near_threshold=config.near_duplicate_threshold,
    )
    for _repair_attempt in range(10):
        curated = curate_splits(grouping, config)
        train = to_chat_examples(curated.train, config)
        validation = to_chat_examples(curated.validation, config)
        test = to_chat_examples(curated.test, config)
        if not train and not validation and not test:
            raise ValueError(
                "Bitext input contains no non-conflicting records to curate"
            )
        leakage = check_split_leakage(
            {"train": train, "validation": validation, "test": test},
            near_threshold=config.near_duplicate_threshold,
        )
        if leakage.passed:
            break
        example_groups = {
            example.example_id: example.metadata.split_group
            for example in (*train, *validation, *test)
        }
        links = {
            tuple(
                sorted(
                    (
                        example_groups[finding.example_id_left],
                        example_groups[finding.example_id_right],
                    )
                )
            )
            for finding in leakage.findings
            if finding.example_id_right in example_groups
            and finding.example_id_left in example_groups
        }
        if not links:
            break
        grouping = merge_record_groups(grouping, links)
    else:
        raise ValueError("duplicate-group repair did not converge after 10 passes")
    assert_no_leakage(leakage)
    quality = build_quality_report(
        loaded=loaded,
        canonicalized=canonicalized,
        unique_records=unique,
        exact_duplicate_count=exact_duplicate_count,
        grouping=grouping,
        splits=curated,
        config=config,
    )
    return _PreparedInMemory(
        train=train,
        validation=validation,
        test=test,
        quality=quality,
        leakage=leakage,
        source_member=loaded.source_member,
        all_unique_ids=tuple(record.example_id for record in unique),
    )


def _write_jsonl(path: Path, examples: tuple[ChatExample, ...]) -> None:
    content = "".join(example.model_dump_json() + "\n" for example in examples)
    _write_text(path, content)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _dataset_fingerprint(record_ids: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for record_id in sorted(record_ids):
        digest.update(record_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _raw_file_digests(
    source: Path,
    *,
    source_member: str | None,
    related_paths: Sequence[str | Path],
) -> tuple[FileDigest, ...]:
    paths = {source}
    for supplied in related_paths:
        related = Path(supplied).expanduser().resolve()
        if not related.is_file():
            raise FileNotFoundError(f"related raw file does not exist: {related}")
        paths.add(related)
    digests = [file_digest(path, display_path=path.name) for path in sorted(paths)]
    if source_member is not None:
        digests.append(zip_member_digest(source, source_member))
    return tuple(sorted(digests, key=lambda digest: digest.path))


def _code_revision() -> str:
    code_path = Path(__file__).resolve()
    try:
        revision = subprocess.run(
            ["git", "-C", str(code_path.parent), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(code_path.parent), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    return f"{revision}+dirty" if dirty else revision


def _processing_source_sha256() -> str:
    """Fingerprint every version-controlled data-pipeline module in this package."""

    digest = hashlib.sha256()
    package = Path(__file__).resolve().parent
    for path in sorted(package.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
