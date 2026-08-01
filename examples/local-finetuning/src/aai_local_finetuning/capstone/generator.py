"""Deterministic, offline capstone dataset generation."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

from .policy import DEFAULT_AS_OF, POLICY_VERSION, ReadinessPolicyEngine
from .schemas import (
    DATASET_SCHEMA_VERSION,
    DATASET_VERSION,
    CapstoneRecord,
    CapstoneRecordMetadata,
    DatasetSplit,
    SplitArtifact,
    SplitManifest,
)

DEFAULT_SEED = 42
TRAIN_COUNT = 400
VALIDATION_COUNT = 100
TEST_COUNT = 150

REQUIRED_FROZEN_TEST_SLICES: tuple[str, ...] = (
    "fully_production_ready",
    "missing_ownership",
    "missing_evaluation_dataset",
    "stale_evaluation",
    "missing_budget_policy",
    "missing_cost_tags",
    "missing_model_revision",
    "missing_rollback_plan",
    "one_critical_failure",
    "multiple_interacting_failures",
    "conflicting_metadata",
    "unknown_fields",
    "invalid_lifecycle",
    "long_manifest",
    "minimal_manifest",
    "unexpected_nulls",
    "unseen_combinations_of_known_failures",
)

ADDITIONAL_POLICY_SLICES: tuple[str, ...] = (
    "missing_business_domain",
    "failed_evaluation_thresholds",
    "missing_production_support",
    "missing_monitoring",
    "missing_data_classification",
    "missing_approvals",
    "unsupported_framework",
    "known_high_findings",
    "external_lookup_required",
    "human_judgment_required",
)

ALL_GENERATION_SLICES = REQUIRED_FROZEN_TEST_SLICES + ADDITIONAL_POLICY_SLICES

# These combinations are held out of train and validation by construction.
_UNSEEN_TEST_COMBINATIONS: tuple[tuple[str, ...], ...] = (
    ("owner", "monitoring_configured", "required_approvals_complete"),
    ("budget_policy", "rollback_plan", "framework"),
    ("evaluation_dataset", "cost_tags_present", "production_support_owner"),
    ("model_revision_pinned", "data_classifications", "known_high_severity_findings"),
)

_SEEN_FAILURE_COMBINATIONS: tuple[tuple[str, ...], ...] = (
    ("owner", "evaluation_dataset"),
    ("cost_tags_present", "budget_policy"),
    ("rollback_plan", "monitoring_configured"),
    ("production_support_owner", "required_approvals_complete"),
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _base_manifest(split: DatasetSplit, ordinal: int) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "application_name": f"readiness-study-{split.value}-{ordinal:04d}",
        "owner": "platform-ai",
        "business_domain": "customer-operations",
        "lifecycle": "production",
        "model_revision": "tiny-instruct@sha256:1f2e3d4c5b6a",
        "model_revision_pinned": True,
        "evaluation_dataset": "readiness-regression@1.0.0",
        "evaluation_last_run": (DEFAULT_AS_OF - timedelta(days=7)).isoformat(),
        "evaluation_thresholds_passed": True,
        "cost_tags_present": True,
        "budget_policy": "standard",
        "production_support_owner": "platform-operations",
        "rollback_plan": True,
        "monitoring_configured": True,
        "data_classifications": ["internal"],
        "required_approvals_complete": True,
        "framework": "mlflow-agent-server",
        "known_high_severity_findings": 0,
        "external_registry_lookup_required": False,
        "human_risk_judgment_required": False,
        "description": (
            "A synthetic application used only for the offline readiness capstone."
        ),
        "declared_controls": ["change-review", "regression-gate"],
    }


def _set_missing(manifest: dict[str, Any], field: str) -> None:
    nullable_fields = {
        "owner",
        "business_domain",
        "model_revision",
        "model_revision_pinned",
        "evaluation_dataset",
        "evaluation_last_run",
        "evaluation_thresholds_passed",
        "cost_tags_present",
        "budget_policy",
        "production_support_owner",
        "rollback_plan",
        "monitoring_configured",
        "data_classifications",
        "required_approvals_complete",
        "framework",
        "known_high_severity_findings",
    }
    if field not in nullable_fields:
        raise ValueError(f"Unsupported controlled failure field: {field}")
    manifest[field] = None


def _apply_slice(
    manifest: dict[str, Any],
    slice_name: str,
    *,
    split: DatasetSplit,
    variant: int,
) -> tuple[str, ...]:
    slices = [slice_name]
    if slice_name in {"fully_production_ready", "long_manifest"}:
        if slice_name == "long_manifest":
            manifest["description"] = " ".join(
                f"control-{number:03d}" for number in range(350)
            )
            manifest["declared_controls"] = [
                f"documented-control-{number:03d}" for number in range(64)
            ]
        return tuple(slices)
    if slice_name == "missing_ownership":
        _set_missing(manifest, "owner")
    elif slice_name == "missing_business_domain":
        _set_missing(manifest, "business_domain")
    elif slice_name == "missing_evaluation_dataset":
        _set_missing(manifest, "evaluation_dataset")
    elif slice_name == "stale_evaluation":
        manifest["evaluation_last_run"] = (
            DEFAULT_AS_OF - timedelta(days=31 + variant % 180)
        ).isoformat()
    elif slice_name == "failed_evaluation_thresholds":
        manifest["evaluation_thresholds_passed"] = False
    elif slice_name == "missing_budget_policy":
        _set_missing(manifest, "budget_policy")
    elif slice_name == "missing_cost_tags":
        manifest["cost_tags_present"] = False
    elif slice_name == "missing_model_revision":
        manifest["model_revision"] = None
        manifest["model_revision_pinned"] = False
    elif slice_name == "missing_production_support":
        _set_missing(manifest, "production_support_owner")
    elif slice_name == "missing_rollback_plan":
        manifest["rollback_plan"] = False
    elif slice_name == "missing_monitoring":
        manifest["monitoring_configured"] = False
    elif slice_name == "missing_data_classification":
        manifest["data_classifications"] = []
    elif slice_name == "missing_approvals":
        manifest["required_approvals_complete"] = False
    elif slice_name in {"one_critical_failure", "known_high_findings"}:
        manifest["known_high_severity_findings"] = 1 + variant % 4
    elif slice_name == "multiple_interacting_failures":
        fields = _SEEN_FAILURE_COMBINATIONS[variant % len(_SEEN_FAILURE_COMBINATIONS)]
        for field in fields:
            _set_missing(manifest, field)
        slices.extend(f"failure:{field}" for field in fields)
    elif slice_name == "conflicting_metadata":
        manifest["data_classifications"] = ["public", "restricted"]
    elif slice_name == "unknown_fields":
        # The engine reports this as unknown rather than trusting the asserted fact.
        manifest["registry_verified"] = True
    elif slice_name == "invalid_lifecycle":
        manifest["lifecycle"] = "candidate"
    elif slice_name == "minimal_manifest":
        application_name = manifest["application_name"]
        manifest.clear()
        manifest.update(
            {"schema_version": "1.0.0", "application_name": application_name}
        )
    elif slice_name == "unexpected_nulls":
        for field in (
            "owner",
            "lifecycle",
            "evaluation_last_run",
            "cost_tags_present",
            "rollback_plan",
        ):
            manifest[field] = None
    elif slice_name == "unseen_combinations_of_known_failures":
        if split is not DatasetSplit.TEST:
            raise ValueError(
                "unseen failure combinations are reserved for the frozen test split"
            )
        fields = _UNSEEN_TEST_COMBINATIONS[variant % len(_UNSEEN_TEST_COMBINATIONS)]
        for field in fields:
            if field == "known_high_severity_findings":
                manifest[field] = 2
            else:
                _set_missing(manifest, field)
        slices.extend(f"held_out_failure:{field}" for field in fields)
    elif slice_name == "unsupported_framework":
        manifest["framework"] = "unsupported-agent-framework"
    elif slice_name == "external_lookup_required":
        manifest["external_registry_lookup_required"] = True
    elif slice_name == "human_judgment_required":
        manifest["human_risk_judgment_required"] = True
    else:
        raise ValueError(f"Unknown capstone slice: {slice_name}")
    return tuple(slices)


def _slice_schedule(split: DatasetSplit, count: int, rng: random.Random) -> list[str]:
    eligible = list(ALL_GENERATION_SLICES)
    if split is not DatasetSplit.TEST:
        eligible.remove("unseen_combinations_of_known_failures")
    schedule = [eligible[index % len(eligible)] for index in range(count)]
    rng.shuffle(schedule)
    return schedule


def build_records(
    split: DatasetSplit,
    count: int,
    *,
    seed: int = DEFAULT_SEED,
) -> tuple[CapstoneRecord, ...]:
    """Build logical records in memory without touching the filesystem."""

    split_offset = {
        DatasetSplit.TRAIN: 11,
        DatasetSplit.VALIDATION: 23,
        DatasetSplit.TEST: 37,
    }[split]
    rng = random.Random(seed + split_offset)
    schedule = _slice_schedule(split, count, rng)
    engine = ReadinessPolicyEngine(as_of=DEFAULT_AS_OF)
    records: list[CapstoneRecord] = []
    for ordinal, slice_name in enumerate(schedule):
        manifest = _base_manifest(split, ordinal)
        slices = _apply_slice(
            manifest,
            slice_name,
            split=split,
            variant=ordinal + seed,
        )
        identity_material = {
            "dataset_version": DATASET_VERSION,
            "split": split.value,
            "manifest": manifest,
        }
        example_id = (
            f"capstone-{_sha256(_canonical_json(identity_material).encode())[:16]}"
        )
        records.append(
            CapstoneRecord(
                schema_version=DATASET_SCHEMA_VERSION,
                example_id=example_id,
                source_dataset="aai-application-production-readiness",
                source_version=DATASET_VERSION,
                manifest=manifest,
                expected_output=engine.review(manifest),
                metadata=CapstoneRecordMetadata(
                    split=split,
                    slices=slices,
                    generator_seed=seed,
                    policy_version=POLICY_VERSION,
                ),
            )
        )
    return tuple(records)


def _records_bytes(records: tuple[CapstoneRecord, ...]) -> bytes:
    lines = [
        _canonical_json(record.model_dump(mode="json", round_trip=True))
        for record in records
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def generate_capstone_dataset(
    output_dir: str | Path,
    *,
    seed: int = DEFAULT_SEED,
    writer: Callable[[Path, bytes], None] | None = None,
) -> SplitManifest:
    """Write the three splits and their content-addressed manifest.

    ``writer`` exists for teaching and tests; the default performs an ordinary
    local write.  No acquisition, network, clock, or model dependency is used.
    """

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    def default_writer(path: Path, content: bytes) -> None:
        path.write_bytes(content)

    write = writer or default_writer
    artifacts: list[SplitArtifact] = []
    content_hashes: list[str] = []
    for split, count in (
        (DatasetSplit.TRAIN, TRAIN_COUNT),
        (DatasetSplit.VALIDATION, VALIDATION_COUNT),
        (DatasetSplit.TEST, TEST_COUNT),
    ):
        records = build_records(split, count, seed=seed)
        content = _records_bytes(records)
        filename = f"{split.value}.jsonl"
        write(target / filename, content)
        content_hash = _sha256(content)
        id_bytes = ("\n".join(record.example_id for record in records) + "\n").encode()
        artifacts.append(
            SplitArtifact(
                split=split,
                path=filename,
                record_count=len(records),
                sha256=content_hash,
                example_ids_sha256=_sha256(id_bytes),
            )
        )
        content_hashes.append(f"{split.value}:{content_hash}")
    split_manifest = SplitManifest(
        schema_version=DATASET_SCHEMA_VERSION,
        dataset_name="aai-application-production-readiness",
        dataset_version=DATASET_VERSION,
        policy_version=POLICY_VERSION,
        seed=seed,
        strategy="controlled_policy_slices",
        frozen_test=True,
        artifacts=tuple(artifacts),
        dataset_sha256=_sha256("\n".join(content_hashes).encode()),
    )
    manifest_bytes = (
        json.dumps(
            split_manifest.model_dump(mode="json"),
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
    ).encode()
    write(target / "split-manifest.json", manifest_bytes)
    return split_manifest
