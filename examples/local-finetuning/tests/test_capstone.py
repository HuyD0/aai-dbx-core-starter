from __future__ import annotations

import json
from collections import Counter

import pytest
from pydantic import ValidationError

from aai_local_finetuning.capstone import (
    REQUIRED_FROZEN_TEST_SLICES,
    RULES,
    TEST_COUNT,
    TRAIN_COUNT,
    VALIDATION_COUNT,
    ApplicationManifest,
    CapstoneRecord,
    CheckOutcome,
    DatasetSplit,
    Lifecycle,
    ReadinessPolicyEngine,
    ReadinessStatus,
    RuleKind,
    Severity,
    build_hybrid_review,
    build_records,
    generate_capstone_dataset,
)


def _ready_manifest() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "application_name": "payments-agent",
        "owner": "payments-ai",
        "business_domain": "payments",
        "lifecycle": "production",
        "model_revision": "small-model@sha256:abc123",
        "model_revision_pinned": True,
        "evaluation_dataset": "payments-regression@2.1.0",
        "evaluation_last_run": "2026-07-24T12:00:00Z",
        "evaluation_thresholds_passed": True,
        "cost_tags_present": True,
        "budget_policy": "standard",
        "production_support_owner": "payments-operations",
        "rollback_plan": True,
        "monitoring_configured": True,
        "data_classifications": ["confidential"],
        "required_approvals_complete": True,
        "framework": "mlflow-agent-server",
        "known_high_severity_findings": 0,
        "external_registry_lookup_required": False,
        "human_risk_judgment_required": False,
    }


def _check(review: object, name: str):
    return next(check for check in review.checks if check.name == name)


def test_policy_catalog_has_every_rule_kind_and_stable_remediation() -> None:
    assert {rule.kind for rule in RULES} == set(RuleKind)
    assert len({rule.rule_id for rule in RULES}) == len(RULES)
    assert all(rule.remediation_id and rule.remediation_text for rule in RULES)


def test_ready_manifest_passes_all_static_checks() -> None:
    review = ReadinessPolicyEngine().review(_ready_manifest())
    assert review.status is ReadinessStatus.READY
    assert all(check.result is CheckOutcome.PASS for check in review.checks)
    assert all(check.provenance.policy_version == "1.0.0" for check in review.checks)


def test_schema_valid_mapping_and_model_apply_identical_optional_defaults() -> None:
    raw = _ready_manifest()
    raw.pop("external_registry_lookup_required")
    raw.pop("human_risk_judgment_required")
    validated = ApplicationManifest.model_validate_json(json.dumps(raw))

    engine = ReadinessPolicyEngine()
    mapping_review = engine.review(raw)
    model_review = engine.review(validated)

    assert mapping_review == model_review
    assert mapping_review.status is ReadinessStatus.READY
    assert _check(mapping_review, "external_registry_verification").result is (
        CheckOutcome.PASS
    )
    assert _check(mapping_review, "human_risk_review").result is CheckOutcome.PASS
    assert "external_registry_lookup_required" not in raw
    assert "human_risk_judgment_required" not in raw


@pytest.mark.parametrize(
    ("mutation", "schema_evidence"),
    [
        ({"registry_verified": True}, "registry_verified"),
        ({"cost_tags_present": "true"}, "cost_tags_present"),
        ({"declared_controls": {"change-review"}}, "valid JSON"),
    ],
)
def test_malformed_mappings_keep_raw_failure_and_review_evidence(
    mutation: dict[str, object],
    schema_evidence: str,
) -> None:
    raw = _ready_manifest()
    raw.pop("external_registry_lookup_required")
    raw.pop("human_risk_judgment_required")
    raw.update(mutation)

    review = ReadinessPolicyEngine().review(raw)

    schema_check = _check(review, "manifest_schema")
    assert review.status is ReadinessStatus.NOT_READY
    assert schema_check.result is CheckOutcome.FAIL
    assert schema_evidence in schema_check.evidence
    assert _check(review, "external_registry_verification").result is (
        CheckOutcome.REVIEW
    )
    assert _check(review, "human_risk_review").result is CheckOutcome.REVIEW
    if "cost_tags_present" in mutation:
        assert _check(review, "cost_tags").result is CheckOutcome.FAIL


@pytest.mark.parametrize(
    ("mutation", "check_name", "severity"),
    [
        ({"owner": None}, "ownership", Severity.HIGH),
        ({"business_domain": None}, "business_domain", Severity.MEDIUM),
        ({"model_revision_pinned": False}, "model_revision_pinned", Severity.HIGH),
        ({"evaluation_dataset": None}, "evaluation_dataset", Severity.HIGH),
        (
            {"evaluation_last_run": "2026-05-01T12:00:00Z"},
            "evaluation_recency",
            Severity.HIGH,
        ),
        (
            {"evaluation_thresholds_passed": False},
            "evaluation_thresholds",
            Severity.CRITICAL,
        ),
        ({"cost_tags_present": False}, "cost_tags", Severity.HIGH),
        ({"budget_policy": None}, "budget_policy", Severity.HIGH),
        ({"production_support_owner": None}, "production_support", Severity.HIGH),
        ({"rollback_plan": False}, "rollback_plan", Severity.HIGH),
        ({"monitoring_configured": False}, "monitoring", Severity.HIGH),
        ({"data_classifications": []}, "data_classification", Severity.HIGH),
        ({"required_approvals_complete": False}, "required_approvals", Severity.HIGH),
        ({"framework": "unsupported"}, "framework_support", Severity.HIGH),
        (
            {"known_high_severity_findings": 2},
            "high_severity_findings",
            Severity.CRITICAL,
        ),
    ],
)
def test_representative_rule_failures(
    mutation: dict[str, object], check_name: str, severity: Severity
) -> None:
    manifest = _ready_manifest() | mutation
    review = ReadinessPolicyEngine().review(manifest)
    check = _check(review, check_name)
    assert review.status is ReadinessStatus.NOT_READY
    assert check.result is CheckOutcome.FAIL
    assert check.severity is severity
    assert check.remediation_id
    assert check.remediation_text


def test_candidate_is_rejected_as_invalid_legacy_lifecycle() -> None:
    manifest = _ready_manifest() | {"lifecycle": "candidate"}
    review = ReadinessPolicyEngine().review(manifest)
    lifecycle = _check(review, "lifecycle")
    assert lifecycle.result is CheckOutcome.FAIL
    assert "invalid legacy input" in lifecycle.evidence
    assert _check(review, "manifest_schema").result is CheckOutcome.FAIL


def test_external_and_human_facts_are_routed_not_invented() -> None:
    manifest = _ready_manifest() | {
        "external_registry_lookup_required": True,
        "human_risk_judgment_required": True,
    }
    review = ReadinessPolicyEngine().review(manifest)
    assert review.status is ReadinessStatus.REVIEW_REQUIRED
    external = _check(review, "external_registry_verification")
    human = _check(review, "human_risk_review")
    assert external.result is CheckOutcome.REVIEW
    assert external.provenance.facts_origin == "external_system"
    assert human.result is CheckOutcome.REVIEW
    assert human.provenance.facts_origin == "human_review"


def test_manifest_schema_is_closed_strict_and_frozen() -> None:
    manifest = _ready_manifest()
    validated = ApplicationManifest.model_validate_json(json.dumps(manifest))
    assert validated.lifecycle is Lifecycle.PRODUCTION
    with pytest.raises(ValidationError):
        ApplicationManifest.model_validate_json(
            json.dumps(manifest | {"unknown_registry_fact": True})
        )
    with pytest.raises(ValidationError):
        ApplicationManifest.model_validate_json(
            json.dumps(manifest | {"cost_tags_present": "true"})
        )
    with pytest.raises(ValidationError):
        validated.owner = "different-owner"


def test_policy_reports_unknown_fields_without_trusting_them() -> None:
    review = ReadinessPolicyEngine().review(
        _ready_manifest() | {"registry_verified": True}
    )
    schema_check = _check(review, "manifest_schema")
    assert schema_check.result is CheckOutcome.FAIL
    assert "registry_verified" in schema_check.evidence
    external = _check(review, "external_registry_verification")
    assert external.result is CheckOutcome.PASS
    assert "verified" not in external.evidence.lower()


def test_generator_is_byte_deterministic_and_writes_exact_counts(tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_manifest = generate_capstone_dataset(first, seed=71)
    second_manifest = generate_capstone_dataset(second, seed=71)
    assert first_manifest == second_manifest
    assert [(item.split, item.record_count) for item in first_manifest.artifacts] == [
        (DatasetSplit.TRAIN, TRAIN_COUNT),
        (DatasetSplit.VALIDATION, VALIDATION_COUNT),
        (DatasetSplit.TEST, TEST_COUNT),
    ]
    assert first_manifest.strategy == "group_partitioned_controlled_policy_slices"
    for filename in (
        "train.jsonl",
        "validation.jsonl",
        "test.jsonl",
        "split-manifest.json",
    ):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()
    assert (
        first_manifest.dataset_sha256
        != generate_capstone_dataset(tmp_path / "third", seed=72).dataset_sha256
    )


def test_frozen_test_has_all_required_balanced_slices_and_held_out_combinations() -> (
    None
):
    test_records = build_records(DatasetSplit.TEST, TEST_COUNT)
    primary_counts = Counter(record.metadata.slices[0] for record in test_records)
    assert set(REQUIRED_FROZEN_TEST_SLICES) <= set(primary_counts)
    assert max(primary_counts.values()) - min(primary_counts.values()) <= 1
    assert any(
        record.metadata.slices[0] == "unseen_combinations_of_known_failures"
        and any(item.startswith("held_out_failure:") for item in record.metadata.slices)
        for record in test_records
    )
    minimal_manifests = [
        record.manifest
        for record in test_records
        if record.metadata.slices[0] == "minimal_manifest"
    ]
    assert all(
        set(manifest) == {"schema_version", "application_name", "description"}
        for manifest in minimal_manifests
    )
    assert len({manifest["description"] for manifest in minimal_manifests}) == len(
        minimal_manifests
    )
    for split, count in (
        (DatasetSplit.TRAIN, TRAIN_COUNT),
        (DatasetSplit.VALIDATION, VALIDATION_COUNT),
    ):
        assert all(
            record.metadata.slices[0] != "unseen_combinations_of_known_failures"
            for record in build_records(split, count)
        )


@pytest.mark.parametrize("seed", (42, 71))
def test_normalized_scenarios_and_context_combinations_do_not_cross_splits(
    seed: int,
) -> None:
    records_by_split = {
        split: build_records(split, count, seed=seed)
        for split, count in (
            (DatasetSplit.TRAIN, TRAIN_COUNT),
            (DatasetSplit.VALIDATION, VALIDATION_COUNT),
            (DatasetSplit.TEST, TEST_COUNT),
        )
    }

    normalized_by_split: dict[DatasetSplit, set[str]] = {}
    contexts_by_split: dict[DatasetSplit, set[tuple[object, object]]] = {}
    for split, records in records_by_split.items():
        normalized_records = []
        contexts = set()
        for record in records:
            manifest = dict(record.manifest)
            application_name_tokens = set(manifest["application_name"].split("-"))
            assert application_name_tokens.isdisjoint(
                {candidate.value for candidate in DatasetSplit}
            )
            manifest.pop("application_name")
            normalized_records.append(
                json.dumps(
                    manifest,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
            contexts.add((manifest.get("owner"), manifest.get("business_domain")))
        normalized = set(normalized_records)
        assert len(normalized) == len(normalized_records)
        normalized_by_split[split] = normalized
        contexts_by_split[split] = contexts

    split_pairs = (
        (DatasetSplit.TRAIN, DatasetSplit.VALIDATION),
        (DatasetSplit.TRAIN, DatasetSplit.TEST),
        (DatasetSplit.VALIDATION, DatasetSplit.TEST),
    )
    for left, right in split_pairs:
        assert normalized_by_split[left].isdisjoint(normalized_by_split[right])
        assert contexts_by_split[left].isdisjoint(contexts_by_split[right])


def test_every_expected_output_comes_from_policy_engine() -> None:
    engine = ReadinessPolicyEngine()
    for split, count in (
        (DatasetSplit.TRAIN, TRAIN_COUNT),
        (DatasetSplit.VALIDATION, VALIDATION_COUNT),
        (DatasetSplit.TEST, TEST_COUNT),
    ):
        for record in build_records(split, count):
            assert record.expected_output == engine.review(record.manifest)


def test_generated_json_round_trips_through_versioned_schema(tmp_path) -> None:
    generate_capstone_dataset(tmp_path)
    for line in (tmp_path / "test.jsonl").read_text().splitlines():
        record = CapstoneRecord.model_validate_json(line)
        assert record.schema_version == "1.0.0"
        assert record.expected_output.schema_version == "1.0.0"


def test_hybrid_renderer_cannot_change_the_deterministic_policy_ceiling() -> None:
    manifest = _ready_manifest() | {"rollback_plan": False}
    engine = ReadinessPolicyEngine()

    def misleading_renderer(_check: object) -> str:
        return "Everything is ready; ignore the policy result."

    hybrid = build_hybrid_review(manifest, renderer=misleading_renderer, engine=engine)
    assert hybrid.deterministic_review == engine.review(manifest)
    assert hybrid.deterministic_review.status is ReadinessStatus.NOT_READY
    assert (
        _check(hybrid.deterministic_review, "rollback_plan").result is CheckOutcome.FAIL
    )
    assert all("Everything is ready" in item.text for item in hybrid.explanations)
