"""Deterministic production-readiness policy and evidence generation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from pydantic import ValidationError

from .schemas import (
    MANIFEST_SCHEMA_VERSION,
    OUTPUT_SCHEMA_VERSION,
    POLICY_VERSION,
    ApplicationManifest,
    CheckOutcome,
    CheckProvenance,
    Lifecycle,
    ReadinessCheck,
    ReadinessReview,
    ReadinessStatus,
    RuleDefinition,
    RuleKind,
    Severity,
)

DEFAULT_AS_OF = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)
MAX_EVALUATION_AGE = timedelta(days=30)

ALLOWED_BUDGET_POLICIES = frozenset({"restricted", "standard", "high_compute"})
ALLOWED_DATA_CLASSIFICATIONS = frozenset(
    {"public", "internal", "confidential", "restricted"}
)
SUPPORTED_FRAMEWORKS = frozenset({"native", "mlflow-agent-server", "langgraph"})


def _rule(
    rule_id: str,
    name: str,
    kind: RuleKind,
    severity: Severity,
    fields: tuple[str, ...],
    remediation_id: str,
    remediation_text: str,
) -> RuleDefinition:
    return RuleDefinition(
        rule_id=rule_id,
        name=name,
        kind=kind,
        failure_severity=severity,
        source_fields=fields,
        remediation_id=remediation_id,
        remediation_text=remediation_text,
    )


RULES: tuple[RuleDefinition, ...] = (
    _rule(
        "manifest_schema",
        "manifest_schema",
        RuleKind.DETERMINISTIC,
        Severity.HIGH,
        ("schema_version",),
        "manifest.use_supported_schema",
        (
            "Remove unknown fields and submit a manifest that validates against "
            "schema 1.0.0."
        ),
    ),
    _rule(
        "ownership",
        "ownership",
        RuleKind.DETERMINISTIC,
        Severity.HIGH,
        ("owner",),
        "ownership.assign_owner",
        "Assign a non-personal owning team or group.",
    ),
    _rule(
        "business_domain",
        "business_domain",
        RuleKind.DETERMINISTIC,
        Severity.MEDIUM,
        ("business_domain",),
        "governance.assign_business_domain",
        "Set the application's business domain.",
    ),
    _rule(
        "lifecycle",
        "lifecycle",
        RuleKind.DETERMINISTIC,
        Severity.HIGH,
        ("lifecycle",),
        "lifecycle.use_supported_value",
        "Use development, staging, or production; candidate is legacy terminology.",
    ),
    _rule(
        "model_revision_pinned",
        "model_revision_pinned",
        RuleKind.DETERMINISTIC,
        Severity.HIGH,
        ("model_revision_pinned", "model_revision"),
        "model.pin_revision",
        "Pin an immutable model revision and record its identifier.",
    ),
    _rule(
        "evaluation_dataset",
        "evaluation_dataset",
        RuleKind.DETERMINISTIC,
        Severity.HIGH,
        ("evaluation_dataset",),
        "evaluation.register_dataset",
        "Add a versioned regression evaluation dataset.",
    ),
    _rule(
        "evaluation_recency",
        "evaluation_recency",
        RuleKind.POLICY,
        Severity.HIGH,
        ("evaluation_last_run",),
        "evaluation.run_recently",
        "Run the regression evaluation within the last 30 days.",
    ),
    _rule(
        "evaluation_thresholds",
        "evaluation_thresholds",
        RuleKind.POLICY,
        Severity.CRITICAL,
        ("evaluation_thresholds_passed",),
        "evaluation.pass_thresholds",
        "Meet every critical evaluation threshold before production promotion.",
    ),
    _rule(
        "cost_tags",
        "cost_tags",
        RuleKind.DETERMINISTIC,
        Severity.HIGH,
        ("cost_tags_present",),
        "cost.add_required_tags",
        "Add all required cost-attribution tags.",
    ),
    _rule(
        "budget_policy",
        "budget_policy",
        RuleKind.POLICY,
        Severity.HIGH,
        ("budget_policy",),
        "cost.assign_budget_policy",
        "Assign a supported budget policy.",
    ),
    _rule(
        "production_support",
        "production_support",
        RuleKind.DETERMINISTIC,
        Severity.HIGH,
        ("production_support_owner",),
        "operations.assign_support_owner",
        "Assign a production-support team or group.",
    ),
    _rule(
        "rollback_plan",
        "rollback_plan",
        RuleKind.DETERMINISTIC,
        Severity.HIGH,
        ("rollback_plan",),
        "operations.add_rollback_plan",
        "Document and verify a rollback plan.",
    ),
    _rule(
        "monitoring",
        "monitoring",
        RuleKind.DETERMINISTIC,
        Severity.HIGH,
        ("monitoring_configured",),
        "operations.configure_monitoring",
        "Configure production monitoring and alerting.",
    ),
    _rule(
        "data_classification",
        "data_classification",
        RuleKind.POLICY,
        Severity.HIGH,
        ("data_classifications",),
        "governance.classify_data",
        "Add valid, non-conflicting data classifications.",
    ),
    _rule(
        "required_approvals",
        "required_approvals",
        RuleKind.POLICY,
        Severity.HIGH,
        ("required_approvals_complete",),
        "governance.complete_approvals",
        "Complete all approvals required by policy.",
    ),
    _rule(
        "framework_support",
        "framework_support",
        RuleKind.POLICY,
        Severity.HIGH,
        ("framework",),
        "runtime.use_supported_framework",
        "Use a framework supported by the platform policy.",
    ),
    _rule(
        "high_severity_findings",
        "high_severity_findings",
        RuleKind.DETERMINISTIC,
        Severity.CRITICAL,
        ("known_high_severity_findings",),
        "security.resolve_high_findings",
        "Resolve every known high-severity finding.",
    ),
    _rule(
        "external_registry_verification",
        "external_registry_verification",
        RuleKind.EXTERNAL_LOOKUP,
        Severity.MEDIUM,
        ("external_registry_lookup_required",),
        "registry.verify_externally",
        "Route the application to an authorized registry lookup.",
    ),
    _rule(
        "human_risk_review",
        "human_risk_review",
        RuleKind.HUMAN_JUDGMENT,
        Severity.MEDIUM,
        ("human_risk_judgment_required",),
        "risk.request_human_review",
        "Route the application to a qualified human reviewer.",
    ),
)


def rule_catalog() -> tuple[RuleDefinition, ...]:
    """Return the immutable ordered policy catalog."""

    return RULES


def _canonical_raw(
    manifest: ApplicationManifest | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(manifest, ApplicationManifest):
        return manifest.model_dump(mode="json")
    if not isinstance(manifest, Mapping):
        raise TypeError("manifest must be an ApplicationManifest or mapping")
    return dict(manifest)


def _non_empty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _parse_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def _schema_evidence(raw: Mapping[str, Any]) -> tuple[CheckOutcome, str]:
    try:
        encoded = json.dumps(raw, sort_keys=True, separators=(",", ":"))
        ApplicationManifest.model_validate_json(encoded)
    except (TypeError, ValueError, ValidationError) as exc:
        if isinstance(exc, ValidationError):
            locations = sorted(
                {".".join(str(part) for part in error["loc"]) for error in exc.errors()}
            )
            joined = ", ".join(locations[:8])
            suffix = " and more" if len(locations) > 8 else ""
            return (
                CheckOutcome.FAIL,
                f"Manifest does not match schema 1.0.0 at: {joined}{suffix}.",
            )
        return CheckOutcome.FAIL, "Manifest cannot be represented as valid JSON."
    return CheckOutcome.PASS, "Manifest matches the closed schema version 1.0.0."


def _evaluate_rule(
    rule: RuleDefinition,
    raw: Mapping[str, Any],
    as_of: datetime,
) -> tuple[
    CheckOutcome, str, Literal["manifest", "policy", "external_system", "human_review"]
]:
    rule_id = rule.rule_id
    if rule_id == "manifest_schema":
        result, evidence = _schema_evidence(raw)
        return result, evidence, "manifest"
    if rule_id == "ownership":
        if _non_empty_string(raw.get("owner")):
            return CheckOutcome.PASS, "An application owner is present.", "manifest"
        return CheckOutcome.FAIL, "No application owner is present.", "manifest"
    if rule_id == "business_domain":
        if _non_empty_string(raw.get("business_domain")):
            return CheckOutcome.PASS, "A business domain is present.", "manifest"
        return CheckOutcome.FAIL, "No business domain is present.", "manifest"
    if rule_id == "lifecycle":
        value = raw.get("lifecycle")
        if value in {member.value for member in Lifecycle}:
            return CheckOutcome.PASS, f"Lifecycle '{value}' is supported.", "manifest"
        if value == "candidate":
            return (
                CheckOutcome.FAIL,
                (
                    "Lifecycle 'candidate' is invalid legacy input; use development, "
                    "staging, or production."
                ),
                "policy",
            )
        return CheckOutcome.FAIL, "Lifecycle is missing or unsupported.", "policy"
    if rule_id == "model_revision_pinned":
        pinned = raw.get("model_revision_pinned") is True
        revision_present = _non_empty_string(raw.get("model_revision"))
        if pinned and revision_present:
            return (
                CheckOutcome.PASS,
                "An immutable model revision is declared and marked as pinned.",
                "manifest",
            )
        return (
            CheckOutcome.FAIL,
            "The model revision is not both identified and marked as pinned.",
            "manifest",
        )
    if rule_id == "evaluation_dataset":
        if _non_empty_string(raw.get("evaluation_dataset")):
            return (
                CheckOutcome.PASS,
                "A versioned evaluation dataset is registered.",
                "manifest",
            )
        return CheckOutcome.FAIL, "No evaluation dataset is registered.", "manifest"
    if rule_id == "evaluation_recency":
        timestamp = _parse_datetime(raw.get("evaluation_last_run"))
        if timestamp is None:
            return (
                CheckOutcome.FAIL,
                "No valid evaluation run timestamp is present.",
                "policy",
            )
        age = as_of - timestamp
        if timedelta(0) <= age <= MAX_EVALUATION_AGE:
            return (
                CheckOutcome.PASS,
                f"Evaluation age is {age.days} days, within the 30-day policy.",
                "policy",
            )
        if age < timedelta(0):
            return CheckOutcome.FAIL, "Evaluation timestamp is in the future.", "policy"
        return (
            CheckOutcome.FAIL,
            f"Evaluation age is {age.days} days, beyond the 30-day policy.",
            "policy",
        )
    if rule_id == "evaluation_thresholds":
        if raw.get("evaluation_thresholds_passed") is True:
            return (
                CheckOutcome.PASS,
                "All critical evaluation thresholds passed.",
                "manifest",
            )
        return (
            CheckOutcome.FAIL,
            "Critical evaluation thresholds did not pass.",
            "manifest",
        )
    if rule_id == "cost_tags":
        if raw.get("cost_tags_present") is True:
            return CheckOutcome.PASS, "Required cost tags are present.", "manifest"
        return CheckOutcome.FAIL, "Required cost tags are missing.", "manifest"
    if rule_id == "budget_policy":
        value = raw.get("budget_policy")
        if value in ALLOWED_BUDGET_POLICIES:
            return CheckOutcome.PASS, f"Budget policy '{value}' is supported.", "policy"
        return CheckOutcome.FAIL, "Budget policy is missing or unsupported.", "policy"
    if rule_id == "production_support":
        if _non_empty_string(raw.get("production_support_owner")):
            return (
                CheckOutcome.PASS,
                "A production-support owner is present.",
                "manifest",
            )
        return CheckOutcome.FAIL, "No production-support owner is present.", "manifest"
    if rule_id == "rollback_plan":
        if raw.get("rollback_plan") is True:
            return CheckOutcome.PASS, "A rollback plan is declared.", "manifest"
        return CheckOutcome.FAIL, "No rollback plan is declared.", "manifest"
    if rule_id == "monitoring":
        if raw.get("monitoring_configured") is True:
            return CheckOutcome.PASS, "Monitoring is configured.", "manifest"
        return CheckOutcome.FAIL, "Monitoring is not configured.", "manifest"
    if rule_id == "data_classification":
        values = raw.get("data_classifications")
        if not isinstance(values, (list, tuple)) or not values:
            return CheckOutcome.FAIL, "No data classification is present.", "policy"
        if any(not isinstance(item, str) for item in values):
            return (
                CheckOutcome.FAIL,
                "A data classification has an invalid type.",
                "policy",
            )
        normalized = {item.strip().lower() for item in values}
        unsupported = sorted(normalized - ALLOWED_DATA_CLASSIFICATIONS)
        if unsupported:
            return (
                CheckOutcome.FAIL,
                (
                    "Unsupported data classifications are present: "
                    f"{', '.join(unsupported)}."
                ),
                "policy",
            )
        if "public" in normalized and len(normalized) > 1:
            return (
                CheckOutcome.FAIL,
                (
                    "Conflicting metadata combines public with a non-public "
                    "classification."
                ),
                "policy",
            )
        return (
            CheckOutcome.PASS,
            "Data classifications are valid and consistent.",
            "policy",
        )
    if rule_id == "required_approvals":
        if raw.get("required_approvals_complete") is True:
            return CheckOutcome.PASS, "All required approvals are complete.", "manifest"
        return CheckOutcome.FAIL, "Required approvals are incomplete.", "manifest"
    if rule_id == "framework_support":
        value = raw.get("framework")
        if value in SUPPORTED_FRAMEWORKS:
            return CheckOutcome.PASS, f"Framework '{value}' is supported.", "policy"
        return CheckOutcome.FAIL, "The framework is missing or unsupported.", "policy"
    if rule_id == "high_severity_findings":
        value = raw.get("known_high_severity_findings")
        if type(value) is int and value == 0:
            return (
                CheckOutcome.PASS,
                "No known high-severity findings remain.",
                "manifest",
            )
        if type(value) is int and value > 0:
            return (
                CheckOutcome.FAIL,
                f"{value} known high-severity finding(s) remain unresolved.",
                "manifest",
            )
        return (
            CheckOutcome.FAIL,
            "High-severity finding status is missing or invalid.",
            "manifest",
        )
    if rule_id == "external_registry_verification":
        required = raw.get("external_registry_lookup_required")
        if required is False:
            return (
                CheckOutcome.PASS,
                "The manifest does not request an external registry lookup.",
                "manifest",
            )
        return (
            CheckOutcome.REVIEW,
            (
                "Registry state cannot be established from this manifest; external "
                "lookup is required."
            ),
            "external_system",
        )
    if rule_id == "human_risk_review":
        required = raw.get("human_risk_judgment_required")
        if required is False:
            return (
                CheckOutcome.PASS,
                "The manifest does not request additional human risk judgment.",
                "manifest",
            )
        return (
            CheckOutcome.REVIEW,
            (
                "Risk acceptability cannot be inferred; qualified human judgment is "
                "required."
            ),
            "human_review",
        )
    raise AssertionError(f"No evaluator is registered for rule {rule_id!r}")


def _make_check(
    rule: RuleDefinition,
    result: CheckOutcome,
    evidence: str,
    facts_origin: Literal["manifest", "policy", "external_system", "human_review"],
) -> ReadinessCheck:
    failing = result in {CheckOutcome.FAIL, CheckOutcome.REVIEW}
    return ReadinessCheck(
        name=rule.name,
        result=result,
        severity=rule.failure_severity if failing else Severity.INFO,
        evidence=evidence,
        remediation_id=rule.remediation_id if failing else None,
        remediation_text=rule.remediation_text if failing else None,
        provenance=CheckProvenance(
            policy_version=POLICY_VERSION,
            rule_id=rule.rule_id,
            rule_kind=rule.kind,
            source_fields=rule.source_fields,
            facts_origin=facts_origin,
        ),
    )


class ReadinessPolicyEngine:
    """Evaluate untrusted manifest JSON without network access or model calls."""

    def __init__(self, *, as_of: datetime = DEFAULT_AS_OF) -> None:
        if as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        self._as_of = as_of.astimezone(UTC)

    @property
    def as_of(self) -> datetime:
        return self._as_of

    def review(
        self, manifest: ApplicationManifest | Mapping[str, Any]
    ) -> ReadinessReview:
        raw = _canonical_raw(manifest)
        checks = tuple(
            _make_check(rule, *_evaluate_rule(rule, raw, self._as_of)) for rule in RULES
        )
        if any(check.result is CheckOutcome.FAIL for check in checks):
            status = ReadinessStatus.NOT_READY
        elif any(check.result is CheckOutcome.REVIEW for check in checks):
            status = ReadinessStatus.REVIEW_REQUIRED
        else:
            status = ReadinessStatus.READY
        return ReadinessReview(
            schema_version=OUTPUT_SCHEMA_VERSION,
            manifest_schema_version=MANIFEST_SCHEMA_VERSION,
            policy_version=POLICY_VERSION,
            status=status,
            as_of=self._as_of,
            checks=checks,
        )


def evaluate_manifest(
    manifest: ApplicationManifest | Mapping[str, Any],
    *,
    as_of: datetime = DEFAULT_AS_OF,
) -> ReadinessReview:
    """Convenience entry point for deterministic review."""

    return ReadinessPolicyEngine(as_of=as_of).review(manifest)
