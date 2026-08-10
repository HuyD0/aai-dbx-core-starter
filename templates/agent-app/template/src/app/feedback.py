"""Native MLflow assessment helpers for feedback and reviewed curation."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Literal

import mlflow
from mlflow.entities import AssessmentSource, AssessmentSourceType

_REVIEWED = "aai.reviewed"
_REVIEWER_GROUP = "aai.reviewer_group"
_LEARNING_ELIGIBILITY = "aai.learning_eligibility"
_SOURCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")


def record_human_feedback(
    *,
    trace_id: str,
    name: str,
    value: Any,
    source_id: str,
    rationale: str | None = None,
    span_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    mlflow_module: Any = mlflow,
) -> Any:
    """Log feedback on its originating trace and return MLflow's Assessment."""

    return mlflow_module.log_feedback(
        trace_id=_required("trace_id", trace_id),
        name=_required("name", name),
        value=value,
        source=_human_source(_opaque_source_id("source_id", source_id)),
        rationale=rationale,
        metadata=dict(metadata or {}),
        span_id=span_id,
    )


def curate_reviewed_expectation(
    *,
    trace_id: str,
    name: str,
    value: Any,
    reviewer_group: str,
    reviewed: bool,
    span_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    mlflow_module: Any = mlflow,
) -> Any:
    """Log curation-ready ground truth only after explicit human review.

    This records native MLflow expectation evidence; it does not mutate the
    release dataset. A reviewed repository change remains the promotion path.
    """

    if reviewed is not True:
        raise ValueError(
            "Only explicitly reviewed expectations may enter the curation path"
        )
    group = _opaque_source_id("reviewer_group", reviewer_group)
    details = dict(metadata or {})
    controlled = {_REVIEWED, _REVIEWER_GROUP}.intersection(details)
    if controlled:
        raise ValueError(
            "Review metadata is controlled by the curation helper: "
            + ", ".join(sorted(controlled))
        )
    details.update({_REVIEWED: "true", _REVIEWER_GROUP: group})
    return mlflow_module.log_expectation(
        trace_id=_required("trace_id", trace_id),
        name=_required("name", name),
        value=value,
        source=_human_source(group),
        metadata=details,
        span_id=span_id,
    )


def record_intervention(
    *,
    trace_id: str,
    value: Any,
    source_id: str,
    rationale: str | None = None,
    span_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    mlflow_module: Any = mlflow,
) -> Any:
    """Record a human intervention as behavior evidence, not training data."""

    return record_human_feedback(
        trace_id=trace_id,
        name="aai.intervention",
        value=value,
        source_id=source_id,
        rationale=rationale,
        span_id=span_id,
        metadata=_review_required_metadata(metadata),
        mlflow_module=mlflow_module,
    )


def record_business_outcome(
    *,
    trace_id: str,
    value: Any,
    source_id: str,
    source_type: Literal["code", "human"] = "code",
    rationale: str | None = None,
    span_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    mlflow_module: Any = mlflow,
) -> Any:
    """Attach a delayed business outcome with review-required eligibility."""

    source_types = {
        "code": AssessmentSourceType.CODE,
        "human": AssessmentSourceType.HUMAN,
    }
    try:
        assessment_source = source_types[source_type]
    except KeyError as error:
        raise ValueError("source_type must be 'code' or 'human'") from error
    return mlflow_module.log_feedback(
        trace_id=_required("trace_id", trace_id),
        name="aai.business_outcome",
        value=value,
        source=AssessmentSource(
            source_type=assessment_source,
            source_id=_opaque_source_id("source_id", source_id),
        ),
        rationale=rationale,
        metadata=_review_required_metadata(metadata),
        span_id=span_id,
    )


def _human_source(source_id: str) -> AssessmentSource:
    return AssessmentSource(
        source_type=AssessmentSourceType.HUMAN,
        source_id=source_id,
    )


def _review_required_metadata(
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    details = dict(metadata or {})
    if _LEARNING_ELIGIBILITY in details:
        raise ValueError(
            f"{_LEARNING_ELIGIBILITY} is controlled by the feedback helper"
        )
    details[_LEARNING_ELIGIBILITY] = "review_required"
    return details


def _opaque_source_id(field: str, value: str) -> str:
    identifier = _required(field, value)
    lowered = identifier.casefold()
    if (
        "@" in identifier
        or not _SOURCE_ID.fullmatch(identifier)
        or lowered.startswith(
            (
                "bearer",
                "dapi",
                "eyj",
                "github_pat_",
                "ghp_",
                "gho_",
                "ghr_",
                "ghs_",
                "ghu_",
                "pat-",
                "sk-",
            )
        )
    ):
        raise ValueError(
            f"{field} must be a stable non-personal, non-secret group or system id"
        )
    return identifier


def _required(field: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()
