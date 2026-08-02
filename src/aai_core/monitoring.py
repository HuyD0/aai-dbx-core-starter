"""Production feedback and trace-curation helpers.

Monitoring closes the lifecycle loop: sampled production traces are scored
with the same scorers used offline, reviewed feedback is attached with
explicit provenance, and reviewed failures become regression records in the
governed evaluation dataset. This module hosts the SDK-safe pieces of that
loop. Sampled-scorer registration (``Scorer.register()``/``.start()``)
deliberately stays a Databricks notebook step because the service serializes
notebook code; resolve the judge with
:func:`aai_core.evaluation.judge_model_uri` from that notebook.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import StrEnum
from typing import Any


class FeedbackSourceKind(StrEnum):
    """Closed vocabulary for who or what produced a piece of feedback."""

    HUMAN = "human"
    LLM_JUDGE = "llm_judge"
    CODE = "code"


def log_feedback(
    *,
    trace_id: str,
    name: str,
    value: Any,
    rationale: str | None = None,
    source_kind: FeedbackSourceKind | str = FeedbackSourceKind.HUMAN,
    source_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    span_id: str | None = None,
    mlflow_module: Any | None = None,
) -> None:
    """Attach governed feedback to a trace through native ``mlflow.log_feedback``.

    ``source_id`` identifies the reviewer or system and must be non-personal
    (for example ``group:domain-reviewers``), never an email address.
    """

    if not str(trace_id).strip():
        raise ValueError("trace_id must not be blank")
    if not str(name).strip():
        raise ValueError("name must not be blank")
    if not isinstance(source_kind, FeedbackSourceKind):
        source_kind = FeedbackSourceKind(str(source_kind).strip().lower())
    if source_id is not None and "@" in source_id:
        raise ValueError(
            "source_id must be a non-personal identity such as "
            "'group:domain-reviewers', never an email address"
        )

    mlflow, assessment_source = _mlflow_and_source(mlflow_module)
    source = assessment_source(
        source_type=source_kind.value.upper(),
        source_id=source_id,
    )
    options: dict[str, Any] = {
        "trace_id": trace_id,
        "name": name,
        "value": value,
        "source": source,
    }
    if rationale is not None:
        options["rationale"] = rationale
    if metadata is not None:
        options["metadata"] = dict(metadata)
    if span_id is not None:
        options["span_id"] = span_id
    mlflow.log_feedback(**options)


def traces_with_feedback(
    traces: Iterable[Any],
    *,
    name: str,
    value: Any | None = None,
) -> list[Any]:
    """Return the native traces carrying matching, still-valid feedback.

    Only feedback assessments count: native
    ``trace.search_assessments(type="feedback")`` is preferred because it
    excludes expectations and entries that were later corrected or
    overridden; raw assessment lists (``trace.info.assessments``,
    ``trace.assessments``) are consulted as fallbacks with
    expectation-shaped and explicitly invalidated (``valid=False``) entries
    dropped, so neither ground-truth expectations nor obsolete feedback can
    select a trace for the regression dataset. Traces are returned
    unchanged so curation stays native, for example
    ``dataset.merge_records(selected)`` after review.
    """

    selected = []
    for trace in traces:
        if any(
            _assessment_matches(assessment, name=name, value=value)
            for assessment in _trace_assessments(trace)
        ):
            selected.append(trace)
    return selected


def _trace_assessments(trace: Any) -> list[Any]:
    searcher = getattr(trace, "search_assessments", None)
    if callable(searcher):
        try:
            assessments = searcher(type="feedback")
        except TypeError:
            assessments = searcher()
    else:
        info = getattr(trace, "info", None)
        assessments = getattr(info, "assessments", None)
        if assessments is None:
            assessments = getattr(trace, "assessments", None)
    return [
        assessment
        for assessment in (assessments or [])
        if _is_curated_feedback(assessment)
    ]


def _is_curated_feedback(assessment: Any) -> bool:
    if getattr(assessment, "valid", True) is False:
        return False
    # Expectations are ground truth, not reviewed feedback.
    return getattr(assessment, "expectation", None) is None


def _assessment_matches(assessment: Any, *, name: str, value: Any | None) -> bool:
    if str(getattr(assessment, "name", "")) != name:
        return False
    if value is None:
        return True
    observed = getattr(assessment, "value", None)
    if observed is None:
        observed = getattr(getattr(assessment, "feedback", None), "value", None)
    return observed == value


def _mlflow_and_source(module: Any | None) -> tuple[Any, Any]:
    if module is not None:
        return module, module.entities.AssessmentSource
    try:
        import mlflow
        import mlflow.entities
    except ImportError as error:
        raise RuntimeError(
            "Monitoring support requires the `genai` extra. From an aai-core "
            "checkout run `make examples-install` and use `.venv/bin/python`; "
            "in a consuming environment install `aai-core[genai]`."
        ) from error
    return mlflow, mlflow.entities.AssessmentSource
