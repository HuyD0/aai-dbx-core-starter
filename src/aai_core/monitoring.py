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
from re import fullmatch
from typing import Any


class FeedbackSourceKind(StrEnum):
    """Closed vocabulary for who or what produced a piece of feedback."""

    HUMAN = "human"
    LLM_JUDGE = "llm_judge"
    CODE = "code"


# Provenance identities are namespaced by source kind so a personal
# identity — a username, employee id, or email address — cannot pass as
# provenance. The shape cannot prove a string names a real group, but it
# forces the non-personal claim to be structural rather than aspirational,
# the same way the tagging standard requires owner_group over an
# individual email.
_SOURCE_NAMESPACES = {
    FeedbackSourceKind.HUMAN: "group",
    FeedbackSourceKind.LLM_JUDGE: "judge",
    FeedbackSourceKind.CODE: "code",
}


def log_feedback(
    *,
    trace_id: str,
    name: str,
    value: Any,
    rationale: str | None = None,
    source_kind: FeedbackSourceKind | str = FeedbackSourceKind.HUMAN,
    source_id: str,
    metadata: Mapping[str, Any] | None = None,
    span_id: str | None = None,
    mlflow_module: Any | None = None,
) -> None:
    """Attach governed feedback to a trace through native ``mlflow.log_feedback``.

    ``source_id`` is required so no governed feedback can be recorded
    without provenance, and it is namespaced by ``source_kind`` so a
    personal identity can never pass as provenance: human feedback uses
    ``group:<reviewer-group>`` (for example ``group:domain-reviewers``),
    judge feedback uses ``judge:<judge-name>``, and code scorer feedback
    uses ``code:<scorer-name>``. Usernames, employee ids, and email
    addresses are personal identities and are rejected.
    """

    # This is an untrusted-input boundary: str() would turn None into the
    # nonblank literal "None" and an int into a plausible id, addressing
    # the wrong trace instead of failing here. Require the declared type,
    # then normalize — an untrimmed trace id addresses a different (or no)
    # trace, and an untrimmed name records feedback under a label later
    # lookups will not match.
    for label, supplied in (("trace_id", trace_id), ("name", name)):
        if not isinstance(supplied, str):
            raise TypeError(f"{label} must be a string; got {type(supplied).__name__}")
    trace_id = trace_id.strip()
    if not trace_id:
        raise ValueError("trace_id must not be blank")
    name = name.strip()
    if not name:
        raise ValueError("name must not be blank")
    if span_id is not None:
        if not isinstance(span_id, str):
            raise TypeError(f"span_id must be a string; got {type(span_id).__name__}")
        span_id = span_id.strip()
        if not span_id:
            raise ValueError("span_id must not be blank when provided")
    if not isinstance(source_kind, FeedbackSourceKind):
        source_kind = FeedbackSourceKind(str(source_kind).strip().lower())
    if not str(source_id).strip():
        raise ValueError(
            "source_id must not be blank: governed feedback always carries "
            "a non-personal provenance identity such as 'group:domain-reviewers'"
        )
    namespace = _SOURCE_NAMESPACES[source_kind]
    prefix, separator, identifier = str(source_id).partition(":")
    if (
        not separator
        or prefix != namespace
        or not fullmatch(r"[A-Za-z0-9._-]{1,64}", identifier)
    ):
        raise ValueError(
            f"source_id must be a namespaced non-personal identity of the "
            f"form '{namespace}:<identifier>' for {source_kind.value} "
            "feedback; usernames, employee ids, and email addresses are "
            "personal identities and never valid provenance"
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
    select a trace for the regression dataset.

    Traces are returned unchanged; convert them before merging, because
    managed evaluation datasets accept record dictionaries or dataframes,
    not native ``Trace`` objects. Build each record's ``inputs`` from the
    trace request plus its still-valid expectation assessments, keep only
    rows whose expectations carry a nonblank string ``expected_response``
    (the reference-based scorers fail anything else as a dataset defect,
    so a malformed row would fail future gates for the wrong reason),
    then ``dataset.merge_records(records)`` — lab 14's connected curation
    cell is the reference implementation of that conversion.
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
    # Expectations are ground truth and issue references are triage
    # artifacts; neither is reviewed feedback.
    if getattr(assessment, "expectation", None) is not None:
        return False
    if getattr(assessment, "issue", None) is not None:
        return False
    # A failed scorer records an error, not reviewed feedback; a wildcard
    # match must never curate scorer failures into the regression dataset.
    if getattr(assessment, "error", None) is not None:
        return False
    return getattr(getattr(assessment, "feedback", None), "error", None) is None


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
