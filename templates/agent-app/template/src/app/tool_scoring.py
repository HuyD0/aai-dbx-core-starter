"""MLflow 3.15.1-compatible tool trajectory, decision, and operation scoring."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Mapping
from typing import Any

_DECISION_TYPE = "agent.decision.type"
_SELECTED_ACTION = "agent.decision.selected_action"
_TOOL_SELECTION = "tool_selection"
_EVIDENCE_SUFFICIENCY = "evidence_sufficiency"
_ANSWER_READINESS = "answer_readiness"
_OTEL_TOOL_NAME = "gen_ai.tool.name"
_OTEL_TOOL_ARGUMENTS = "gen_ai.tool.call.arguments"


def exact_tool_call_scorer() -> Callable[..., Any]:
    """Return an exact, unordered MLflow tool-call scorer.

    MLflow 3.15.1's native scorer can fall back to model-assisted extraction, treats
    an empty expected list as missing, and does not fully preserve duplicate-call
    multiplicity. This deterministic compatibility scorer canonicalizes SDK-native
    and OTel GenAI TOOL spans, then compares names and arguments in code only.
    """

    from mlflow.entities import (
        AssessmentSource,
        AssessmentSourceType,
        Feedback,
    )
    from mlflow.genai.scorers import scorer

    source = AssessmentSource(
        source_type=AssessmentSourceType.CODE,
        source_id="aai-exact-tool-calls",
    )

    @scorer(name="tool_call_correctness")
    def score(
        trace: Any,
        expectations: Mapping[str, Any] | None,
    ) -> Any:
        expected = (expectations or {}).get("expected_tool_calls")
        if not isinstance(expected, list):
            return Feedback(
                name="tool_call_correctness",
                error=ValueError("expected_tool_calls must be a list"),
                source=source,
            )

        try:
            expected_counts = _expected_counts(expected)
            actual_spans = list(trace.search_spans(span_type="TOOL"))
            actual_counts = _actual_counts(actual_spans)
        except (TypeError, ValueError) as error:
            return Feedback(
                name="tool_call_correctness",
                error=error,
                source=source,
            )

        if expected_counts != actual_counts:
            return Feedback(
                name="tool_call_correctness",
                value="no",
                rationale=(
                    f"Expected {sum(expected_counts.values())} exact tool call(s), "
                    f"but traced {sum(actual_counts.values())}; names, arguments, "
                    "or multiplicity differed."
                ),
                source=source,
            )
        if not expected:
            return Feedback(
                name="tool_call_correctness",
                value="yes",
                rationale="No tool call was expected or traced.",
                source=source,
            )
        return Feedback(
            name="tool_call_correctness",
            value="yes",
            rationale=(
                "All expected tool names, arguments, and multiplicities match "
                "the canonical TOOL spans."
            ),
            source=source,
        )

    return score


def trace_execution_success_scorer() -> Callable[..., Any]:
    """Report whether root and TOOL spans completed without an error status.

    This is operational evidence only. It does not claim the answer was
    correct, the trajectory was expected, or the selected action was
    appropriate.
    """

    from mlflow.entities import (
        AssessmentSource,
        AssessmentSourceType,
        Feedback,
    )
    from mlflow.genai.scorers import scorer

    source = AssessmentSource(
        source_type=AssessmentSourceType.CODE,
        source_id="aai-trace-execution-success",
    )

    @scorer(name="trace_execution_success")
    def score(trace: Any) -> Any:
        try:
            roots = _root_spans(trace)
            tools = _ordered_spans(trace.search_spans(span_type="TOOL"))
            inspected = [*roots, *tools]
            statuses = [(_span_label(span), _status_code(span)) for span in inspected]
        except (TypeError, ValueError) as error:
            return Feedback(
                name="trace_execution_success",
                error=error,
                source=source,
            )

        errors = [label for label, status in statuses if status == "ERROR"]
        if errors:
            return Feedback(
                name="trace_execution_success",
                value="no",
                rationale=(
                    "Observed ERROR status on root/TOOL span(s): "
                    + ", ".join(errors)
                    + "."
                ),
                source=source,
            )
        return Feedback(
            name="trace_execution_success",
            value="yes",
            rationale="No root or TOOL span reported an ERROR status.",
            source=source,
        )

    return score


def decision_action_consistency_scorer() -> Callable[..., Any]:
    """Check that claimed tool selections match observed TOOL spans in order.

    A TOOL span remains the authoritative execution record. A failed tool can
    therefore still be consistent with the preceding selection decision; its
    failure status is assessed from the TOOL span, not copied into the decision.
    Only an explicit request-root ERROR waives a missing terminal decision,
    because a failed tool may be recovered before the request completes.
    """

    from mlflow.entities import (
        AssessmentSource,
        AssessmentSourceType,
        Feedback,
    )
    from mlflow.genai.scorers import scorer

    source = AssessmentSource(
        source_type=AssessmentSourceType.CODE,
        source_id="aai-decision-action-consistency",
    )

    @scorer(name="decision_action_consistency")
    def score(trace: Any) -> Any:
        try:
            decision_spans = _decision_spans(trace)
            tool_selection_spans = _tool_selection_spans(decision_spans)
            selected = _selected_tool_actions(tool_selection_spans)
            observed_tool_spans = _ordered_spans(trace.search_spans(span_type="TOOL"))
            observed = _tool_span_names(observed_tool_spans)
            root_errored = _root_reports_error(trace)
        except (TypeError, ValueError) as error:
            return Feedback(
                name="decision_action_consistency",
                error=error,
                source=source,
            )

        if not decision_spans:
            return Feedback(
                name="decision_action_consistency",
                value="no",
                rationale="No decision span was present in the agent trace.",
                source=source,
            )
        if selected != observed:
            return Feedback(
                name="decision_action_consistency",
                value="no",
                rationale=(
                    f"Decision spans selected {selected!r}, while TOOL spans "
                    f"observed {observed!r}."
                ),
                source=source,
            )
        if structural_error := _decision_execution_structure_error(
            tool_selection_spans,
            observed_tool_spans,
        ):
            return Feedback(
                name="decision_action_consistency",
                value="no",
                rationale=structural_error,
                source=source,
            )
        terminal = _terminal_decision(
            decision_spans,
            tool_evidence_exists=bool(selected),
        )
        if not root_errored and terminal is None:
            expected_terminal = _EVIDENCE_SUFFICIENCY if selected else _ANSWER_READINESS
            return Feedback(
                name="decision_action_consistency",
                value="no",
                rationale=(
                    "The action claim and TOOL spans match, but the trace is "
                    f"missing its terminal {expected_terminal!r} decision."
                ),
                source=source,
            )
        if (
            selected
            and terminal is not None
            and (
                terminal_error := _terminal_execution_structure_error(
                    terminal,
                    observed_tool_spans[-1],
                )
            )
        ):
            return Feedback(
                name="decision_action_consistency",
                value="no",
                rationale=terminal_error,
                source=source,
            )
        return Feedback(
            name="decision_action_consistency",
            value="yes",
            rationale=(
                "The ordered tool selections match the authoritative TOOL spans."
            ),
            source=source,
        )

    return score


def decision_tool_appropriateness_scorer() -> Callable[..., Any]:
    """Compare claimed tool choices with reviewed ``expected_tool_calls``.

    This scorer intentionally checks names and multiplicity only. Exact
    arguments and observed execution remain the responsibility of
    ``tool_call_correctness`` and the TOOL spans.
    """

    from mlflow.entities import (
        AssessmentSource,
        AssessmentSourceType,
        Feedback,
    )
    from mlflow.genai.scorers import scorer

    source = AssessmentSource(
        source_type=AssessmentSourceType.CODE,
        source_id="aai-decision-tool-appropriateness",
    )

    @scorer(name="decision_tool_appropriateness")
    def score(
        trace: Any,
        expectations: Mapping[str, Any] | None,
    ) -> Any:
        expected = (expectations or {}).get("expected_tool_calls")
        if not isinstance(expected, list):
            return Feedback(
                name="decision_tool_appropriateness",
                error=ValueError("expected_tool_calls must be a list"),
                source=source,
            )

        try:
            decision_spans = _decision_spans(trace)
            selected = _selected_tool_actions(_tool_selection_spans(decision_spans))
            expected_names = _expected_tool_names(expected)
        except (TypeError, ValueError) as error:
            return Feedback(
                name="decision_tool_appropriateness",
                error=error,
                source=source,
            )

        if not decision_spans:
            return Feedback(
                name="decision_tool_appropriateness",
                value="no",
                rationale="No decision span was present in the agent trace.",
                source=source,
            )
        if Counter(selected) != Counter(expected_names):
            return Feedback(
                name="decision_tool_appropriateness",
                value="no",
                rationale=(
                    f"Reviewed cases expected tool choices {expected_names!r}, "
                    f"but decision spans selected {selected!r}."
                ),
                source=source,
            )
        if not expected_names and not _has_answer_readiness(decision_spans):
            return Feedback(
                name="decision_tool_appropriateness",
                value="no",
                rationale=(
                    "No tool was expected, but the trace has no answer-readiness "
                    "decision."
                ),
                source=source,
            )
        return Feedback(
            name="decision_tool_appropriateness",
            value="yes",
            rationale=(
                "The selected tool names and multiplicity match the reviewed "
                "expectation."
            ),
            source=source,
        )

    return score


def _expected_counts(calls: list[Any]) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for call in calls:
        if not isinstance(call, Mapping):
            raise TypeError("expected_tool_calls entries must be objects")
        counts[_signature(call.get("name"), call.get("arguments"))] += 1
    return counts


def _decision_spans(trace: Any) -> list[Any]:
    spans = _ordered_spans(trace.search_spans(span_type="AGENT"))
    decisions = [
        span
        for span in spans
        if isinstance(getattr(span, "name", None), str)
        and span.name.startswith("decision.")
    ]
    return decisions


def _root_spans(trace: Any) -> list[Any]:
    data = getattr(trace, "data", None)
    spans = getattr(data, "spans", None)
    if spans is None:
        spans = trace.search_spans(span_type="AGENT")
    roots = [span for span in spans if getattr(span, "parent_id", None) is None]
    if not roots:
        raise ValueError("trace_execution_success requires a root span")
    return _ordered_spans(roots)


def _root_reports_error(trace: Any) -> bool:
    """Return true only for an explicit authoritative root ERROR status."""

    try:
        roots = _root_spans(trace)
    except (TypeError, ValueError):
        return False
    for root in roots:
        status = getattr(root, "status", None)
        code = getattr(status, "status_code", status)
        value = getattr(code, "value", code)
        if isinstance(value, str) and value.upper() == "ERROR":
            return True
    return False


def _status_code(span: Any) -> str:
    status = getattr(span, "status", None)
    code = getattr(status, "status_code", status)
    value = getattr(code, "value", code)
    if not isinstance(value, str):
        raise TypeError(f"span {_span_label(span)!r} has no status code")
    normalized = value.upper()
    if normalized not in {"ERROR", "OK", "UNSET"}:
        raise ValueError(
            f"span {_span_label(span)!r} has unknown status {normalized!r}"
        )
    return normalized


def _span_label(span: Any) -> str:
    name = getattr(span, "name", None)
    return name if isinstance(name, str) and name else "<unnamed>"


def _tool_selection_spans(spans: list[Any]) -> list[Any]:
    return [
        span
        for span in spans
        if _span_attribute(span, _DECISION_TYPE) == _TOOL_SELECTION
    ]


def _selected_tool_actions(spans: list[Any]) -> list[str]:
    actions: list[str] = []
    for span in spans:
        action = _span_attribute(span, _SELECTED_ACTION)
        if not isinstance(action, str) or not action:
            raise ValueError("tool-selection decision spans require a selected action")
        actions.append(action)
    return actions


def _decision_execution_structure_error(
    decisions: list[Any],
    tools: list[Any],
) -> str | None:
    for index, (decision, tool) in enumerate(zip(decisions, tools, strict=True)):
        decision_start = getattr(decision, "start_time_ns", None)
        decision_end = getattr(decision, "end_time_ns", None)
        tool_start = getattr(tool, "start_time_ns", None)
        if (
            isinstance(decision_end, int)
            and isinstance(tool_start, int)
            and decision_end > tool_start
        ):
            return (
                f"Tool selection {index} ended after its TOOL span began; "
                "decision spans must close before observed execution."
            )
        if (
            not isinstance(decision_end, int)
            and isinstance(decision_start, int)
            and isinstance(tool_start, int)
            and decision_start > tool_start
        ):
            return (
                f"Tool selection {index} was recorded after its TOOL span began; "
                "the decision cannot be treated as a preceding action claim."
            )
        if (
            hasattr(decision, "parent_id")
            and hasattr(tool, "parent_id")
            and decision.parent_id != tool.parent_id
        ):
            return (
                f"Tool selection {index} and its TOOL span have different "
                "trace parents."
            )
    return None


def _has_answer_readiness(spans: list[Any]) -> bool:
    return any(
        _span_attribute(span, _DECISION_TYPE) == _ANSWER_READINESS
        and _span_attribute(span, _SELECTED_ACTION) == "answer"
        for span in spans
    )


def _terminal_decision(
    spans: list[Any],
    *,
    tool_evidence_exists: bool,
) -> Any | None:
    terminal_type = _EVIDENCE_SUFFICIENCY if tool_evidence_exists else _ANSWER_READINESS
    last_selection = max(
        (
            index
            for index, span in enumerate(spans)
            if _span_attribute(span, _DECISION_TYPE) == _TOOL_SELECTION
        ),
        default=-1,
    )
    return next(
        (
            span
            for index, span in enumerate(spans)
            if index > last_selection
            and _span_attribute(span, _DECISION_TYPE) == terminal_type
            and _span_attribute(span, _SELECTED_ACTION) == "answer"
        ),
        None,
    )


def _terminal_execution_structure_error(terminal: Any, final_tool: Any) -> str | None:
    terminal_start = getattr(terminal, "start_time_ns", None)
    tool_end = getattr(final_tool, "end_time_ns", None)
    if (
        isinstance(terminal_start, int)
        and isinstance(tool_end, int)
        and terminal_start < tool_end
    ):
        return (
            "The evidence-sufficiency decision began before the final TOOL span "
            "ended; convergence must follow observed tool evidence."
        )
    if (
        hasattr(terminal, "parent_id")
        and hasattr(final_tool, "parent_id")
        and terminal.parent_id != final_tool.parent_id
    ):
        return (
            "The evidence-sufficiency decision and final TOOL span have different "
            "trace parents."
        )
    return None


def _tool_span_names(spans: list[Any]) -> list[str]:
    return [_canonical_tool_name(span) for span in spans]


def _ordered_spans(spans: Any) -> list[Any]:
    ordered = spans if isinstance(spans, list) else list(spans)
    if ordered and all(
        isinstance(getattr(span, "start_time_ns", None), int) for span in ordered
    ):
        return sorted(ordered, key=lambda span: span.start_time_ns)
    return list(ordered)


def _expected_tool_names(calls: list[Any]) -> list[str]:
    names: list[str] = []
    for call in calls:
        if not isinstance(call, Mapping):
            raise TypeError("expected_tool_calls entries must be objects")
        name = call.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("expected tool-call names must be non-empty strings")
        names.append(name)
    return names


def _span_attribute(span: Any, key: str) -> Any:
    get_attribute = getattr(span, "get_attribute", None)
    if callable(get_attribute):
        return get_attribute(key)
    attributes = getattr(span, "attributes", None)
    return attributes.get(key) if isinstance(attributes, Mapping) else None


def _actual_counts(spans: list[Any]) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for span in spans:
        signature = _signature(
            _canonical_tool_name(span),
            _canonical_tool_arguments(span),
        )
        counts[signature] += 1
    return counts


def _canonical_tool_name(span: Any) -> str:
    attribute_name = _span_attribute(span, _OTEL_TOOL_NAME)
    if attribute_name is not None:
        if not isinstance(attribute_name, str):
            raise TypeError(f"{_OTEL_TOOL_NAME} must be a string")
        return _require_tool_name(attribute_name)

    inputs = getattr(span, "inputs", None)
    if isinstance(inputs, Mapping):
        call = inputs.get("call")
        if isinstance(call, Mapping) and "tool_name" in call:
            return _require_tool_name(call.get("tool_name"))
    return _require_tool_name(getattr(span, "name", None))


def _canonical_tool_arguments(span: Any) -> Mapping[str, Any]:
    attribute_arguments = _span_attribute(span, _OTEL_TOOL_ARGUMENTS)
    if attribute_arguments is not None:
        return _arguments_mapping(attribute_arguments, source=_OTEL_TOOL_ARGUMENTS)

    inputs = getattr(span, "inputs", None)
    if not isinstance(inputs, Mapping):
        raise TypeError(
            "TOOL spans require object arguments in inputs or "
            f"{_OTEL_TOOL_ARGUMENTS}"
        )
    call = inputs.get("call")
    if isinstance(call, Mapping) and "tool_name" in call:
        if "arguments" not in call:
            raise ValueError("native MLflow TOOL call inputs are missing arguments")
        return _arguments_mapping(
            call.get("arguments"),
            source="native MLflow TOOL call inputs",
        )
    if "arguments" in inputs and set(inputs).issubset({"name", "arguments"}):
        return _arguments_mapping(
            inputs.get("arguments"),
            source="native MLflow TOOL inputs",
        )
    return dict(inputs)


def _arguments_mapping(value: Any, *, source: str) -> Mapping[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(f"{source} must contain valid JSON") from error
    if not isinstance(value, Mapping):
        raise TypeError(f"{source} must contain an object")
    return dict(value)


def _require_tool_name(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("TOOL spans require non-empty names")
    return value


def _signature(name: Any, arguments: Any) -> tuple[str, str]:
    if not isinstance(name, str) or not name:
        raise ValueError("tool-call names must be non-empty strings")
    if not isinstance(arguments, Mapping):
        raise TypeError("tool-call arguments must be objects")
    canonical = json.dumps(
        dict(arguments),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return name, canonical
