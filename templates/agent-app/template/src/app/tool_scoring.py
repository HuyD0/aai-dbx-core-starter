"""MLflow 3.14-compatible exact tool-trajectory scoring."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from typing import Any


def exact_tool_call_scorer():
    """Return an exact, unordered MLflow tool-call scorer.

    MLflow 3.14's native exact scorer treats an empty expected list as missing
    and its unordered comparison does not fully preserve duplicate-call
    multiplicity. This compatibility layer closes those two release-gate
    gaps, then delegates matching non-empty trajectories to the native scorer.
    """

    from mlflow.entities import (
        AssessmentSource,
        AssessmentSourceType,
        Feedback,
    )
    from mlflow.genai.scorers import ToolCallCorrectness, scorer

    native = ToolCallCorrectness(should_exact_match=True)
    source = AssessmentSource(
        source_type=AssessmentSourceType.CODE,
        source_id="aai-exact-tool-calls",
    )

    @scorer(name="tool_call_correctness")
    def score(trace, expectations):
        expected = (expectations or {}).get("expected_tool_calls")
        if not isinstance(expected, list):
            return Feedback(
                name="tool_call_correctness",
                error=ValueError("expected_tool_calls must be a list"),
                source=source,
            )

        try:
            expected_counts = _expected_counts(expected)
            actual_spans = trace.search_spans(span_type="TOOL")
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
        return native(trace=trace, expectations=expectations)

    return score


def _expected_counts(calls: list[Any]) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for call in calls:
        if not isinstance(call, Mapping):
            raise TypeError("expected_tool_calls entries must be objects")
        counts[_signature(call.get("name"), call.get("arguments"))] += 1
    return counts


def _actual_counts(spans: list[Any]) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for span in spans:
        signature = _signature(
            getattr(span, "name", None),
            getattr(span, "inputs", None),
        )
        counts[signature] += 1
    return counts


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
