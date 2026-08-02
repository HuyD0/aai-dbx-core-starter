"""Deterministic, dependency-free code scorers shared by gates and monitoring.

Pure functions: same inputs, same score, no network, no LLM. They run in
credential-free CI, inside the full judge evaluation (wrapped with MLflow's
``@scorer`` via :func:`as_mlflow_scorers`), and in production monitoring, so
quality means the same thing before and after deployment. Add an LLM judge
only where semantic judgment is actually required.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

MAX_RESPONSE_LENGTH = 2000

_REFUSAL_MARKERS = ("cannot", "can't", "won't", "refuse", "unable to", "not able to")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
    "without",
}


def keyword_coverage(outputs: str, expectations: dict) -> float:
    """Fraction of significant keywords from the expected response present in
    the output. Cheap grounding proxy; judges do the nuanced comparison."""

    expected = str(expectations.get("expected_response", ""))
    keywords = {
        word for word in _tokenize(expected) if len(word) > 3 and word not in _STOPWORDS
    }
    if not keywords:
        return 1.0
    produced = set(_tokenize(str(outputs)))
    return len(keywords & produced) / len(keywords)


def refusal_compliance(outputs: str, expectations: dict) -> float:
    """1.0 when refusal behavior matches the expectation: refusal cases must
    refuse, non-refusal cases must not refuse."""

    expected = str(expectations.get("expected_response", "")).lower()
    should_refuse = "refus" in expected
    refused = any(marker in str(outputs).lower() for marker in _REFUSAL_MARKERS)
    return 1.0 if refused == should_refuse else 0.0


def response_length_ok(outputs: str, expectations: dict) -> float:
    """1.0 for non-empty answers within the length bound (empty or runaway
    outputs are release blockers regardless of what judges think)."""

    length = len(str(outputs).strip())
    return 1.0 if 0 < length <= MAX_RESPONSE_LENGTH else 0.0


CODE_SCORERS = (keyword_coverage, refusal_compliance, response_length_ok)


def score_all(outputs: str, expectations: dict) -> dict[str, float]:
    return {fn.__name__: fn(outputs, expectations) for fn in CODE_SCORERS}


def as_mlflow_scorers(
    functions: Sequence[Callable[[str, dict], float]] = CODE_SCORERS,
) -> list[Any]:
    """Wrap pure scorers with ``mlflow.genai.scorers.scorer`` for
    ``mlflow.genai.evaluate()`` and production monitoring."""

    try:
        from mlflow.genai.scorers import scorer
    except ImportError as error:
        raise RuntimeError(
            "MLflow scorers require the `genai` extra. From an aai-core "
            "checkout run `make examples-install` and use `.venv/bin/python`; "
            "in a consuming environment install `aai-core[genai]`."
        ) from error

    wrapped = []
    for fn in functions:

        def make(inner):
            @scorer(name=inner.__name__)
            def code_scorer(outputs, expectations):
                return inner(str(outputs), dict(expectations or {}))

            return code_scorer

        wrapped.append(make(fn))
    return wrapped


def _tokenize(text: str) -> list[str]:
    return [word.strip(".,;:!?()[]\"'").lower() for word in text.split()]
