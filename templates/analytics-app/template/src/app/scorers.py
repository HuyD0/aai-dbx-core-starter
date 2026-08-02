"""Deterministic code scorers — pure functions, zero cloud, zero LLM.

They grade the provenance discipline the platform requires of every answer,
in both tiers: pull-request CI scores the recorded answer sheet
(evals/offline_checks.py), and the credentialed release gate wraps the same
functions with mlflow's @scorer (evals/evaluate.py). Scorers that need an
expectation the case does not carry return 1.0 (vacuously satisfied) so
means stay comparable across mixed case categories.
"""

from __future__ import annotations

from app.provenance import SourceTier, parse_footer
from app.semantics.executor import WarehouseExecutionError, ensure_read_only

_VALUE_TOLERANCE = 1e-6


def routing_tier_match(outputs: str, expectations: dict) -> float:
    """The answer's primary source tier matches the expected routing."""

    expected = expectations.get("expected_tier")
    if not expected:
        return 1.0
    records = parse_footer(str(outputs))
    if not records:
        return 0.0
    return 1.0 if records[0].tier.value == expected else 0.0


def provenance_complete(outputs: str, expectations: dict) -> float:
    """Every answer carries evidence; semantic claims must show their SQL."""

    records = parse_footer(str(outputs))
    if not records:
        return 0.0
    for record in records:
        if record.tier is SourceTier.SEMANTIC_LAYER and not record.sql:
            return 0.0
    return 1.0


def sql_read_only(outputs: str, expectations: dict) -> float:
    """Any SQL shown in provenance passes the read-only guard."""

    for record in parse_footer(str(outputs)):
        if not record.sql:
            continue
        try:
            ensure_read_only(record.sql)
        except WarehouseExecutionError:
            return 0.0
    return 1.0


def execution_match(outputs: str, expectations: dict) -> float:
    """The recorded numeric result equals the snapshot-pinned expectation."""

    expected = expectations.get("expected_value")
    if expected is None:
        return 1.0
    try:
        expected_number = float(expected)
    except (TypeError, ValueError):
        return 0.0
    for record in parse_footer(str(outputs)):
        if record.value is None:
            continue
        try:
            observed = float(record.value)
        except ValueError:
            continue
        tolerance = max(_VALUE_TOLERANCE, abs(expected_number) * _VALUE_TOLERANCE)
        if abs(observed - expected_number) <= tolerance:
            return 1.0
    return 0.0


def semantic_share(outputs: str, expectations: dict) -> float:
    """Share of answers resolved through the semantic layer (monitor the
    mean; clarifications and sanctioned raw fallbacks legitimately score 0)."""

    records = parse_footer(str(outputs))
    return (
        1.0
        if any(record.tier is SourceTier.SEMANTIC_LAYER for record in records)
        else 0.0
    )


CODE_SCORERS = (
    routing_tier_match,
    provenance_complete,
    sql_read_only,
    execution_match,
    semantic_share,
)


def score_all(outputs: str, expectations: dict) -> dict[str, float]:
    return {fn.__name__: fn(outputs, expectations) for fn in CODE_SCORERS}
