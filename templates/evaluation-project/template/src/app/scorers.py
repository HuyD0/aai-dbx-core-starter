"""Deterministic code scorers from the shared enterprise registry.

These are pure functions: same inputs, same score, no network. They run in
pull-request CI (tier 1) and inside the full judge evaluation (tier 2).

Scorers live in :mod:`aai_core.agentkit.catalog` because their name, behavior,
and version are platform contracts. This module re-exports the selected code
scorers so project code and tests retain stable local imports.
"""

from __future__ import annotations

from aai_core.agentkit.catalog import (
    CODE_SCORER_FUNCTIONS,
    keyword_coverage,
    refusal_compliance,
    response_length_ok,
    score_all,
)

CODE_SCORERS = tuple(CODE_SCORER_FUNCTIONS.values())

__all__ = [
    "CODE_SCORERS",
    "keyword_coverage",
    "refusal_compliance",
    "response_length_ok",
    "score_all",
]
