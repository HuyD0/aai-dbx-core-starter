"""Deterministic code scorers — from the shared enterprise registry.

These are pure functions: same inputs, same score, no network. They run in
pull-request CI (tier 1) and inside the full judge evaluation (tier 2).

They live in `aai_core.agentkit.catalog`, not here, because a scorer is a
versioned platform asset: when two teams report `keyword_coverage/mean` of
0.8 it has to mean the same thing. This module re-exports them so project
code and tests have a stable local name. Run `agentkit scorers ls` to see
the whole registry.
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
