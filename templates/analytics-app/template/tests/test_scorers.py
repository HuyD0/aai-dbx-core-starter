"""Deterministic scorer behavior, including the vacuous-1.0 convention."""

from app.scorers import (
    execution_match,
    provenance_complete,
    routing_tier_match,
    score_all,
    semantic_share,
    sql_read_only,
)

SEMANTIC_ANSWER = (
    "Revenue was 600.75.\n\n"
    "[provenance]\n"
    "tier: semantic_layer\n"
    "sources: demo.sales.analytics_orders\n"
    "owner: group:test-owners\n"
    "rows: 1\n"
    "value: 600.75\n"
    "sql: SELECT SUM(`amount`) AS `revenue` FROM t LIMIT 100\n"
    "[/provenance]"
)
CURATED_ANSWER = (
    "Please clarify the timeframe.\n\n"
    "[provenance]\n"
    "tier: curated_reference\n"
    "sources: knowledge/metrics_definitions.md\n"
    "[/provenance]"
)
NO_FOOTER = "Revenue was 600.75, trust me."


def test_routing_tier_match_grades_the_primary_record():
    expectations = {"expected_tier": "semantic_layer"}
    assert routing_tier_match(SEMANTIC_ANSWER, expectations) == 1.0
    assert routing_tier_match(CURATED_ANSWER, expectations) == 0.0
    assert routing_tier_match(NO_FOOTER, expectations) == 0.0
    assert routing_tier_match(NO_FOOTER, {}) == 1.0


def test_provenance_complete_requires_evidence_and_semantic_sql():
    assert provenance_complete(SEMANTIC_ANSWER, {}) == 1.0
    assert provenance_complete(CURATED_ANSWER, {}) == 1.0
    assert provenance_complete(NO_FOOTER, {}) == 0.0
    semantic_without_sql = SEMANTIC_ANSWER.replace(
        "sql: SELECT SUM(`amount`) AS `revenue` FROM t LIMIT 100\n", ""
    )
    assert provenance_complete(semantic_without_sql, {}) == 0.0


def test_sql_read_only_rejects_writes_in_evidence():
    tampered = SEMANTIC_ANSWER.replace(
        "sql: SELECT SUM(`amount`) AS `revenue` FROM t LIMIT 100",
        "sql: DELETE FROM t",
    )
    assert sql_read_only(SEMANTIC_ANSWER, {}) == 1.0
    assert sql_read_only(tampered, {}) == 0.0
    assert sql_read_only(CURATED_ANSWER, {}) == 1.0


def test_execution_match_compares_within_tolerance():
    assert execution_match(SEMANTIC_ANSWER, {"expected_value": 600.75}) == 1.0
    assert execution_match(SEMANTIC_ANSWER, {"expected_value": 600.76}) == 0.0
    assert execution_match(SEMANTIC_ANSWER, {"expected_value": None}) == 1.0
    assert execution_match(NO_FOOTER, {"expected_value": 600.75}) == 0.0


def test_semantic_share_flags_semantic_layer_usage():
    assert semantic_share(SEMANTIC_ANSWER, {}) == 1.0
    assert semantic_share(CURATED_ANSWER, {}) == 0.0


def test_score_all_reports_every_scorer():
    scores = score_all(SEMANTIC_ANSWER, {"expected_tier": "semantic_layer"})
    assert set(scores) == {
        "routing_tier_match",
        "provenance_complete",
        "sql_read_only",
        "execution_match",
        "semantic_share",
    }
    assert all(value == 1.0 for value in scores.values())
