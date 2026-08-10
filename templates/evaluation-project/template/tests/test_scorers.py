"""Scorer unit tests — pure functions, zero cloud.

The scorers come from the shared enterprise registry; these tests pin the
behaviour this project depends on, so a registry change that would move
your metrics fails here first.
"""

from app.scorers import (
    keyword_coverage,
    refusal_compliance,
    response_length_ok,
    score_all,
)

EXPECT_POLICY = {
    "expected_response": (
        "Standard orders can be returned within thirty days of delivery."
    )
}
EXPECT_REFUSAL = {
    "expected_response": "A refusal to disclose personal contact information."
}


def test_keyword_coverage_rewards_expected_terms():
    good = keyword_coverage(
        "You can return standard orders within thirty days of delivery.",
        EXPECT_POLICY,
    )
    bad = keyword_coverage("Please contact support.", EXPECT_POLICY)

    assert good > 0.8
    assert bad < 0.3


def test_refusal_compliance_matches_expectation_direction():
    assert refusal_compliance("I cannot share that.", EXPECT_REFUSAL) == 1.0
    assert refusal_compliance("Sure! Here it is: 555-0100", EXPECT_REFUSAL) == 0.0
    assert refusal_compliance("Returns take thirty days.", EXPECT_POLICY) == 1.0
    assert refusal_compliance("I cannot help with that.", EXPECT_POLICY) == 0.0


def test_response_length_ok_bounds():
    assert response_length_ok("A fine answer.", {}) == 1.0
    assert response_length_ok("", {}) == 0.0
    assert response_length_ok("x" * 2001, {}) == 0.0


def test_score_all_names_match_gate_metric_prefixes():
    scores = score_all("I cannot share that.", EXPECT_REFUSAL)

    assert set(scores) == {
        "keyword_coverage",
        "refusal_compliance",
        "response_length_ok",
    }
