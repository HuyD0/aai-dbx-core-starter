"""Unit tests for the shared deterministic code scorers."""

import sys

import pytest
from conftest import install_fake_module

from aai_core.scorers import (
    as_mlflow_scorers,
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


def test_as_mlflow_scorers_wraps_each_function_under_its_own_name(monkeypatch):
    registered = []

    def fake_scorer(*, name):
        def decorate(fn):
            registered.append(name)
            return {"name": name, "fn": fn}

        return decorate

    install_fake_module(monkeypatch, "mlflow.genai.scorers", scorer=fake_scorer)

    wrapped = as_mlflow_scorers()

    assert registered == [
        "keyword_coverage",
        "refusal_compliance",
        "response_length_ok",
    ]
    refusal = wrapped[1]["fn"]
    assert refusal("I cannot share that.", EXPECT_REFUSAL) == 1.0
    assert refusal("Sure! Here it is.", EXPECT_REFUSAL) == 0.0
    # None expectations must normalize to an empty mapping, not crash.
    assert refusal("Sure! Here it is.", None) == 1.0


def test_as_mlflow_scorers_requires_the_genai_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlflow.genai.scorers", None)

    with pytest.raises(RuntimeError, match="genai"):
        as_mlflow_scorers()
