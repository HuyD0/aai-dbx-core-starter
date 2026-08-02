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
    # A failed prediction arrives as None and must not score as "None".
    assert response_length_ok(None, {}) == 0.0


def test_score_all_names_match_gate_metric_prefixes():
    scores = score_all("I cannot share that.", EXPECT_REFUSAL)

    assert set(scores) == {
        "keyword_coverage",
        "refusal_compliance",
        "response_length_ok",
    }


def test_as_mlflow_scorers_wraps_self_contained_bodies_under_stable_names(
    monkeypatch,
):
    registered = []

    def fake_scorer(*, name):
        def decorate(fn):
            registered.append((name, fn.__name__))
            return {"name": name, "fn": fn}

        return decorate

    install_fake_module(monkeypatch, "mlflow.genai.scorers", scorer=fake_scorer)

    wrapped = as_mlflow_scorers()

    # Metric names stay stable while the decorated function is the
    # registered_* sibling whose body survives body-only serialization.
    assert registered == [
        ("keyword_coverage", "registered_keyword_coverage"),
        ("refusal_compliance", "registered_refusal_compliance"),
        ("response_length_ok", "registered_response_length_ok"),
    ]
    refusal = wrapped[1]["fn"]
    assert refusal("I cannot share that.", EXPECT_REFUSAL) == 1.0
    assert refusal("Sure! Here it is.", EXPECT_REFUSAL) == 0.0
    # None expectations must normalize to an empty mapping, not crash.
    assert refusal("Sure! Here it is.", None) == 1.0


def test_registered_bodies_survive_dependency_free_reconstruction():
    # MLflow rebuilds a registered scorer by exec()ing the extracted function
    # source in a managed service without module globals, closures, or
    # aai-core installed — an empty namespace is the faithful simulation.
    import inspect

    from aai_core.scorers import _REGISTERED_BODIES

    for registered in _REGISTERED_BODIES.values():
        namespace: dict = {}
        exec(inspect.getsource(registered), {}, namespace)
        rebuilt = namespace[registered.__name__]
        assert rebuilt("I cannot share that.", EXPECT_REFUSAL) in (0.0, 1.0)
        assert rebuilt("Sure! Here it is.", None) in (0.0, 1.0)


def test_registered_bodies_stay_equivalent_to_the_pure_scorers():
    from aai_core.scorers import _REGISTERED_BODIES

    cases = [
        ("You can return standard orders within thirty days.", EXPECT_POLICY),
        ("Please contact support.", EXPECT_POLICY),
        ("I cannot share that.", EXPECT_REFUSAL),
        ("Sure! Here it is: 555-0100", EXPECT_REFUSAL),
        ("", EXPECT_POLICY),
        ("x" * 2001, {}),
        ("A fine answer.", {}),
        (None, {}),
    ]
    for pure, registered in _REGISTERED_BODIES.items():
        for outputs, expectations in cases:
            assert registered(outputs, expectations) == pure(
                outputs, dict(expectations)
            ), f"{registered.__name__} drifted from {pure.__name__}"


def test_as_mlflow_scorers_rejects_closure_dependent_custom_functions(
    monkeypatch,
):
    install_fake_module(
        monkeypatch,
        "mlflow.genai.scorers",
        scorer=lambda *, name: (lambda fn: fn),
    )

    def custom_scorer(outputs, expectations):
        return 1.0

    with pytest.raises(ValueError, match="self-contained"):
        as_mlflow_scorers([custom_scorer])


def test_as_mlflow_scorers_requires_the_genai_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlflow.genai.scorers", None)

    with pytest.raises(RuntimeError, match="genai"):
        as_mlflow_scorers()
