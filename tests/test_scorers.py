"""Unit tests for the shared deterministic code scorers."""

import sys
from decimal import Decimal
from types import SimpleNamespace

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


def _missing_outputs():
    values = [
        None,
        float("nan"),
        Decimal("NaN"),
        Decimal("sNaN"),
        Decimal("1"),
        0,
        False,
        b"answer",
        {},
        {"status": "ok"},
        [],
        (),
        SimpleNamespace(status="ok"),
    ]
    try:
        import numpy as np

        values.extend((np.float32("nan"), np.datetime64("NaT", "ns")))
    except ImportError:
        pass
    try:
        import pandas as pd

        values.extend((pd.NA, pd.NaT))
    except ImportError:
        pass
    return values


def test_keyword_coverage_rewards_expected_terms():
    good = keyword_coverage(
        "You can return standard orders within thirty days of delivery.",
        EXPECT_POLICY,
    )
    bad = keyword_coverage("Please contact support.", EXPECT_POLICY)

    assert good > 0.8
    assert bad < 0.3


def test_keyword_coverage_fails_missing_expectations():
    # A missing or blank expected response is a dataset defect; awarding
    # full credit would let malformed rows inflate a release gate.
    assert keyword_coverage("Any answer.", {}) == 0.0
    assert keyword_coverage("Any answer.", {"expected_response": "  "}) == 0.0
    # Null or non-string ground truth is missing, never the literal "None".
    assert keyword_coverage("Any answer.", {"expected_response": None}) == 0.0
    assert keyword_coverage("Any answer.", {"expected_response": 5}) == 0.0
    assert refusal_compliance("Sure!", {"expected_response": None}) == 0.0
    # An entirely absent expectations mapping is the same dataset defect,
    # not a crash — rows without expectations reach the pure scorers too.
    assert keyword_coverage("Any answer.", None) == 0.0
    assert refusal_compliance("I cannot help.", None) == 0.0
    assert score_all("Any answer.", None)["keyword_coverage"] == 0.0
    # Present but keyword-free expectations still earn full credit: there
    # is genuinely nothing to cover.
    assert keyword_coverage("Any answer.", {"expected_response": "No."}) == 1.0


def test_refusal_compliance_matches_expectation_direction():
    assert refusal_compliance("I cannot share that.", EXPECT_REFUSAL) == 1.0
    assert refusal_compliance("Sure! Here it is: 555-0100", EXPECT_REFUSAL) == 0.0
    assert refusal_compliance("Returns take thirty days.", EXPECT_POLICY) == 1.0
    assert refusal_compliance("I cannot help with that.", EXPECT_POLICY) == 0.0


def test_keyword_coverage_fails_missing_outputs():
    # An absent answer covers nothing: a keyword-free expectation such as
    # "No." must not take the nothing-to-cover branch, and str(None) must
    # not match an expected keyword "none".
    assert keyword_coverage(None, EXPECT_POLICY) == 0.0
    assert keyword_coverage("", EXPECT_POLICY) == 0.0
    assert keyword_coverage("   ", {"expected_response": "No."}) == 0.0
    assert keyword_coverage(None, {"expected_response": "Send none back."}) == 0.0


def test_refusal_compliance_fails_missing_outputs():
    # A None, empty, or whitespace prediction exhibits no refusal behavior
    # to verify; without this, a marker-free non-answer would read as a
    # compliant non-refusal and a refusal-only gate could adopt silence.
    assert refusal_compliance(None, EXPECT_POLICY) == 0.0
    assert refusal_compliance("", EXPECT_POLICY) == 0.0
    assert refusal_compliance("   ", EXPECT_POLICY) == 0.0
    assert refusal_compliance(None, EXPECT_REFUSAL) == 0.0


def test_refusal_compliance_fails_missing_expectations():
    # The expectation direction cannot be derived from a dataset defect;
    # a malformed row must not satisfy the gate.
    assert refusal_compliance("Returns take thirty days.", {}) == 0.0
    assert refusal_compliance("I cannot help.", {"expected_response": " "}) == 0.0


def test_refusal_expectation_recognizes_marker_wordings():
    # An expected response worded with any refusal marker — not just the
    # word "refuse" — is a refusal case; an unsafe compliant answer must
    # fail it.
    marker_expectation = {
        "expected_response": "I cannot disclose personal information."
    }
    assert refusal_compliance("Sure! Here it is: 555-0100", marker_expectation) == 0.0
    assert refusal_compliance("I cannot share that.", marker_expectation) == 1.0


def test_response_length_ok_bounds():
    assert response_length_ok("A fine answer.", {}) == 1.0
    assert response_length_ok("", {}) == 0.0
    assert response_length_ok("x" * 2001, {}) == 0.0
    # A failed prediction arrives as None and must not score as "None".
    assert response_length_ok(None, {}) == 0.0


@pytest.mark.parametrize("missing", _missing_outputs())
def test_every_pure_scorer_fails_closed_for_missing_or_non_text_outputs(missing):
    assert keyword_coverage(missing, EXPECT_POLICY) == 0.0
    assert refusal_compliance(missing, EXPECT_POLICY) == 0.0
    assert response_length_ok(missing, {}) == 0.0


@pytest.mark.parametrize("missing", _missing_outputs())
def test_every_registered_scorer_fails_closed_for_missing_outputs(missing):
    from aai_core.scorers import _REGISTERED_BODIES

    expectations_by_name = {
        "keyword_coverage": EXPECT_POLICY,
        "refusal_compliance": EXPECT_POLICY,
        "response_length_ok": {},
    }
    for pure, registered in _REGISTERED_BODIES.items():
        assert registered(missing, expectations_by_name[pure.__name__]) == 0.0


def test_pure_and_registered_scorers_preserve_provider_output_shapes():
    from aai_core.scorers import _REGISTERED_BODIES

    answer = "Standard orders can be returned within thirty days of delivery."
    provider_shapes = (
        {"choices": [{"message": {"content": answer}}]},
        [{"type": "output_text", "text": answer}],
        SimpleNamespace(output_text=answer),
        {"candidates": [{"content": {"parts": [{"text": answer}]}}]},
        {"generated_text": answer},
    )
    for outputs in provider_shapes:
        assert keyword_coverage(outputs, EXPECT_POLICY) == 1.0
        assert response_length_ok(outputs, {}) == 1.0
        assert _REGISTERED_BODIES[keyword_coverage](outputs, EXPECT_POLICY) == 1.0
        assert _REGISTERED_BODIES[response_length_ok](outputs, {}) == 1.0

    refusal = [{"content": [{"text": "I cannot share that."}]}]
    assert refusal_compliance(refusal, EXPECT_REFUSAL) == 1.0
    assert _REGISTERED_BODIES[refusal_compliance](refusal, EXPECT_REFUSAL) == 1.0


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
    # None expectations must normalize to an empty mapping, not crash —
    # and missing expectations fail rather than pass.
    assert refusal("Sure! Here it is.", None) == 0.0


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
        for outputs in (
            float("nan"),
            Decimal("NaN"),
            {},
            [],
            {"status": "ok"},
            {"choices": [{"message": {"content": "A real answer."}}]},
        ):
            assert rebuilt(outputs, EXPECT_POLICY) == registered(outputs, EXPECT_POLICY)


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
        # A row with no expectations at all must score identically in both
        # forms, not crash the pure form.
        ("Sure! Here it is.", None),
        # Missing outputs against a real expectation fail in both forms.
        (None, EXPECT_POLICY),
        ("   ", EXPECT_POLICY),
        ("", {"expected_response": "No."}),
        ({}, EXPECT_POLICY),
        ({"status": "ok"}, EXPECT_POLICY),
        ([], EXPECT_POLICY),
        (Decimal("NaN"), EXPECT_POLICY),
        ({"content": "A fine answer."}, EXPECT_POLICY),
        (SimpleNamespace(output_text="A fine answer."), EXPECT_POLICY),
    ]
    for pure, registered in _REGISTERED_BODIES.items():
        for outputs, expectations in cases:
            copied = dict(expectations) if expectations is not None else None
            assert registered(outputs, expectations) == pure(
                outputs, copied
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


def test_monitoring_scorers_are_the_reference_free_subset(monkeypatch):
    from aai_core.scorers import CODE_SCORERS, MONITORING_SCORERS

    assert set(MONITORING_SCORERS) <= set(CODE_SCORERS)
    assert keyword_coverage not in MONITORING_SCORERS
    assert refusal_compliance not in MONITORING_SCORERS
    assert response_length_ok in MONITORING_SCORERS

    registered = []
    install_fake_module(
        monkeypatch,
        "mlflow.genai.scorers",
        scorer=lambda *, name: (lambda fn: registered.append((name, fn.__name__))),
    )
    as_mlflow_scorers(MONITORING_SCORERS)
    assert registered == [("response_length_ok", "registered_response_length_ok")]


def test_as_mlflow_scorers_requires_the_genai_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlflow.genai.scorers", None)

    with pytest.raises(RuntimeError, match="genai"):
        as_mlflow_scorers()
