"""Feedback stays native MLflow evidence and curation fails closed."""

from types import SimpleNamespace

import pytest

from app.feedback import (
    curate_reviewed_expectation,
    record_business_outcome,
    record_human_feedback,
    record_intervention,
)


class FakeMlflow:
    def __init__(self):
        self.calls = []

    def log_feedback(self, **values):
        assessment = SimpleNamespace(kind="feedback", values=values)
        self.calls.append(assessment)
        return assessment

    def log_expectation(self, **values):
        assessment = SimpleNamespace(kind="expectation", values=values)
        self.calls.append(assessment)
        return assessment


def test_human_feedback_returns_native_assessment_and_source():
    native = FakeMlflow()

    assessment = record_human_feedback(
        trace_id="tr-123",
        span_id="sp-456",
        name="user_helpful",
        value=False,
        source_id="support-reviewers",
        rationale="Answer ignored the order id",
        metadata={"channel": "support"},
        mlflow_module=native,
    )

    assert assessment is native.calls[0]
    assert assessment.kind == "feedback"
    assert assessment.values["source"].source_type == "HUMAN"
    assert assessment.values["source"].source_id == "support-reviewers"
    assert assessment.values["trace_id"] == "tr-123"
    assert assessment.values["span_id"] == "sp-456"


def test_reviewed_expectation_is_attested_and_remains_native():
    native = FakeMlflow()

    assessment = curate_reviewed_expectation(
        trace_id="tr-123",
        name="expected_response",
        value="Order A-1001 has shipped.",
        reviewer_group="quality-reviewers",
        reviewed=True,
        metadata={"failure_mode": "stale_status"},
        mlflow_module=native,
    )

    assert assessment is native.calls[0]
    assert assessment.kind == "expectation"
    assert assessment.values["source"].source_type == "HUMAN"
    assert assessment.values["source"].source_id == "quality-reviewers"
    assert assessment.values["metadata"] == {
        "failure_mode": "stale_status",
        "aai.reviewed": "true",
        "aai.reviewer_group": "quality-reviewers",
    }


@pytest.mark.parametrize("reviewed", [False, 0, None])
def test_unreviewed_expectations_never_enter_curation(reviewed):
    native = FakeMlflow()

    with pytest.raises(ValueError, match="explicitly reviewed"):
        curate_reviewed_expectation(
            trace_id="tr-123",
            name="expected_response",
            value="answer",
            reviewer_group="quality-reviewers",
            reviewed=reviewed,
            mlflow_module=native,
        )

    assert native.calls == []


def test_caller_cannot_spoof_controlled_review_metadata():
    with pytest.raises(ValueError, match="controlled"):
        curate_reviewed_expectation(
            trace_id="tr-123",
            name="expected_response",
            value="answer",
            reviewer_group="quality-reviewers",
            reviewed=True,
            metadata={"aai.reviewed": "false"},
            mlflow_module=FakeMlflow(),
        )


@pytest.mark.parametrize(
    "source_id",
    [
        "person@example.com",
        "sk-secret",
        "bearer x",
        "github_pat_secret",
        "ghp_secret",
        "gho_secret",
        "ghr_secret",
        "ghs_secret",
        "ghu_secret",
    ],
)
def test_assessment_sources_reject_personal_or_secret_shaped_ids(source_id):
    with pytest.raises(ValueError, match="non-personal") as error:
        record_human_feedback(
            trace_id="tr-123",
            name="user_helpful",
            value=True,
            source_id=source_id,
            mlflow_module=FakeMlflow(),
        )

    assert source_id not in str(error.value)


def test_interventions_and_business_outcomes_require_review_before_learning():
    native = FakeMlflow()

    intervention = record_intervention(
        trace_id="tr-123",
        value={"action": "human_approved"},
        source_id="operations-reviewers",
        mlflow_module=native,
    )
    outcome = record_business_outcome(
        trace_id="tr-123",
        value={"resolved": True},
        source_id="order-system-v2",
        source_type="code",
        mlflow_module=native,
    )

    assert intervention.values["name"] == "aai.intervention"
    assert intervention.values["source"].source_type == "HUMAN"
    assert outcome.values["name"] == "aai.business_outcome"
    assert outcome.values["source"].source_type == "CODE"
    assert intervention.values["metadata"] == {
        "aai.learning_eligibility": "review_required"
    }
    assert outcome.values["metadata"] == {"aai.learning_eligibility": "review_required"}
