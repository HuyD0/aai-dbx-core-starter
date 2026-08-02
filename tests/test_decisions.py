"""Unit tests for the persisted decision vocabulary and evidence run."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from aai_core.decisions import Decision, DecisionRecord, record_decision
from aai_core.evaluation import (
    GateFailure,
    GatePolicy,
    GateResult,
    MetricDirection,
    MetricRule,
)
from aai_core.experiments import ExperimentManager
from aai_core.testing import dev_settings

CITATION_POLICY = GatePolicy(
    rules=(
        MetricRule(
            metric="citation_rate",
            direction=MetricDirection.HIGHER,
            required=1.0,
        ),
    )
)


class FakeMlflow:
    def __init__(self):
        self.params: dict = {}
        self.metrics: dict = {}
        self.tags: dict = {}
        self.artifacts: list = []

    def set_experiment(self, name):
        self.experiment = name

    def start_run(self, run_name=None, nested=False, description=None):
        class _Run:
            def __enter__(self):
                return SimpleNamespace(
                    info=SimpleNamespace(run_id="run-decision-1", run_name=run_name)
                )

            def __exit__(self, *args):
                return False

        return _Run()

    def set_tags(self, tags):
        self.tags.update(tags)

    def log_params(self, params):
        self.params.update(params)

    def log_metrics(self, metrics):
        self.metrics.update(metrics)

    def log_artifact(self, path, artifact_path=None):
        self.artifacts.append((Path(path).name, artifact_path))


def _record(**overrides):
    values = {
        "decision": Decision.ADOPT,
        "change_id": "prompt-v2",
        "change_summary": "Require one exact source citation.",
        "rationale": "Citation rate reached 1.0 with no quality regression.",
        "baseline_run_id": "run-baseline",
        "change_run_id": "run-change",
        "gate": GateResult(metrics={"citation_rate": 1.0}, policy=CITATION_POLICY),
    }
    values.update(overrides)
    return DecisionRecord(**values)


def test_decision_record_is_a_strict_frozen_serializable_contract():
    record = _record(
        prompt_name="main.app.earnings_summary",
        prompt_version=2,
        prompt_digest="a" * 64,
        release_digest="b" * 64,
        decided_by="group:app-owners",
    )

    with pytest.raises(ValidationError):
        DecisionRecord(**{**record.model_dump(), "verdict": "extra"})
    with pytest.raises(ValidationError):
        record.rationale = "rewritten"
    assert record.model_dump(mode="json") == {
        "decision": "adopt",
        "change_id": "prompt-v2",
        "change_summary": "Require one exact source citation.",
        "rationale": "Citation rate reached 1.0 with no quality regression.",
        "baseline_run_id": "run-baseline",
        "change_run_id": "run-change",
        "gate": {
            "metrics": {"citation_rate": 1.0},
            "failures": [],
            "policy": CITATION_POLICY.model_dump(mode="json"),
            "baseline_metrics": None,
        },
        "prompt_name": "main.app.earnings_summary",
        "prompt_version": 2,
        "prompt_digest": "a" * 64,
        "release_digest": "b" * 64,
        "decided_by": "group:app-owners",
        "schema_version": "1",
    }
    assert record.as_tags()["prompt_digest"] == "a" * 64
    assert record.as_tags()["prompt_name"] == "main.app.earnings_summary"
    assert record.as_tags()["prompt_version"] == "2"


def test_run_ids_accept_only_bounded_opaque_identifiers():
    # Free text and secrets must never reach governed tags through the
    # run-id fields.
    for field in ("baseline_run_id", "change_run_id"):
        for bad in (
            "Summarize {{excerpt}} politely.",
            "user@example.com",
            "run id with spaces",
            "a" * 65,
            "",
        ):
            with pytest.raises(ValidationError):
                _record(**{field: bad})
        assert getattr(_record(**{field: "0123456789abcdef" * 2}), field)


def test_prompt_name_accepts_only_a_qualified_registry_name():
    # Typos, prompt text, and credential-like values must never enter the
    # governed aai.prompt_name tag.
    for bad in (
        "earnings_summary",
        "main.app",
        "a.b.c.d",
        "Summarize {{excerpt}} politely.",
        "main.app.name with spaces",
        # Placeholder components pass the character class but are still
        # unusable evidence.
        "unset.app.prompt",
        "replace-with-catalog.app.prompt",
    ):
        with pytest.raises(ValidationError):
            _record(prompt_name=bad)
    assert _record(prompt_name="main.app.earnings_summary").prompt_name


def test_digest_fields_accept_only_a_sha256_hexdigest():
    # Raw prompt text, user content, or secrets must never reach the
    # persisted tags through these fields; both known digests are sha256
    # hexdigests (prompt_digest() and ApplicationRelease.digest).
    for field in ("prompt_digest", "release_digest"):
        for bad in ("Summarize {{excerpt}} politely.", "A" * 64, "deadbeef", ""):
            with pytest.raises(ValidationError):
                _record(**{field: bad})
        assert getattr(_record(**{field: "0123456789abcdef" * 4}), field)


def test_decision_parses_the_documented_string_vocabulary():
    assert _record(decision="  Reject ").decision is Decision.REJECT
    with pytest.raises(ValidationError):
        _record(decision="ship_it")
    with pytest.raises(ValidationError):
        _record(decision="keep_baseline")


def test_adopt_requires_passing_gate_evidence():
    failing = GateResult(
        metrics={"citation_rate": 0.4},
        failures=(GateFailure(metric="citation_rate", reason="0.4 below 1"),),
    )

    with pytest.raises(ValidationError, match="failing gate"):
        _record(gate=failing)
    with pytest.raises(ValidationError, match="requires gate evidence"):
        _record(gate=None)
    with pytest.raises(ValidationError, match="recorded metrics"):
        _record(gate=GateResult(metrics={}))
    with pytest.raises(ValidationError, match="applied release policy"):
        _record(gate=GateResult(metrics={"irrelevant": 1.0}))
    with pytest.raises(ValidationError, match="release rule"):
        _record(gate=GateResult(metrics={"irrelevant": 1.0}, policy=GatePolicy()))
    rejected = _record(decision=Decision.REJECT, gate=failing)
    assert rejected.as_tags()["gate_passed"] == "false"
    ungated_reject = _record(decision=Decision.INCONCLUSIVE, gate=None)
    assert "gate_passed" not in ungated_reject.as_tags()


def test_decided_by_refuses_personal_email_identity():
    with pytest.raises(ValidationError, match="non-personal"):
        _record(decided_by="reviewer@example.com")


def test_record_decision_emits_governed_searchable_evidence():
    fake = FakeMlflow()
    manager = ExperimentManager(
        experiment_name="/Shared/test",
        context=dev_settings().resource,
        mlflow_module=fake,
    )

    run_id = record_decision(_record(), experiments=manager)

    assert run_id == "run-decision-1"
    assert fake.tags["aai.decision"] == "adopt"
    assert fake.tags["aai.run_purpose"] == "decision"
    assert fake.tags["aai.change_id"] == "prompt-v2"
    assert fake.tags["aai.baseline_run_id"] == "run-baseline"
    assert fake.tags["aai.change_run_id"] == "run-change"
    assert fake.tags["aai.gate_passed"] == "true"
    assert fake.metrics == {"citation_rate": 1.0}
    assert ("decision.json", "decision") in fake.artifacts
