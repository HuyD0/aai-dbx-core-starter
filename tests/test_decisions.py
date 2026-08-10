"""Unit tests for the persisted decision vocabulary and evidence run."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from aai_core.decisions import (
    Decision,
    DecisionEvidenceError,
    DecisionRecord,
    decision_digest,
    load_decision,
    record_decision,
)
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
        self.run_names: list = []

    def set_experiment(self, name):
        self.experiment = name

    def start_run(self, run_name=None, nested=False, description=None):
        self.run_names.append(run_name)

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


class FakeDecisionClient:
    def __init__(self, owner):
        self.owner = owner

    def get_run(self, run_id):
        self.owner.requested_run_ids.append(run_id)
        if self.owner.get_error is not None:
            raise self.owner.get_error
        return self.owner.run

    def download_artifacts(self, run_id, artifact_path):
        self.owner.downloads.append((run_id, artifact_path))
        if self.owner.download_error is not None:
            raise self.owner.download_error
        return str(self.owner.artifact)


class FakeDecisionMlflow:
    def __init__(self, *, run, artifact, get_error=None, download_error=None):
        self.run = run
        self.artifact = artifact
        self.get_error = get_error
        self.download_error = download_error
        self.requested_run_ids: list[str] = []
        self.downloads: list[tuple[str, str]] = []

    def MlflowClient(self):
        return FakeDecisionClient(self)


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


def _persisted_mlflow(tmp_path, record, *, status="FINISHED", tags=None, metrics=None):
    artifact = tmp_path / "decision.json"
    artifact.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
    expected_tags = {
        "aai.run_purpose": "decision",
        "aai.change_id": record.change_id,
        "aai.change_summary": record.change_summary,
        "aai.decision_digest": decision_digest(record),
        **{f"aai.{name}": value for name, value in record.as_tags().items()},
    }
    if record.baseline_run_id:
        expected_tags["aai.baseline_run_id"] = record.baseline_run_id
    run = SimpleNamespace(
        info=SimpleNamespace(run_id="run-decision-1", status=status),
        data=SimpleNamespace(
            tags=dict(expected_tags if tags is None else tags),
            metrics=dict(
                record.gate.metrics
                if metrics is None and record.gate is not None
                else metrics or {}
            ),
        ),
    )
    return FakeDecisionMlflow(run=run, artifact=artifact)


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


def test_tagged_change_fields_are_bounded():
    # change_id and change_summary become searchable tags: the id is an
    # identifier and the summary is bounded prose.
    with pytest.raises(ValidationError):
        _record(change_id="prompt v2 with spaces")
    with pytest.raises(ValidationError):
        _record(change_id="a" * 65)
    with pytest.raises(ValidationError):
        _record(change_summary="x" * 201)
    assert _record(change_summary="x" * 200).change_summary


def test_free_text_evidence_must_be_substantive():
    # min_length=1 alone accepts "   ": a whitespace summary would tag
    # aai.change_summary blank and a whitespace rationale would persist a
    # decision.json stating no reason at all.
    for blank in ("   ", "\t", "\n"):
        with pytest.raises(ValidationError, match="substantive"):
            _record(change_summary=blank)
        with pytest.raises(ValidationError, match="substantive"):
            _record(rationale=blank)
        with pytest.raises(ValidationError, match="substantive"):
            _record(decided_by=blank)
    # Surrounding whitespace is trimmed so the stored evidence is exactly
    # what a reader sees in the tag and the artifact.
    trimmed = _record(
        change_summary="  Require one exact source citation.  ",
        rationale="\tCitation rate reached 1.0.\n",
        decided_by=" group:app-owners ",
    )
    assert trimmed.change_summary == "Require one exact source citation."
    assert trimmed.rationale == "Citation rate reached 1.0."
    assert trimmed.decided_by == "group:app-owners"


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
    # A zero cost-coverage threshold rejects no coverage value, so it is
    # not a substantive release rule.
    with pytest.raises(ValidationError, match="zero cost-coverage"):
        _record(
            gate=GateResult(
                metrics={"cost/coverage": 0.0},
                policy=GatePolicy(minimum_cost_coverage=0.0),
            )
        )
    rejected = _record(decision=Decision.REJECT, gate=failing)
    assert rejected.as_tags()["gate_passed"] == "false"


def test_adopt_requires_an_applied_constraint():
    # A regression-only policy that waives missing baselines skips its
    # only check when the baseline lacks the metric; such a gate passes
    # while constraining nothing, so it cannot authorize adoption.
    regression_only = GatePolicy(
        rules=(
            MetricRule(
                metric="citation_rate",
                direction=MetricDirection.HIGHER,
                max_regression=0.05,
            ),
        ),
        allow_missing_regression_baseline=True,
    )

    with pytest.raises(ValidationError, match="without their baseline"):
        _record(gate=GateResult(metrics={"citation_rate": 1.0}, policy=regression_only))
    with pytest.raises(ValidationError, match="without their baseline"):
        _record(
            gate=GateResult(
                metrics={"citation_rate": 1.0},
                policy=regression_only,
                baseline_metrics={"unrelated": 0.9},
            )
        )
    # The same rule with its baseline value recorded is genuinely applied.
    compared = _record(
        gate=GateResult(
            metrics={"citation_rate": 1.0},
            policy=regression_only,
            baseline_metrics={"citation_rate": 0.98},
        )
    )
    assert compared.decision is Decision.ADOPT
    # An absolute rule is applied regardless of the waived baseline.
    absolute_too = _record(
        gate=GateResult(
            metrics={"citation_rate": 1.0},
            policy=GatePolicy(
                rules=(
                    MetricRule(
                        metric="citation_rate",
                        direction=MetricDirection.HIGHER,
                        required=1.0,
                        max_regression=0.05,
                    ),
                ),
                allow_missing_regression_baseline=True,
            ),
        )
    )
    assert absolute_too.decision is Decision.ADOPT
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
    assert fake.tags["aai.decision_digest"] == decision_digest(_record())
    assert fake.metrics == {"citation_rate": 1.0}
    assert ("decision.json", "decision") in fake.artifacts


def test_load_decision_verifies_the_finished_run_and_artifact(tmp_path):
    record = _record(
        prompt_name="main.app.earnings_summary",
        prompt_version=2,
        prompt_digest="a" * 64,
    )
    mlflow = _persisted_mlflow(tmp_path, record)

    loaded = load_decision("run-decision-1", mlflow_module=mlflow)

    assert loaded == record
    assert mlflow.requested_run_ids == ["run-decision-1"]
    assert mlflow.downloads == [("run-decision-1", "decision/decision.json")]


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("running", "not finished"),
        ("identity", "different run identity"),
        ("purpose", "tag 'aai.run_purpose' contradicts"),
        ("digest", "tag 'aai.decision_digest' contradicts"),
        ("tag", "tag 'aai.decision' contradicts"),
        ("metric", "metric 'citation_rate' contradicts"),
        ("artifact", "valid decision/decision.json"),
    ],
)
def test_load_decision_rejects_contradictory_persisted_evidence(
    tmp_path, mutation, match
):
    record = _record()
    mlflow = _persisted_mlflow(tmp_path, record)
    if mutation == "running":
        mlflow.run.info.status = "RUNNING"
    elif mutation == "identity":
        mlflow.run.info.run_id = "a-different-run"
    elif mutation == "purpose":
        mlflow.run.data.tags["aai.run_purpose"] = "change"
    elif mutation == "digest":
        mlflow.run.data.tags["aai.decision_digest"] = "0" * 64
    elif mutation == "tag":
        mlflow.run.data.tags["aai.decision"] = "reject"
    elif mutation == "metric":
        mlflow.run.data.metrics["citation_rate"] = 0.5
    else:
        mlflow.artifact.write_text("not json", encoding="utf-8")

    with pytest.raises(DecisionEvidenceError, match=match):
        load_decision("run-decision-1", mlflow_module=mlflow)


@pytest.mark.parametrize(
    "missing",
    [
        FileNotFoundError("missing decision artifact"),
        type(
            "MissingArtifactError",
            (Exception,),
            {"error_code": "RESOURCE_DOES_NOT_EXIST"},
        )("decision artifact does not exist"),
    ],
)
def test_load_decision_converts_a_missing_artifact_to_stable_evidence_error(
    tmp_path, missing
):
    mlflow = _persisted_mlflow(tmp_path, _record())
    mlflow.download_error = missing

    with pytest.raises(
        DecisionEvidenceError, match="valid decision/decision.json"
    ) as excinfo:
        load_decision("run-decision-1", mlflow_module=mlflow)

    assert excinfo.value.code == "aai_core.decisions.evidence_invalid"
    assert excinfo.value.__cause__ is missing
    assert mlflow.downloads == [("run-decision-1", "decision/decision.json")]


@pytest.mark.parametrize(
    "missing",
    [
        FileNotFoundError("missing decision run"),
        type(
            "MissingRunError",
            (Exception,),
            {"error_code": "NOT_FOUND"},
        )("decision run does not exist"),
    ],
)
def test_load_decision_converts_a_missing_run_to_stable_evidence_error(
    tmp_path, missing
):
    mlflow = _persisted_mlflow(tmp_path, _record())
    mlflow.get_error = missing

    with pytest.raises(
        DecisionEvidenceError, match="decision run does not exist"
    ) as excinfo:
        load_decision("run-decision-1", mlflow_module=mlflow)

    assert excinfo.value.code == "aai_core.decisions.evidence_invalid"
    assert excinfo.value.__cause__ is missing
    assert mlflow.requested_run_ids == ["run-decision-1"]
    assert mlflow.downloads == []


@pytest.mark.parametrize(
    "provider_error",
    [
        type(
            "DeniedArtifactError",
            (Exception,),
            {"error_code": "PERMISSION_DENIED"},
        )("decision artifact does not exist"),
        ConnectionError("decision artifact does not exist"),
        PermissionError("decision artifact not found"),
        TimeoutError("decision artifact does not exist"),
        OSError("decision artifact not found"),
    ],
)
def test_load_decision_propagates_artifact_auth_and_transport_failures(
    tmp_path, provider_error
):
    mlflow = _persisted_mlflow(tmp_path, _record())
    mlflow.download_error = provider_error

    with pytest.raises(type(provider_error)) as excinfo:
        load_decision("run-decision-1", mlflow_module=mlflow)

    assert excinfo.value is provider_error


@pytest.mark.parametrize(
    "provider_error",
    [
        type(
            "DeniedRunError",
            (Exception,),
            {"error_code": "PERMISSION_DENIED"},
        )("decision run does not exist"),
        ConnectionError("decision run does not exist"),
        PermissionError("decision run not found"),
        TimeoutError("decision run does not exist"),
        OSError("decision run not found"),
    ],
)
def test_load_decision_propagates_run_auth_and_transport_failures(
    tmp_path, provider_error
):
    mlflow = _persisted_mlflow(tmp_path, _record())
    mlflow.get_error = provider_error

    with pytest.raises(type(provider_error)) as excinfo:
        load_decision("run-decision-1", mlflow_module=mlflow)

    assert excinfo.value is provider_error
    assert mlflow.downloads == []


def test_load_decision_validates_run_ids_before_mlflow_access():
    for invalid in (None, 123, "", "run id", "a" * 65):
        error = TypeError if not isinstance(invalid, str) else ValueError
        with pytest.raises(error):
            load_decision(invalid, mlflow_module=object())


def test_record_decision_refuses_a_record_that_skipped_validation():
    # model_copy(update=...) bypasses validators, so a rejected decision
    # can be flipped to adopt without its gate ever being rechecked.
    # Nothing may reach the run before the contract is re-established.
    fake = FakeMlflow()
    manager = ExperimentManager(
        experiment_name="/Shared/test",
        context=dev_settings().resource,
        mlflow_module=fake,
    )
    failing = GateResult(
        metrics={"citation_rate": 0.4},
        failures=(GateFailure(metric="citation_rate", reason="0.4 below 1"),),
    )
    rejected = _record(decision=Decision.REJECT, gate=failing)
    forged = rejected.model_copy(update={"decision": Decision.ADOPT})
    assert forged.decision is Decision.ADOPT  # the bypass really works

    with pytest.raises(ValidationError, match="failing gate"):
        record_decision(forged, experiments=manager)
    assert fake.run_names == []
    assert fake.tags == {}
    assert fake.artifacts == []


def test_record_decision_derives_the_run_name_from_the_bounded_change_id():
    # MLflow persists run names as the mlflow.runName tag, so a free-form
    # name override would let prompts, user content, or secrets bypass the
    # record's bounded fields; the name derives exclusively from change_id.
    fake = FakeMlflow()
    manager = ExperimentManager(
        experiment_name="/Shared/test",
        context=dev_settings().resource,
        mlflow_module=fake,
    )

    record_decision(_record(), experiments=manager)

    assert fake.run_names == ["decision-prompt-v2"]
    with pytest.raises(TypeError):
        record_decision(_record(), experiments=manager, run_name="free text")
