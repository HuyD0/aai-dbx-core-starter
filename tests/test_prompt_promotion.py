"""Unit tests for idempotent registration and evidence-gated promotion."""

from types import SimpleNamespace

import pytest

from aai_core.decisions import Decision, DecisionRecord
from aai_core.evaluation import GateFailure, GateResult
from aai_core.prompts import PromptManager, PromptPromotionError, prompt_digest
from aai_core.testing import dev_settings

TEMPLATE = "Summarize only facts from {{earnings_excerpt}}."


class FakeClient:
    def __init__(self, owner):
        self.owner = owner

    def get_prompt(self, name):
        if self.owner.get_error is not None:
            raise self.owner.get_error
        return self.owner.prompt

    def search_prompt_versions(self, name):
        self.owner.searched_name = name
        return list(self.owner.versions)


class FakeGenAI:
    def __init__(self):
        self.registered = []
        self.alias = None

    def register_prompt(self, **kwargs):
        self.registered.append(kwargs)
        return SimpleNamespace(
            name=kwargs["name"], version=len(self.registered), kwargs=kwargs
        )

    def set_prompt_alias(self, **kwargs):
        self.alias = kwargs


class FakeMlflow:
    def __init__(self, *, prompt=None, versions=(), get_error=None):
        self.prompt = prompt
        self.versions = list(versions)
        self.get_error = get_error
        self.searched_name = None
        self.genai = FakeGenAI()

    def MlflowClient(self):
        return FakeClient(self)


def _manager(mlflow):
    return PromptManager(
        context=dev_settings().resource,
        catalog="main",
        schema="app",
        mlflow_module=mlflow,
    )


def test_prompt_digest_is_stable_and_content_sensitive():
    assert prompt_digest(TEMPLATE) == prompt_digest(TEMPLATE)
    assert prompt_digest(TEMPLATE) != prompt_digest(TEMPLATE + " Cite {{source_id}}.")
    messages = [{"role": "system", "content": TEMPLATE}]
    assert prompt_digest(messages) == prompt_digest(
        [{"content": TEMPLATE, "role": "system"}]
    )


def test_ensure_version_reuses_an_identical_registered_version():
    existing = SimpleNamespace(
        template=TEMPLATE,
        version=3,
        tags={"aai_prompt_digest": prompt_digest(TEMPLATE)},
    )
    mlflow = FakeMlflow(prompt=object(), versions=[existing])

    result = _manager(mlflow).ensure_version(
        "earnings_summary", TEMPLATE, commit_message="Reuse baseline"
    )

    assert result is existing
    assert mlflow.genai.registered == []
    assert mlflow.searched_name == "main.app.earnings_summary"


def test_ensure_version_registers_when_the_prompt_is_missing():
    missing = RuntimeError("RESOURCE_DOES_NOT_EXIST: no such prompt")
    mlflow = FakeMlflow(get_error=missing)

    result = _manager(mlflow).ensure_version(
        "earnings_summary",
        TEMPLATE,
        commit_message="Establish the baseline",
        tags={"experiment_role": "baseline", "prompt_digest": "user-override"},
    )

    assert result.version == 1
    registered = mlflow.genai.registered[0]
    assert registered["name"] == "main.app.earnings_summary"
    assert registered["tags"]["aai_prompt_digest"] == prompt_digest(TEMPLATE)
    assert registered["tags"]["aai_experiment_role"] == "baseline"


def test_ensure_version_registers_a_changed_template_as_a_new_version():
    existing = SimpleNamespace(
        template=TEMPLATE,
        version=1,
        tags={"aai_prompt_digest": prompt_digest(TEMPLATE)},
    )
    mlflow = FakeMlflow(prompt=object(), versions=[existing])
    changed = TEMPLATE + " Cite {{source_id}} exactly once."

    _manager(mlflow).ensure_version(
        "earnings_summary", changed, commit_message="Require a citation"
    )

    assert len(mlflow.genai.registered) == 1


def test_ensure_version_raises_on_unrelated_registry_errors():
    mlflow = FakeMlflow(get_error=PermissionError("permission denied"))

    with pytest.raises(PermissionError):
        _manager(mlflow).ensure_version(
            "earnings_summary", TEMPLATE, commit_message="Reuse"
        )


def test_promote_moves_the_alias_on_a_passing_gate():
    mlflow = FakeMlflow()

    _manager(mlflow).promote(
        "earnings_summary",
        version=2,
        evidence=GateResult(metrics={"citation_rate": 1.0}),
    )

    assert mlflow.genai.alias == {
        "name": "main.app.earnings_summary",
        "alias": "production",
        "version": 2,
    }


def test_promote_refuses_a_failing_gate_and_names_the_metrics():
    mlflow = FakeMlflow()
    failing = GateResult(
        metrics={"citation_rate": 0.4},
        failures=(GateFailure(metric="citation_rate", reason="0.4 below 1"),),
    )

    with pytest.raises(PromptPromotionError, match="citation_rate") as excinfo:
        _manager(mlflow).promote("earnings_summary", version=2, evidence=failing)

    assert excinfo.value.code == "aai_core.prompts.promotion_blocked"
    assert mlflow.genai.alias is None


@pytest.mark.parametrize("decision", [Decision.REJECT, Decision.INCONCLUSIVE])
def test_promote_refuses_non_adopt_decisions(decision):
    mlflow = FakeMlflow()
    record = DecisionRecord(
        decision=decision,
        change_id="prompt-v2",
        change_summary="Require one exact source citation.",
        rationale="Citation rate stayed below the release threshold.",
    )

    with pytest.raises(PromptPromotionError, match=decision.value):
        _manager(mlflow).promote("earnings_summary", version=2, evidence=record)

    assert mlflow.genai.alias is None


def test_promote_accepts_an_adopt_decision():
    mlflow = FakeMlflow()
    record = DecisionRecord(
        decision=Decision.ADOPT,
        change_id="prompt-v2",
        change_summary="Require one exact source citation.",
        rationale="Citation rate reached 1.0 with no quality regression.",
        gate=GateResult(metrics={"citation_rate": 1.0}),
    )

    _manager(mlflow).promote("earnings_summary", version=2, evidence=record)

    assert mlflow.genai.alias["alias"] == "production"


def test_promote_rejects_unknown_evidence_types():
    with pytest.raises(TypeError, match="GateResult or DecisionRecord"):
        _manager(FakeMlflow()).promote(
            "earnings_summary", version=2, evidence={"passed": True}
        )
