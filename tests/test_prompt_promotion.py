"""Unit tests for idempotent registration and evidence-gated promotion."""

from types import SimpleNamespace

import pytest

from aai_core.decisions import (
    Decision,
    DecisionEvidenceError,
    DecisionRecord,
    decision_digest,
)
from aai_core.evaluation import GatePolicy, GateResult, MetricDirection, MetricRule
from aai_core.prompts import (
    PromptManager,
    PromptPromotionError,
    is_missing_prompt_error,
    prompt_digest,
)
from aai_core.testing import dev_settings

TEMPLATE = "Summarize only facts from {{earnings_excerpt}}."


def _passing_gate():
    return GateResult(
        metrics={"citation_rate": 1.0},
        policy=GatePolicy(
            rules=(
                MetricRule(
                    metric="citation_rate",
                    direction=MetricDirection.HIGHER,
                    required=1.0,
                ),
            )
        ),
    )


class _Page(list):
    def __init__(self, items, token=None):
        super().__init__(items)
        self.token = token


class FakeClient:
    def __init__(self, owner):
        self.owner = owner

    def get_prompt(self, name):
        if self.owner.get_error is not None:
            raise self.owner.get_error
        return self.owner.prompt

    def search_prompt_versions(self, name, page_token=None):
        self.owner.searched_name = name
        self.owner.page_tokens_requested.append(page_token)
        if self.owner.pages is not None:
            index = 0 if page_token is None else int(page_token)
            return self.owner.pages[index]
        return list(self.owner.versions)

    def get_prompt_version(self, name, version):
        # Deliberately the only fetch the fake client offers: promote()
        # must never use a linking load, client-level or fluent.
        if self.owner.version_error is not None:
            raise self.owner.version_error
        uri = f"prompts:/{name}/{version}"
        return SimpleNamespace(
            uri=uri, template=self.owner.genai.templates_by_uri.get(uri)
        )

    def get_run(self, run_id):
        self.owner.decision_run_requests.append(run_id)
        if self.owner.decision_run_error is not None:
            raise self.owner.decision_run_error
        if self.owner.decision_run is None:
            raise AssertionError("promotion requested an unexpected decision run")
        return self.owner.decision_run

    def download_artifacts(self, run_id, artifact_path):
        self.owner.decision_artifact_requests.append((run_id, artifact_path))
        if self.owner.decision_artifact_error is not None:
            raise self.owner.decision_artifact_error
        if self.owner.decision_artifact is None:
            raise AssertionError("promotion requested an unexpected artifact")
        return str(self.owner.decision_artifact)


class FakeGenAI:
    def __init__(self, templates_by_uri=None):
        self.registered = []
        self.alias = None
        self.templates_by_uri = dict(templates_by_uri or {})
        # The fluent load links to active lineage; promote() must never use it.
        self.linking_loads: list = []

    def register_prompt(self, **kwargs):
        self.registered.append(kwargs)
        return SimpleNamespace(
            name=kwargs["name"], version=len(self.registered), kwargs=kwargs
        )

    def load_prompt(self, uri, **kwargs):
        self.linking_loads.append(uri)
        return SimpleNamespace(uri=uri, template=self.templates_by_uri.get(uri))

    def set_prompt_alias(self, **kwargs):
        self.alias = kwargs


class FakeMlflow:
    def __init__(
        self,
        *,
        prompt=None,
        versions=(),
        get_error=None,
        templates_by_uri=None,
        pages=None,
        decision_run=None,
        decision_run_error=None,
        decision_artifact=None,
        decision_artifact_error=None,
        version_error=None,
    ):
        self.version_error = version_error
        self.prompt = prompt
        self.versions = list(versions)
        self.get_error = get_error
        self.pages = pages
        self.searched_name = None
        self.page_tokens_requested: list = []
        self.decision_run = decision_run
        self.decision_run_error = decision_run_error
        self.decision_artifact = decision_artifact
        self.decision_artifact_error = decision_artifact_error
        self.decision_run_requests: list[str] = []
        self.decision_artifact_requests: list[tuple[str, str]] = []
        self.genai = FakeGenAI(templates_by_uri)

    def MlflowClient(self):
        return FakeClient(self)


def _manager(mlflow):
    return PromptManager(
        context=dev_settings().resource,
        catalog="main",
        schema="app",
        mlflow_module=mlflow,
    )


def _persisted_mlflow(tmp_path, record, *, template=TEMPLATE):
    artifact = tmp_path / "decision.json"
    artifact.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
    tags = {
        "aai.run_purpose": "decision",
        "aai.change_id": record.change_id,
        "aai.change_summary": record.change_summary,
        "aai.decision_digest": decision_digest(record),
        **{f"aai.{name}": value for name, value in record.as_tags().items()},
    }
    if record.baseline_run_id:
        tags["aai.baseline_run_id"] = record.baseline_run_id
    run = SimpleNamespace(
        info=SimpleNamespace(run_id="run-decision-1", status="FINISHED"),
        data=SimpleNamespace(tags=tags, metrics=dict(record.gate.metrics)),
    )
    return FakeMlflow(
        templates_by_uri={"prompts:/main.app.earnings_summary/2": template},
        decision_run=run,
        decision_artifact=artifact,
    )


def test_is_missing_prompt_error_recognizes_only_absence():
    class RegistryError(Exception):
        def __init__(self, message, error_code=""):
            super().__init__(message)
            self.error_code = error_code

    assert is_missing_prompt_error(RegistryError("x", error_code="NOT_FOUND"))
    assert is_missing_prompt_error(RegistryError("prompt does not exist"))
    # Any structured non-absence code is authoritative. Falling through to
    # message wording here would swallow a real provider failure.
    assert not is_missing_prompt_error(
        RegistryError("prompt does not exist", error_code="INTERNAL_ERROR")
    )
    # The file/SQL registries report a missing alias as
    # INVALID_PARAMETER_VALUE with "Registered model alias ... not found."
    assert is_missing_prompt_error(
        RegistryError(
            "Registered model alias production not found.",
            error_code="INVALID_PARAMETER_VALUE",
        )
    )
    assert not is_missing_prompt_error(
        RegistryError("bad page token", error_code="INVALID_PARAMETER_VALUE")
    )
    # Auth, permission, and transient failures are not absence: a caller
    # seeding a first promotion must never swallow them — even when the
    # registry words the denial as "does not exist" to avoid disclosure.
    assert not is_missing_prompt_error(
        RegistryError("denied", error_code="PERMISSION_DENIED")
    )
    assert not is_missing_prompt_error(
        RegistryError("prompt does not exist", error_code="PERMISSION_DENIED")
    )
    assert not is_missing_prompt_error(RegistryError("401 unauthorized"))
    assert not is_missing_prompt_error(RegistryError("connection reset"))


@pytest.mark.parametrize(
    ("catalog", "schema"),
    [
        ("unset", "app"),
        ("main", "todo"),
        ("replace-with-catalog", "app"),
        ("main", " "),
    ],
)
def test_manager_fails_locally_on_placeholder_qualifiers(catalog, schema):
    manager = PromptManager(
        context=dev_settings().resource,
        catalog=catalog,
        schema=schema,
        # Any registry use would fail loudly: validation must come first.
        mlflow_module=object(),
    )

    with pytest.raises(ValueError, match="platform.catalog"):
        manager.ensure_version(
            "earnings_summary", TEMPLATE, commit_message="Initial version"
        )
    with pytest.raises(ValueError, match="platform.catalog"):
        manager.load("earnings_summary", version=1)
    # Explicit full qualification remains the caller's escape hatch.
    assert manager.qualify("main.app.earnings_summary") == "main.app.earnings_summary"


def test_qualify_rejects_blank_and_malformed_names():
    manager = PromptManager(
        context=dev_settings().resource,
        catalog="main",
        schema="app",
        # Malformed names must fail before any registry access.
        mlflow_module=object(),
    )

    for bad in (
        "",
        "   ",
        "main.app.",
        "..",
        "a..c",
        # The evidence contract refuses these shapes, so a prompt registered
        # under one could never receive promotion evidence.
        "monthly summary",
        "main. app.prompt",
        "main.app.name with spaces",
        # Explicit qualification is not an escape hatch from the
        # placeholder vocabulary.
        "unset.app.prompt",
        "replace-with-catalog.app.prompt",
        "unset",
    ):
        with pytest.raises(ValueError):
            manager.qualify(bad)
    with pytest.raises(ValueError, match="blank"):
        manager.ensure_version("  ", TEMPLATE, commit_message="Blank name")
    # str() would make None the valid-looking component "None", so
    # register()/load()/set_alias() would address a real, wrong prompt.
    for wrong_type in (None, 123, ["earnings_summary"]):
        with pytest.raises(TypeError, match="strings"):
            manager.qualify(wrong_type)
    assert manager.qualify("  earnings_summary  ") == "main.app.earnings_summary"


def test_registry_qualifiers_must_be_strings():
    # A non-string catalog or schema would coerce into a valid-looking
    # qualifier and address a real, wrong registry namespace.
    for catalog, schema in ((None, "app"), (123, "app"), ("main", None)):
        manager = PromptManager(
            context=dev_settings().resource,
            catalog=catalog,
            schema=schema,
            mlflow_module=object(),
        )
        with pytest.raises(TypeError, match="qualifier"):
            manager.qualify("earnings_summary")


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


def test_promote_refuses_bare_gate_evidence_even_when_it_passed():
    mlflow = FakeMlflow(
        templates_by_uri={"prompts:/main.app.earnings_summary/2": TEMPLATE}
    )

    with pytest.raises(PromptPromotionError, match="template identity") as excinfo:
        _manager(mlflow).promote(
            "earnings_summary",
            version=2,
            evidence=GateResult(metrics={"citation_rate": 1.0}),
        )

    assert excinfo.value.code == "aai_core.prompts.promotion_blocked"
    assert mlflow.genai.alias is None


def test_promote_refuses_an_adopt_decision_without_a_content_binding():
    mlflow = FakeMlflow(
        templates_by_uri={"prompts:/main.app.earnings_summary/2": TEMPLATE}
    )
    record = DecisionRecord(
        decision=Decision.ADOPT,
        change_id="prompt-v2",
        change_summary="Require one exact source citation.",
        rationale="Citation rate reached 1.0 with no quality regression.",
        gate=_passing_gate(),
    )

    with pytest.raises(PromptPromotionError, match="not bound"):
        _manager(mlflow).promote("earnings_summary", version=2, evidence=record)

    assert mlflow.genai.alias is None


def test_promote_refuses_a_version_whose_content_disagrees_with_evidence(tmp_path):
    changed = TEMPLATE + " Cite {{source_id}} exactly once."
    record = DecisionRecord(
        decision=Decision.ADOPT,
        change_id="prompt-v2",
        change_summary="Require one exact source citation.",
        rationale="Citation rate reached 1.0 with no quality regression.",
        gate=_passing_gate(),
        prompt_name="main.app.earnings_summary",
        prompt_version=2,
        prompt_digest=prompt_digest(TEMPLATE),
    )
    mlflow = _persisted_mlflow(tmp_path, record, template=changed)

    with pytest.raises(PromptPromotionError, match="content digest"):
        _manager(mlflow).promote(
            "earnings_summary",
            version=2,
            decision_run_id="run-decision-1",
        )

    assert mlflow.genai.alias is None


def test_promote_binds_evidence_to_the_exact_prompt_and_version():
    # Content identity is not registry identity: evidence for one prompt
    # must never promote another that shares the same template.
    mlflow = FakeMlflow(
        templates_by_uri={"prompts:/main.app.earnings_summary/2": TEMPLATE}
    )

    def _record(**overrides):
        values = {
            "decision": Decision.ADOPT,
            "change_id": "prompt-v2",
            "change_summary": "Require one exact source citation.",
            "rationale": "Citation rate reached 1.0 with no regression.",
            "gate": _passing_gate(),
            "prompt_digest": prompt_digest(TEMPLATE),
        }
        values.update(overrides)
        return DecisionRecord(**values)

    with pytest.raises(PromptPromotionError, match="names no prompt"):
        _manager(mlflow).promote("earnings_summary", version=2, evidence=_record())
    with pytest.raises(PromptPromotionError, match="bound to 'main.app.other_prompt'"):
        _manager(mlflow).promote(
            "earnings_summary",
            version=2,
            evidence=_record(prompt_name="main.app.other_prompt"),
        )
    # Two immutable versions can share a template, so version-unbound
    # evidence is refused outright, not just on mismatch.
    with pytest.raises(PromptPromotionError, match="not bound to a registry version"):
        _manager(mlflow).promote(
            "earnings_summary",
            version=2,
            evidence=_record(prompt_name="main.app.earnings_summary"),
        )
    with pytest.raises(PromptPromotionError, match="bound to version 3"):
        _manager(mlflow).promote(
            "earnings_summary",
            version=2,
            evidence=_record(prompt_name="main.app.earnings_summary", prompt_version=3),
        )
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


def test_promote_requires_a_persisted_decision_run():
    record = DecisionRecord(
        decision=Decision.ADOPT,
        change_id="prompt-v2",
        change_summary="Require one exact source citation.",
        rationale="Citation rate reached 1.0 with no quality regression.",
        gate=_passing_gate(),
        prompt_name="main.app.earnings_summary",
        prompt_version=2,
        prompt_digest=prompt_digest(TEMPLATE),
    )

    mlflow = FakeMlflow()
    with pytest.raises(PromptPromotionError, match="persisted decision run"):
        _manager(mlflow).promote("earnings_summary", version=2, evidence=record)

    assert mlflow.genai.alias is None
    assert mlflow.decision_run_requests == []


def test_promote_accepts_a_verified_persisted_adopt_decision(tmp_path):
    record = DecisionRecord(
        decision=Decision.ADOPT,
        change_id="prompt-v2",
        change_summary="Require one exact source citation.",
        rationale="Citation rate reached 1.0 with no quality regression.",
        gate=_passing_gate(),
        prompt_name="main.app.earnings_summary",
        prompt_version=2,
        prompt_digest=prompt_digest(TEMPLATE),
    )
    mlflow = _persisted_mlflow(tmp_path, record)

    _manager(mlflow).promote(
        "earnings_summary",
        version=2,
        decision_run_id="run-decision-1",
    )

    assert mlflow.genai.alias["alias"] == "production"
    assert mlflow.decision_run_requests == ["run-decision-1"]
    assert mlflow.decision_artifact_requests == [
        ("run-decision-1", "decision/decision.json")
    ]
    # Verification fetches through the raw client: the fluent load would
    # link the candidate to any active run, model, or trace.
    assert mlflow.genai.linking_loads == []


def test_promote_reports_a_missing_decision_artifact_as_blocked_evidence(tmp_path):
    class MissingArtifactError(Exception):
        error_code = "RESOURCE_DOES_NOT_EXIST"

    record = DecisionRecord(
        decision=Decision.ADOPT,
        change_id="prompt-v2",
        change_summary="Require one exact source citation.",
        rationale="Citation rate reached 1.0 with no quality regression.",
        gate=_passing_gate(),
        prompt_name="main.app.earnings_summary",
        prompt_version=2,
        prompt_digest=prompt_digest(TEMPLATE),
    )
    mlflow = _persisted_mlflow(tmp_path, record)
    missing = MissingArtifactError("decision artifact does not exist")
    mlflow.decision_artifact_error = missing

    with pytest.raises(
        PromptPromotionError, match="valid decision/decision.json"
    ) as excinfo:
        _manager(mlflow).promote(
            "earnings_summary",
            version=2,
            decision_run_id="run-decision-1",
        )

    assert excinfo.value.remediation is not None
    assert isinstance(excinfo.value.__cause__, DecisionEvidenceError)
    assert excinfo.value.__cause__.__cause__ is missing
    assert mlflow.genai.alias is None


def test_promote_reports_a_missing_prompt_version_as_refusal(tmp_path):
    # The third fetch on the promotion path: a version that was never
    # registered is invalid promotion input, so it must reach the caller as
    # the guarded refusal with remediation, not as a raw registry error.
    class MissingVersionError(Exception):
        error_code = "RESOURCE_DOES_NOT_EXIST"

    record = DecisionRecord(
        decision=Decision.ADOPT,
        change_id="prompt-v2",
        change_summary="Require one exact source citation.",
        rationale="Citation rate reached 1.0 with no quality regression.",
        gate=_passing_gate(),
        prompt_name="main.app.earnings_summary",
        prompt_version=2,
        prompt_digest=prompt_digest(TEMPLATE),
    )
    mlflow = _persisted_mlflow(tmp_path, record)
    missing = MissingVersionError("prompt version does not exist")
    mlflow.version_error = missing

    with pytest.raises(PromptPromotionError, match="does not exist") as excinfo:
        _manager(mlflow).promote(
            "earnings_summary",
            version=2,
            decision_run_id="run-decision-1",
        )

    assert excinfo.value.remediation is not None
    assert excinfo.value.__cause__ is missing
    assert mlflow.genai.alias is None


@pytest.mark.parametrize(
    "version_error",
    [
        type("DeniedVersionError", (Exception,), {"error_code": "PERMISSION_DENIED"})(
            "prompt version does not exist"
        ),
        type("TransportVersionError", (Exception,), {})("connection reset"),
    ],
)
def test_promote_propagates_prompt_version_auth_and_transport_failures(
    tmp_path, version_error
):
    # An access or connectivity fault must never be reported as a missing
    # version — the alias stays untouched either way, but the operator
    # needs the real cause.
    record = DecisionRecord(
        decision=Decision.ADOPT,
        change_id="prompt-v2",
        change_summary="Require one exact source citation.",
        rationale="Citation rate reached 1.0 with no quality regression.",
        gate=_passing_gate(),
        prompt_name="main.app.earnings_summary",
        prompt_version=2,
        prompt_digest=prompt_digest(TEMPLATE),
    )
    mlflow = _persisted_mlflow(tmp_path, record)
    mlflow.version_error = version_error

    with pytest.raises(type(version_error)):
        _manager(mlflow).promote(
            "earnings_summary",
            version=2,
            decision_run_id="run-decision-1",
        )

    assert mlflow.genai.alias is None


def test_promote_reports_a_missing_decision_run_as_blocked_evidence(tmp_path):
    class MissingRunError(Exception):
        error_code = "RESOURCE_DOES_NOT_EXIST"

    record = DecisionRecord(
        decision=Decision.ADOPT,
        change_id="prompt-v2",
        change_summary="Require one exact source citation.",
        rationale="Citation rate reached 1.0 with no quality regression.",
        gate=_passing_gate(),
        prompt_name="main.app.earnings_summary",
        prompt_version=2,
        prompt_digest=prompt_digest(TEMPLATE),
    )
    mlflow = _persisted_mlflow(tmp_path, record)
    missing = MissingRunError("decision run does not exist")
    mlflow.decision_run_error = missing

    with pytest.raises(
        PromptPromotionError, match="decision run does not exist"
    ) as excinfo:
        _manager(mlflow).promote(
            "earnings_summary",
            version=2,
            decision_run_id="run-decision-1",
        )

    assert excinfo.value.remediation is not None
    assert isinstance(excinfo.value.__cause__, DecisionEvidenceError)
    assert excinfo.value.__cause__.__cause__ is missing
    assert mlflow.decision_artifact_requests == []
    assert mlflow.genai.alias is None


@pytest.mark.parametrize(
    "provider_error",
    [
        type(
            "DeniedRunError",
            (Exception,),
            {"error_code": "PERMISSION_DENIED"},
        )("decision run does not exist"),
        ConnectionError("connection reset"),
    ],
)
def test_promote_propagates_decision_run_auth_and_transport_failures(
    tmp_path, provider_error
):
    record = DecisionRecord(
        decision=Decision.ADOPT,
        change_id="prompt-v2",
        change_summary="Require one exact source citation.",
        rationale="Citation rate reached 1.0 with no quality regression.",
        gate=_passing_gate(),
        prompt_name="main.app.earnings_summary",
        prompt_version=2,
        prompt_digest=prompt_digest(TEMPLATE),
    )
    mlflow = _persisted_mlflow(tmp_path, record)
    mlflow.decision_run_error = provider_error

    with pytest.raises(type(provider_error)) as excinfo:
        _manager(mlflow).promote(
            "earnings_summary",
            version=2,
            decision_run_id="run-decision-1",
        )

    assert excinfo.value is provider_error
    assert mlflow.decision_artifact_requests == []
    assert mlflow.genai.alias is None


def test_promote_refuses_in_memory_evidence_that_differs_from_the_run(tmp_path):
    persisted = DecisionRecord(
        decision=Decision.ADOPT,
        change_id="prompt-v2",
        change_summary="Require one exact source citation.",
        rationale="Citation rate reached 1.0 with no quality regression.",
        gate=_passing_gate(),
        prompt_name="main.app.earnings_summary",
        prompt_version=2,
        prompt_digest=prompt_digest(TEMPLATE),
    )
    supplied = persisted.model_copy(
        update={"rationale": "A different in-memory rationale."}
    )
    mlflow = _persisted_mlflow(tmp_path, persisted)

    with pytest.raises(PromptPromotionError, match="differs from decision.json"):
        _manager(mlflow).promote(
            "earnings_summary",
            version=2,
            decision_run_id="run-decision-1",
            evidence=supplied,
        )

    assert mlflow.genai.alias is None


def test_promote_rejects_unknown_evidence_types():
    with pytest.raises(TypeError, match="DecisionRecord"):
        _manager(FakeMlflow()).promote(
            "earnings_summary", version=2, evidence={"passed": True}
        )


def test_promote_validates_the_alias_before_any_registry_access():
    record = DecisionRecord(
        decision=Decision.ADOPT,
        change_id="prompt-v2",
        change_summary="Require one exact source citation.",
        rationale="Citation rate reached 1.0 with no quality regression.",
        gate=_passing_gate(),
        prompt_digest=prompt_digest(TEMPLATE),
    )
    manager = PromptManager(
        context=dev_settings().resource,
        catalog="main",
        schema="app",
        # An alias typo must fail deterministically before this is touched.
        mlflow_module=object(),
    )

    with pytest.raises(ValueError, match="Unsupported governed prompt alias"):
        manager.promote(
            "earnings_summary", version=2, evidence=record, alias="prodution"
        )


def test_ensure_version_follows_pagination_to_find_an_existing_match():
    filler = SimpleNamespace(template="unrelated", version=1, tags={})
    match = SimpleNamespace(
        template=TEMPLATE,
        version=7,
        tags={"aai_prompt_digest": prompt_digest(TEMPLATE)},
    )
    mlflow = FakeMlflow(
        prompt=object(),
        pages=[_Page([filler], token="1"), _Page([match])],
    )

    result = _manager(mlflow).ensure_version(
        "earnings_summary", TEMPLATE, commit_message="Reuse baseline"
    )

    assert result is match
    assert mlflow.genai.registered == []
    assert mlflow.page_tokens_requested == [None, "1"]
