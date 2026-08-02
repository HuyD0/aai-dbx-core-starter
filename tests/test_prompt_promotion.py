"""Unit tests for idempotent registration and evidence-gated promotion."""

from types import SimpleNamespace

import pytest

from aai_core.decisions import Decision, DecisionRecord
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
        uri = f"prompts:/{name}/{version}"
        return SimpleNamespace(
            uri=uri, template=self.owner.genai.templates_by_uri.get(uri)
        )


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
    ):
        self.prompt = prompt
        self.versions = list(versions)
        self.get_error = get_error
        self.pages = pages
        self.searched_name = None
        self.page_tokens_requested: list = []
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


def test_is_missing_prompt_error_recognizes_only_absence():
    class RegistryError(Exception):
        def __init__(self, message, error_code=""):
            super().__init__(message)
            self.error_code = error_code

    assert is_missing_prompt_error(RegistryError("x", error_code="NOT_FOUND"))
    assert is_missing_prompt_error(RegistryError("prompt does not exist"))
    # MlflowException defaults to INTERNAL_ERROR even for message-only
    # raises, so a non-authoritative code still falls through to markers.
    assert is_missing_prompt_error(
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


def test_promote_refuses_a_version_whose_content_disagrees_with_evidence():
    changed = TEMPLATE + " Cite {{source_id}} exactly once."
    mlflow = FakeMlflow(
        templates_by_uri={"prompts:/main.app.earnings_summary/2": changed}
    )
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

    with pytest.raises(PromptPromotionError, match="content digest"):
        _manager(mlflow).promote("earnings_summary", version=2, evidence=record)

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


def test_promote_accepts_an_adopt_decision_bound_by_prompt_digest():
    mlflow = FakeMlflow(
        templates_by_uri={"prompts:/main.app.earnings_summary/2": TEMPLATE}
    )
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

    _manager(mlflow).promote("earnings_summary", version=2, evidence=record)

    assert mlflow.genai.alias["alias"] == "production"
    # Verification fetches through the raw client: the fluent load would
    # link the candidate to any active run, model, or trace.
    assert mlflow.genai.linking_loads == []


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
