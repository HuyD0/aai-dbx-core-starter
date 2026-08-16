"""RAG releases join the exact code, world, configuration, controls, and gate."""

import json
import sys
from types import SimpleNamespace

import pytest

from aai_core import __version__
from aai_core.experiments import record_reproducibility
from app.rag import DEFAULT_RAG_LIMITS, rag_limit_parameters
from app.release_evidence import (
    configuration_digests,
    endpoint_sha256,
    knowledge_version,
    model_identity,
    release_configuration,
)
from evals.evaluate import _evaluation_model_identities
from scripts import create_release

SOURCE_COMMIT = "c" * 40
DATASET_NAME = "catalog.schema.test_rag_release_cases"
PROMPT_NAME = "catalog.schema.agent-system"


def _settings():
    return SimpleNamespace(
        catalog="catalog",
        schema="schema",
        effective_experiment_name="/Shared/test-rag",
        models={
            "general-chat": {
                "provider": "databricks",
                "deployment": "target-endpoint",
                "endpoint": "https://unused-runtime.example.invalid/api/projects/test",
                "capabilities": {"structured_output": True},
            },
            "judge-model": {
                "provider": "databricks",
                "deployment": "judge-endpoint",
            },
        },
        embeddings={
            "knowledge-embedding": {
                "provider": "databricks",
                "deployment": "embedding-endpoint",
                "endpoint": "https://unused-runtime.example.invalid/api/projects/test",
                "dimensions": 1536,
            }
        },
        retrievers={
            "product-knowledge": {
                "provider": "databricks_ai_search",
                "endpoint": "vector-search-endpoint",
                "index": "catalog.schema.knowledge_v002",
                "id_field": "id",
                "content_field": "content",
                "source_uri_field": "source_uri",
                "chunk_id_field": "chunk_id",
                "vector_fields": ["content_vector"],
                "columns": ["id", "content", "source_uri", "chunk_id"],
                "embedding": "knowledge-embedding",
            }
        },
    )


def _dataset(*, experiment_ids=("experiment-1",), digest="dataset-v1"):
    return SimpleNamespace(
        dataset_id="dataset-1",
        digest=digest,
        experiment_ids=experiment_ids,
    )


def _parameters(settings=None):
    configured = settings or _settings()
    digests = configuration_digests(release_configuration(configured))
    return {
        "prompt_version": "7",
        "prompt_uri": f"prompts:/{PROMPT_NAME}/7",
        "evaluation_dataset": DATASET_NAME,
        "evaluation_dataset_id": "dataset-1",
        "evaluation_dataset_digest": "dataset-v1",
        "target_model": model_identity(configured, "general-chat"),
        "judge_model": model_identity(configured, "judge-model"),
        "source_commit": SOURCE_COMMIT,
        "source_state": "clean",
        "aai_core_version": __version__,
        "knowledge_version": "knowledge-v002",
        "rag_limits_digest": DEFAULT_RAG_LIMITS.digest,
        "gate_policy_digest": "a" * 64,
        "gate_baseline_digest": "b" * 64,
        **digests,
        **rag_limit_parameters(DEFAULT_RAG_LIMITS),
    }


def _run(
    *,
    parameters=None,
    passed="true",
    status="FINISHED",
    experiment_id="experiment-1",
    settings=None,
):
    configured = settings or _settings()
    return SimpleNamespace(
        info=SimpleNamespace(status=status, experiment_id=experiment_id),
        data=SimpleNamespace(
            params=parameters or _parameters(configured),
            tags={
                "aai.gate_passed": passed,
                "aai.target_model": model_identity(configured, "general-chat"),
                "aai.judge_model": model_identity(configured, "judge-model"),
            },
        ),
    )


def _validate(*, run=None, dataset=None, settings=None, source_state="clean"):
    configured = settings or _settings()
    return create_release._validated_gate_evidence(
        run=run or _run(settings=configured),
        dataset=dataset or _dataset(),
        dataset_name=DATASET_NAME,
        expected_experiment_id="experiment-1",
        prompt_name=PROMPT_NAME,
        prompt_version=7,
        settings=configured,
        current_source_commit=SOURCE_COMMIT,
        current_source_state=source_state,
        current_knowledge_version="knowledge-v002",
        current_configuration_digests=configuration_digests(
            release_configuration(configured)
        ),
    )


def test_validated_gate_evidence_happy_path():
    assert _validate() == {
        "gate_policy_digest": "a" * 64,
        "gate_baseline_digest": "b" * 64,
        "target_model": "databricks:target-endpoint",
        "judge_model": "databricks:judge-endpoint",
    }


@pytest.mark.parametrize(
    "field",
    [
        "prompt_version",
        "prompt_uri",
        "evaluation_dataset",
        "evaluation_dataset_id",
        "evaluation_dataset_digest",
        "target_model",
        "judge_model",
        "source_commit",
        "source_state",
        "aai_core_version",
        "knowledge_version",
        "model_configuration_digest",
        "embedding_configuration_digest",
        "retrieval_configuration_digest",
        "index_configuration_digest",
        "rag_configuration_digest",
        "rag_limits_digest",
        "limit_max_output_tokens",
    ],
)
def test_stale_release_join_fields_are_rejected(field):
    parameters = _parameters()
    parameters[field] = "stale"

    with pytest.raises(RuntimeError, match="no longer matches"):
        _validate(run=_run(parameters=parameters))


def test_run_status_gate_experiment_and_dataset_association_fail_closed():
    with pytest.raises(RuntimeError, match="FINISHED"):
        _validate(run=_run(status="FAILED"))
    with pytest.raises(RuntimeError, match="passing"):
        _validate(run=_run(passed="false"))
    with pytest.raises(RuntimeError, match="configured experiment"):
        _validate(run=_run(experiment_id="experiment-2"))
    with pytest.raises(RuntimeError, match="not associated"):
        _validate(dataset=_dataset(experiment_ids=None))


def test_target_and_judge_tags_are_part_of_the_join():
    run = _run()
    run.data.tags["aai.target_model"] = "stale"

    with pytest.raises(RuntimeError, match="aai.target_model"):
        _validate(run=run)


def test_gate_digests_are_validated_and_none_baseline_is_supported():
    parameters = _parameters()
    parameters["gate_policy_digest"] = "not-a-digest"
    with pytest.raises(RuntimeError, match="SHA-256"):
        _validate(run=_run(parameters=parameters))

    parameters = _parameters()
    parameters["gate_baseline_digest"] = "none"
    assert _validate(run=_run(parameters=parameters))["gate_baseline_digest"] is None


def test_current_dirty_or_unknown_source_tree_is_rejected():
    for state in ("dirty", "unknown"):
        with pytest.raises(RuntimeError, match="clean source tree"):
            _validate(source_state=state)


def test_untracked_source_or_eval_file_marks_tree_dirty(monkeypatch):
    monkeypatch.delenv("GIT_DIRTY", raising=False)
    captured = []

    def fake_run(command, **options):
        captured.append((command, options))
        return SimpleNamespace(stdout="?? evals/data/new_failure.json\n")

    monkeypatch.setattr(create_release.subprocess, "run", fake_run)

    assert create_release.source_state() == "dirty"
    assert "--untracked-files=normal" in captured[0][0]


def test_unknown_git_dirty_override_never_becomes_clean(monkeypatch):
    monkeypatch.setenv("GIT_DIRTY", "flase")
    assert create_release.source_state() == "unknown"


@pytest.mark.parametrize("value", ["", "main", "abc123", "g" * 40])
def test_source_commit_rejects_ref_names_and_invalid_object_ids(monkeypatch, value):
    monkeypatch.setenv("GIT_COMMIT", value)

    with pytest.raises(RuntimeError, match="full 40- or 64-character"):
        create_release.source_commit()


def test_source_commit_normalizes_full_sha256_object_id(monkeypatch):
    monkeypatch.setenv("GIT_COMMIT", "A" * 64)
    assert create_release.source_commit() == "a" * 64


def test_bundle_provenance_environment_joins_eval_to_release(monkeypatch):
    class FakeMlflow:
        def __init__(self):
            self.parameters = {}

        def log_params(self, parameters):
            self.parameters.update(parameters)

        def log_artifact(self, *_args, **_kwargs):
            return None

        def set_tags(self, *_args, **_kwargs):
            return None

    monkeypatch.setenv("GIT_COMMIT", SOURCE_COMMIT)
    monkeypatch.setenv("GIT_DIRTY", "false")
    native = FakeMlflow()
    record_reproducibility(mlflow_module=native)
    parameters = _parameters()
    parameters.update(
        {
            "source_commit": native.parameters["source_commit"],
            "source_state": native.parameters["source_state"],
        }
    )

    assert _validate(run=_run(parameters=parameters))["target_model"] == (
        "databricks:target-endpoint"
    )
    assert create_release.source_commit() == SOURCE_COMMIT
    assert create_release.source_state() == "clean"


@pytest.mark.parametrize(
    "value",
    [
        "",
        "contains spaces",
        "knowledge\nprod",
        "x" * 129,
        "replace-with-knowledge-version",
        "sk-do-not-store-this",
    ],
)
def test_knowledge_version_rejects_placeholders_or_secret_values(value):
    with pytest.raises(ValueError, match="bounded non-secret identifier") as error:
        knowledge_version(value)
    if value:
        assert value not in str(error.value)


def test_knowledge_version_rejects_non_string_values():
    with pytest.raises(TypeError, match="bounded non-secret identifier"):
        knowledge_version(None)


def test_configuration_evidence_is_secret_free_and_endpoint_aware():
    settings = _settings()
    configuration = release_configuration(settings)

    assert "endpoint" not in configuration["model"]
    assert "endpoint_sha256" not in configuration["model"]
    assert len(configuration["retrieval"]["endpoint_sha256"]) == 64
    assert "https://" not in json.dumps(configuration)
    evaluated_target, _ = _evaluation_model_identities(settings)
    assert evaluated_target == model_identity(settings, "general-chat")


def test_configuration_evidence_rejects_inline_secrets_without_echoing_them():
    settings = _settings()
    settings.retrievers["product-knowledge"]["api_key"] = "sk-do-not-log-this"

    with pytest.raises(RuntimeError, match="secret") as error:
        release_configuration(settings)
    assert "do-not-log-this" not in str(error.value)


def test_configuration_boundaries_fail_closed_when_incomplete():
    with pytest.raises(ValueError, match="incomplete"):
        configuration_digests({})
    with pytest.raises(TypeError, match="settings.models"):
        release_configuration(SimpleNamespace())

    settings = _settings()
    settings.models.pop("general-chat")
    with pytest.raises(TypeError, match="general-chat"):
        release_configuration(settings)

    settings = _settings()
    settings.embeddings["knowledge-embedding"]["deployment"] = ""
    with pytest.raises(TypeError, match="deployment"):
        release_configuration(settings)


@pytest.mark.parametrize(
    "value",
    [
        None,
        "endpoint with spaces",
        "https://search.example.invalid/index?sig=not-allowed",
        "https://search.example.invalid/a/../b",
    ],
)
def test_endpoint_evidence_rejects_ambiguous_or_secret_bearing_routes(value):
    with pytest.raises((TypeError, ValueError), match="endpoint"):
        endpoint_sha256(value)


def test_credential_shaped_nested_value_is_rejected_without_echoing_it():
    settings = _settings()
    settings.models["general-chat"]["capabilities"] = ["sk-do-not-log-this"]

    with pytest.raises(RuntimeError, match="credential") as error:
        release_configuration(settings)
    assert "do-not-log-this" not in str(error.value)


def test_main_writes_v2_world_learning_and_control_evidence(monkeypatch, tmp_path):
    settings = _settings()
    dataset = _dataset()
    run = _run(settings=settings)
    context = SimpleNamespace(
        settings=settings,
        tags=SimpleNamespace(
            application="test-rag",
            release="dev",
            environment="dev",
        ),
        prompts=SimpleNamespace(qualify=lambda name: f"catalog.schema.{name}"),
    )
    fake_mlflow = SimpleNamespace(
        get_experiment_by_name=lambda name: SimpleNamespace(
            experiment_id="experiment-1"
        ),
        genai=SimpleNamespace(
            datasets=SimpleNamespace(get_dataset=lambda **kwargs: dataset)
        ),
        get_run=lambda run_id: run,
    )
    output = tmp_path / "release.json"
    monkeypatch.setattr(create_release, "bootstrap", lambda path: context)
    monkeypatch.setattr(create_release, "mlflow", fake_mlflow)
    monkeypatch.setattr(create_release, "source_commit", lambda: SOURCE_COMMIT)
    monkeypatch.setattr(create_release, "source_state", lambda: "clean")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "create_release.py",
            "--prompt-version",
            "7",
            "--evaluation-run",
            "run-1",
            "--knowledge-version",
            "knowledge-v002",
            "--output",
            str(output),
        ],
    )

    create_release.main()

    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["schema_version"] == "2"
    assert document["world"]["knowledge_version"] == "knowledge-v002"
    assert document["world"]["retrieval_index"] == ("catalog.schema.knowledge_v002")
    assert document["evaluation"]["dataset_id"] == "dataset-1"
    assert document["evaluation"]["dataset_digest"] == "dataset-v1"
    assert document["prompt"]["uri"] == f"prompts:/{PROMPT_NAME}/7"
    assert len(document["retrieval"]["rag_configuration_sha256"]) == 64
    assert document["retrieval"]["embedding"] == "knowledge-embedding"
    assert document["retrieval"]["embedding_configuration"]["dimensions"] == 1536
    assert document["control"]["rag_limits_digest"] == DEFAULT_RAG_LIMITS.digest
    assert document["control"]["max_output_tokens"] == 1024
    assert set(document["clock_digests"]) == {"world", "learning", "control"}
