"""Release evidence joins only the exact code, controls, data, and gate run."""

import json
import sys
from types import SimpleNamespace

import pytest

from aai_core import __version__
from aai_core.experiments import record_reproducibility
from app.controls import DEFAULT_AGENT_LIMITS
from app.tools import build_registry
from evals.evaluate import _evaluation_models
from scripts import create_release

SOURCE_COMMIT = "c" * 40


def _settings():
    return SimpleNamespace(
        effective_experiment_name="/Shared/test-agent-app",
        models={
            "general-chat": {
                "provider": "databricks",
                "deployment": "target-endpoint",
            },
            "judge-model": {
                "provider": "databricks",
                "deployment": "judge-endpoint",
            },
        },
    )


def _dataset(*, experiment_ids=("experiment-1",), digest="dataset-v1"):
    return SimpleNamespace(
        dataset_id="dataset-1",
        digest=digest,
        experiment_ids=experiment_ids,
    )


def _parameters(
    *,
    tool_digest="tool-current",
    dataset_name="catalog.schema.release-suite",
):
    limits = DEFAULT_AGENT_LIMITS
    return {
        "prompt_version": "7",
        "evaluation_dataset": dataset_name,
        "evaluation_dataset_id": "dataset-1",
        "evaluation_dataset_digest": "dataset-v1",
        "target_model": "databricks:target-endpoint",
        "judge_model": "databricks:judge-endpoint",
        "source_commit": SOURCE_COMMIT,
        "source_state": "clean",
        "aai_core_version": __version__,
        "tool_schema_digest": tool_digest,
        "agent_limits_digest": limits.digest,
        "gate_policy_digest": "a" * 64,
        "gate_baseline_digest": "b" * 64,
        **create_release._agent_limit_parameters(limits),
    }


def _run(
    *,
    parameters=None,
    passed="true",
    status="FINISHED",
    experiment_id="experiment-1",
):
    return SimpleNamespace(
        info=SimpleNamespace(status=status, experiment_id=experiment_id),
        data=SimpleNamespace(
            params=parameters or _parameters(),
            tags={"aai.gate_passed": passed},
        ),
    )


def _validate(
    *,
    run=None,
    dataset=None,
    tool_digest="tool-current",
    source_state="clean",
):
    return create_release._validated_gate_evidence(
        run=run or _run(parameters=_parameters(tool_digest=tool_digest)),
        dataset=dataset or _dataset(),
        dataset_name="catalog.schema.release-suite",
        expected_experiment_id="experiment-1",
        prompt_version=7,
        settings=_settings(),
        current_source_commit=SOURCE_COMMIT,
        current_source_state=source_state,
        current_tool_schema_digest=tool_digest,
        current_limits=DEFAULT_AGENT_LIMITS,
    )


def test_validated_gate_evidence_happy_path():
    evidence = _validate()

    assert evidence == {
        "gate_policy_digest": "a" * 64,
        "gate_baseline_digest": "b" * 64,
        "target_model": "databricks:target-endpoint",
        "judge_model": "databricks:judge-endpoint",
    }


def test_failing_gate_and_missing_experiment_association_are_rejected():
    with pytest.raises(RuntimeError, match="passing"):
        _validate(run=_run(passed="false"))
    with pytest.raises(RuntimeError, match="not associated"):
        _validate(dataset=_dataset(experiment_ids=None))


def test_wrong_run_experiment_is_rejected_even_when_dataset_is_associated():
    dataset = _dataset(experiment_ids=("experiment-1", "experiment-2"))
    wrong_run = _run(experiment_id="experiment-2")

    with pytest.raises(RuntimeError, match="configured experiment"):
        _validate(run=wrong_run, dataset=dataset)


@pytest.mark.parametrize(
    "field",
    [
        "prompt_version",
        "evaluation_dataset_digest",
        "target_model",
        "source_commit",
        "source_state",
        "tool_schema_digest",
        "agent_limits_digest",
        "limit_max_output_tokens",
    ],
)
def test_stale_release_join_fields_are_rejected(field):
    parameters = _parameters()
    parameters[field] = "stale"

    with pytest.raises(RuntimeError, match="no longer matches"):
        _validate(run=_run(parameters=parameters))


def test_gate_digests_are_validated():
    parameters = _parameters()
    parameters["gate_policy_digest"] = "not-a-digest"

    with pytest.raises(RuntimeError, match="SHA-256"):
        _validate(run=_run(parameters=parameters))


def test_current_dirty_source_tree_is_rejected():
    with pytest.raises(RuntimeError, match="clean source tree"):
        _validate(source_state="dirty")


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
    with pytest.raises(RuntimeError, match="clean source tree"):
        _validate(source_state="unknown")


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

    evidence = _validate(run=_run(parameters=parameters))
    assert evidence["target_model"] == "databricks:target-endpoint"
    assert create_release.source_commit() == SOURCE_COMMIT
    assert create_release.source_state() == "clean"


@pytest.mark.parametrize(
    "secret_config",
    [
        {"api_key": "sk-do-not-log-this-value"},
        {"headers": {"Authorization": "Bearer do-not-log-this-value"}},
    ],
)
def test_release_model_rejects_inline_secret_material_without_reflecting_it(
    secret_config,
):
    settings = _settings()
    settings.models["general-chat"].update(secret_config)

    with pytest.raises(RuntimeError, match="secret|credential") as error:
        create_release._release_model_config(settings, "general-chat")

    assert "do-not-log-this-value" not in str(error.value)


def test_release_model_only_serializes_reviewed_non_secret_fields():
    settings = _settings()
    settings.models["general-chat"].update(
        {
            "endpoint": "https://runtime.example.invalid",
            "token_scope": "api://runtime/.default",
            "capabilities": {"tool_calling": True},
        }
    )

    assert create_release._release_model_config(settings, "general-chat") == {
        "logical_name": "general-chat",
        "provider": "databricks",
        "deployment": "target-endpoint",
        "capabilities": {"tool_calling": True},
    }


def test_foundry_release_model_uses_endpoint_digest_without_url():
    settings = _settings()
    settings.models["general-chat"].update(
        {
            "provider": "foundry",
            "endpoint": "https://Foundry-A.example.invalid/api/projects/project-a",
        }
    )

    evidence = create_release._release_model_config(settings, "general-chat")

    assert len(evidence["endpoint_sha256"]) == 64
    assert "endpoint" not in evidence
    assert "https://" not in json.dumps(evidence)


def test_foundry_eval_and_release_use_the_same_endpoint_aware_identity():
    settings = _settings()
    settings.models["general-chat"].update(
        {
            "provider": "foundry",
            "endpoint": "https://foundry.example.invalid/api/projects/project-a/",
        }
    )

    _, evaluated_identity, _ = _evaluation_models(settings)

    assert create_release._model_identity(settings, "general-chat") == (
        evaluated_identity
    )


@pytest.mark.parametrize(
    "value",
    [
        "",
        "contains spaces",
        "orders\nprod",
        "x" * 129,
        "sk-do-not-store-this",
        "github_pat_do_not_store_this",
    ],
)
def test_world_version_rejects_high_cardinality_or_secret_values(value):
    with pytest.raises(ValueError, match="bounded non-secret identifier") as error:
        create_release._world_version(value)

    if value:
        assert value not in str(error.value)


def test_world_version_accepts_bounded_operational_snapshot():
    assert (
        create_release._world_version("orders-api:2026-08-08")
        == "orders-api:2026-08-08"
    )


def test_main_writes_v2_world_learning_and_control_evidence(monkeypatch, tmp_path):
    limits = DEFAULT_AGENT_LIMITS
    tools = build_registry(
        timeout_seconds=limits.tool_timeout_seconds,
        max_output_chars=limits.max_tool_output_chars,
    ).openai_tools()
    tool_digest = create_release._canonical_digest(tools)
    dataset = _dataset()
    dataset_name = f"catalog.schema.{create_release.DATASET_NAME}"
    run = _run(
        parameters=_parameters(
            tool_digest=tool_digest,
            dataset_name=dataset_name,
        )
    )
    settings = _settings()
    settings.catalog = "catalog"
    settings.schema = "schema"
    context = SimpleNamespace(
        settings=settings,
        tags=SimpleNamespace(
            application="test-agent-app",
            release="dev",
            environment="dev",
            owner_group="aai-test-owners",
            cost_center="CC-1234",
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
            "--world-version",
            "orders-api:2026-08-08",
            "--output",
            str(output),
        ],
    )

    create_release.main()

    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["schema_version"] == "2"
    assert document["world"] == {"operational_source_version": "orders-api:2026-08-08"}
    assert "dataset" not in document["world"]
    assert document["evaluation"]["dataset_digest"] == "dataset-v1"
    assert document["tools"]["schema_sha256"] == tool_digest
    assert document["control"]["gate_policy_digest"] == "a" * 64
    assert len(document["control"]["manifest_hash"]) == 64
    assert document["control"]["max_stream_output_chars"] == (
        DEFAULT_AGENT_LIMITS.max_stream_output_chars
    )
    assert set(document["clock_digests"]) == {"world", "learning", "control"}
