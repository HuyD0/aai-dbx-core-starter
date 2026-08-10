"""Unit tests for the thin experiment and reproducibility boundary."""

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from aai_core import experiments
from aai_core.experiments import (
    ExperimentManager,
    ExperimentRunMetadata,
    RunPurpose,
    record_reproducibility,
)
from aai_core.testing import dev_settings


class FakeMlflow:
    def __init__(self):
        self.params: dict = {}
        self.tags: dict = {}
        self.artifacts: list = []
        self.start_run_arguments: dict = {}

    def set_experiment(self, name):
        self.experiment = name

    def start_run(self, run_name=None, nested=False, description=None):
        self.start_run_arguments = {
            "run_name": run_name,
            "nested": nested,
            "description": description,
        }

        class _Run:
            def __enter__(self):
                return SimpleNamespace(
                    info=SimpleNamespace(run_name=run_name, nested=nested)
                )

            def __exit__(self, *args):
                return False

        return _Run()

    def set_tags(self, tags):
        self.tags.update(tags)

    def log_params(self, params):
        self.params.update(params)

    def log_artifact(self, path, artifact_path=None):
        self.artifacts.append((Path(path).name, artifact_path))


def _manager(fake):
    return ExperimentManager(
        experiment_name="/Shared/test",
        context=dev_settings().resource,
        mlflow_module=fake,
    )


def test_manager_governs_run_context_and_exposes_native_mlflow():
    fake = FakeMlflow()
    manager = _manager(fake)

    with manager.run(
        run_name="prompt-v4-token-reduction",
        description="Compare the shorter prompt against the governed baseline.",
        parameters={"temperature": 0.1},
        tags={"evaluation_tier": "release"},
    ):
        pass

    assert manager.native_client is fake
    assert fake.experiment == "/Shared/test"
    assert fake.params == {"temperature": 0.1}
    assert fake.tags["aai.application"] == "test-app"
    assert fake.tags["aai.evaluation_tier"] == "release"
    assert fake.tags["aai.experiment_name"] == "/Shared/test"
    assert fake.start_run_arguments["description"] == (
        "Compare the shorter prompt against the governed baseline."
    )


def test_manager_refuses_blank_run_description():
    with (
        pytest.raises(ValueError, match="description"),
        _manager(FakeMlflow()).run(
            run_name="blank-description",
            description=" ",
        ),
    ):
        pass


def test_manager_refuses_sensitive_parameters():
    with (
        pytest.raises(ValueError, match="sensitive"),
        _manager(FakeMlflow()).run(
            run_name="unsafe-credential-logging",
            parameters={"vendor_api_key": "do-not-log"},
        ),
    ):
        pass


def test_record_reproducibility_logs_commit_seed_and_freeze(monkeypatch):
    commit = "A" * 40
    monkeypatch.setenv("GIT_COMMIT", commit)
    monkeypatch.setenv("GIT_DIRTY", "false")
    fake = FakeMlflow()

    record = record_reproducibility(seed=7, mlflow_module=fake)

    assert record["source_commit"] == commit.lower()
    assert record["source_state"] == "clean"
    assert record["seed"] == "7"
    assert fake.params["seed"] == "7"
    assert ("requirements-frozen.txt", "reproducibility") in fake.artifacts
    assert fake.tags["aai.environment_digest"] == record["environment_digest"]


def test_record_reproducibility_refuses_sensitive_extras():
    with pytest.raises(ValueError, match="sensitive"):
        record_reproducibility(extra={"api_key": "value"}, mlflow_module=FakeMlflow())


def test_source_state_treats_untracked_files_as_dirty(monkeypatch, tmp_path):
    monkeypatch.delenv("GIT_DIRTY", raising=False)
    subprocess.run(["git", "init", "--quiet"], cwd=tmp_path, check=True)
    (tmp_path / "untracked-evaluation.py").write_text("CASES = []\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert experiments._source_state() == "dirty"


def test_source_state_fails_safe_for_unknown_ci_dirty_value(monkeypatch):
    monkeypatch.setenv("GIT_DIRTY", "tru")

    assert experiments._source_state() == "unknown"


def test_source_commit_rejects_ref_names_and_short_hashes(monkeypatch):
    for value in ("main", "abc123"):
        monkeypatch.setenv("GIT_COMMIT", value)
        with pytest.raises(ValueError, match="full 40- or 64-character"):
            experiments._source_commit()


def test_source_commit_allows_explicit_local_development_marker(monkeypatch):
    monkeypatch.setenv("GIT_COMMIT", "local-dev")

    assert experiments._source_commit() == "local-dev"


def test_run_metadata_records_change_intent_and_baseline_lineage():
    fake = FakeMlflow()
    metadata = ExperimentRunMetadata(
        purpose=RunPurpose.CHANGE,
        change_id="prompt-v4",
        change_summary="Use the shorter grounded-answer prompt.",
        hypothesis="Shorter instructions reduce tokens without hurting quality.",
        baseline_run_id="run-baseline",
    )

    with _manager(fake).run(
        run_name="prompt-v4-grounded-answer-change",
        metadata=metadata,
    ):
        pass

    assert fake.tags["aai.run_purpose"] == "change"
    assert fake.tags["aai.change_id"] == "prompt-v4"
    assert fake.tags["aai.baseline_run_id"] == "run-baseline"


def test_run_metadata_is_a_strict_persisted_contract():
    with pytest.raises(ValidationError):
        ExperimentRunMetadata(
            purpose="change",
            change_id="prompt-v4",
            change_summary="Change the prompt.",
        )
    with pytest.raises(ValidationError):
        ExperimentRunMetadata(
            purpose=RunPurpose.CHANGE,
            change_id="prompt-v4",
            change_summary="Change the prompt.",
            candidate_id="legacy-term",
        )
