"""Command-entrypoint tests for prepared dataset integrity gates."""

from __future__ import annotations

from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
from types import SimpleNamespace

import pytest

from aai_local_finetuning import cli, training


@pytest.mark.parametrize(
    ("command", "arguments"),
    (
        (
            cli._cmd_evaluate,
            Namespace(limit=None, max_tokens=0, methods="all", track=False),
        ),
        (
            cli._cmd_capstone_evaluate,
            Namespace(limit=None, max_tokens=-1, methods="all"),
        ),
    ),
)
def test_model_evaluation_requires_a_positive_generation_budget(
    monkeypatch: pytest.MonkeyPatch,
    command: object,
    arguments: Namespace,
) -> None:
    def unexpected_work(*_args: object, **_kwargs: object) -> None:
        pytest.fail("invalid decoding settings reached evaluation work")

    monkeypatch.setattr(cli, "_require_prepared_split_integrity", unexpected_work)
    monkeypatch.setattr(cli, "_require_current_flight_preparation", unexpected_work)

    with pytest.raises(cli.StudyCommandError, match="--max-tokens must be positive"):
        command(arguments, SimpleNamespace())  # type: ignore[operator]


@pytest.mark.parametrize(
    ("command", "arguments"),
    (
        (cli._cmd_baselines, Namespace(track=False)),
        (cli._cmd_train, Namespace(iterations=1)),
        (
            cli._cmd_evaluate,
            Namespace(
                limit=None,
                max_tokens=32,
                methods="all",
                track=False,
            ),
        ),
    ),
)
def test_data_consumers_stop_before_work_when_split_integrity_fails(
    monkeypatch: pytest.MonkeyPatch,
    command: object,
    arguments: Namespace,
) -> None:
    class Settings:
        processed_dir = Path("prepared")

    def reject_integrity(_processed_dir: Path) -> None:
        raise cli.StudyCommandError("prepared dataset integrity check failed")

    def unexpected_work(*_args: object, **_kwargs: object) -> None:
        pytest.fail("command performed work before verifying prepared splits")

    monkeypatch.setattr(cli, "_require_prepared_split_integrity", reject_integrity)
    monkeypatch.setattr(cli, "require_assets", unexpected_work)
    monkeypatch.setattr(cli, "run_lora", unexpected_work)
    monkeypatch.setattr(cli, "_load_splits", unexpected_work)
    monkeypatch.setattr(cli, "_baseline_reports", unexpected_work)

    with pytest.raises(cli.StudyCommandError, match="dataset integrity check failed"):
        command(arguments, Settings())  # type: ignore[operator]


@pytest.mark.parametrize(
    ("command", "arguments"),
    (
        (cli._cmd_train, Namespace(iterations=None)),
        (
            cli._cmd_evaluate,
            Namespace(limit=None, max_tokens=32, methods="all", track=False),
        ),
        (cli._cmd_capstone_train, Namespace(iterations=None)),
        (
            cli._cmd_capstone_evaluate,
            Namespace(limit=None, max_tokens=32, methods="all"),
        ),
    ),
)
def test_promotion_capable_commands_stop_when_flight_source_evidence_is_stale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: object,
    arguments: Namespace,
) -> None:
    class Settings:
        processed_dir = tmp_path / "prepared"

    monkeypatch.setattr(cli, "_require_prepared_split_integrity", lambda _path: None)

    def reject_flight(_settings: object) -> None:
        raise cli.StudyCommandError("flight source evidence is stale")

    def unexpected_work(*_args: object, **_kwargs: object) -> None:
        pytest.fail("command performed work with stale flight source evidence")

    monkeypatch.setattr(cli, "_require_current_flight_preparation", reject_flight)
    monkeypatch.setattr(cli, "require_assets", unexpected_work)
    monkeypatch.setattr(cli, "run_lora", unexpected_work)
    monkeypatch.setattr(cli, "_load_splits", unexpected_work)
    monkeypatch.setattr(cli, "_generate_capstone", unexpected_work)

    with pytest.raises(cli.StudyCommandError, match="flight source evidence is stale"):
        command(arguments, Settings())  # type: ignore[operator]


@pytest.mark.parametrize("failure", ("session", "snapshot"))
def test_support_promotion_file_is_removed_when_post_write_recheck_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    assessment = SimpleNamespace(
        model_dump_json=lambda **_kwargs: "{}",
    )
    snapshot = object()
    session = object()
    decision_path = tmp_path / "promotion.json"

    def recheck_session(_session: object) -> None:
        if failure == "session":
            raise RuntimeError("model session changed after decision")

    monkeypatch.setattr(cli, "recheck_evaluation_session", recheck_session)

    def recheck(_snapshot: object) -> None:
        if failure == "snapshot":
            raise cli.TrainingManifestError("snapshot changed after decision")

    monkeypatch.setattr(cli, "recheck_training_snapshot", recheck)

    with pytest.raises(
        (RuntimeError, cli.StudyCommandError, cli.TrainingManifestError),
        match="changed|package set",
    ):
        cli._write_support_promotion(
            decision_path,
            assessment,  # type: ignore[arg-type]
            snapshot,  # type: ignore[arg-type]
            (session,),  # type: ignore[arg-type]
        )

    assert not decision_path.exists()


@pytest.mark.parametrize("failure", ("session", "snapshot"))
def test_capstone_decision_file_is_removed_when_post_write_recheck_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    decision_session = object()
    decision_path = tmp_path / "decision.json"

    def recheck_session(_session: object) -> None:
        if failure == "session":
            raise RuntimeError("model session changed after decision")

    monkeypatch.setattr(cli, "recheck_evaluation_session", recheck_session)

    def recheck(_snapshot: object) -> None:
        if failure == "snapshot":
            raise cli.TrainingManifestError("snapshot changed after decision")

    monkeypatch.setattr(cli, "recheck_training_snapshot", recheck)

    with pytest.raises(
        (RuntimeError, cli.StudyCommandError, cli.TrainingManifestError),
        match="changed|package set",
    ):
        cli._write_capstone_decision(
            decision_path,
            {"decision": "reject"},
            decision_session,  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
        )

    assert not decision_path.exists()


@pytest.mark.parametrize(
    ("command", "arguments", "adapter_attribute"),
    (
        (
            cli._cmd_evaluate,
            Namespace(limit=None, max_tokens=32, methods="lora", track=False),
            "adapter_dir",
        ),
        (
            cli._cmd_capstone_evaluate,
            Namespace(limit=None, max_tokens=32, methods="lora"),
            "capstone_adapter_dir",
        ),
    ),
)
def test_lora_evaluators_reject_invalid_training_evidence_before_model_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: object,
    arguments: Namespace,
    adapter_attribute: str,
) -> None:
    class Settings:
        processed_dir = tmp_path / "prepared"
        adapter_dir = tmp_path / "adapters" / "support"
        capstone_adapter_dir = tmp_path / "adapters" / "capstone"

    adapter_dir = getattr(Settings, adapter_attribute)
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapters.safetensors").write_bytes(b"stale")

    def reject_manifest(*_args: object, **_kwargs: object) -> None:
        raise cli.TrainingManifestError("missing successful training manifest")

    def unexpected_work(*_args: object, **_kwargs: object) -> None:
        pytest.fail("evaluation performed model or scoring work with stale evidence")

    monkeypatch.setattr(cli, "_require_prepared_split_integrity", lambda _path: None)
    monkeypatch.setattr(cli, "_require_current_flight_preparation", lambda _s: None)
    monkeypatch.setattr(cli, "require_assets", lambda _settings: [])
    monkeypatch.setattr(cli, "_generate_capstone", lambda: None)
    monkeypatch.setattr(cli, "require_valid_training_snapshot", reject_manifest)
    monkeypatch.setattr(cli, "_load_splits", unexpected_work)
    monkeypatch.setattr(cli, "_baseline_reports", unexpected_work)
    monkeypatch.setattr(cli, "load_capstone_records", unexpected_work)
    monkeypatch.setattr(cli, "LocalMLXPredictor", unexpected_work)

    with pytest.raises(cli.StudyCommandError, match="missing, stale, or mismatched"):
        command(arguments, Settings())  # type: ignore[operator]


@pytest.mark.parametrize(
    ("command", "adapter_attribute", "expected_log"),
    (
        (cli._cmd_train, "adapter_dir", "support-smoke"),
        (cli._cmd_capstone_train, "capstone_adapter_dir", "capstone-smoke"),
    ),
)
def test_iteration_overrides_train_into_noncanonical_smoke_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: object,
    adapter_attribute: str,
    expected_log: str,
) -> None:
    class Settings:
        processed_dir = tmp_path / "prepared"
        adapter_dir = tmp_path / "adapters" / "support-v1"
        capstone_adapter_dir = tmp_path / "adapters" / "capstone-v1"

    calls: list[dict[str, object]] = []

    class Evidence:
        @staticmethod
        def model_dump_json(*, indent: int) -> str:
            assert indent == 2
            return "{}"

    def capture_run(**kwargs: object) -> Evidence:
        calls.append(kwargs)
        return Evidence()

    monkeypatch.setattr(cli, "_require_prepared_split_integrity", lambda _path: None)
    monkeypatch.setattr(cli, "_require_current_flight_preparation", lambda _s: None)
    monkeypatch.setattr(cli, "require_assets", lambda _settings: [])
    monkeypatch.setattr(cli, "_generate_capstone", lambda: None)
    monkeypatch.setattr(cli, "run_lora", capture_run)

    command(Namespace(iterations=1), Settings())  # type: ignore[operator]

    canonical = getattr(Settings, adapter_attribute)
    assert calls == [
        {
            "iterations": 1,
            **(
                {
                    "config_path": (
                        cli.PROJECT_ROOT / "configs" / "training" / "capstone-lora.yaml"
                    )
                }
                if command is cli._cmd_capstone_train
                else {}
            ),
            "adapter_path": canonical.with_name(f"{canonical.name}-smoke"),
            "log_name": expected_log,
        }
    ]


def test_support_evaluation_rejects_adapter_change_after_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Settings:
        processed_dir = tmp_path / "prepared"
        adapter_dir = tmp_path / "adapters" / "support"
        model_dir = tmp_path / "model"

    Settings.adapter_dir.mkdir(parents=True)
    (Settings.adapter_dir / "adapters.safetensors").write_bytes(b"adapter")
    snapshot = SimpleNamespace(manifest_sha256="a" * 64)

    monkeypatch.setattr(cli, "_require_prepared_split_integrity", lambda _path: None)
    monkeypatch.setattr(cli, "_require_current_flight_preparation", lambda _s: None)
    monkeypatch.setattr(cli, "require_assets", lambda _settings: [])
    monkeypatch.setattr(cli, "_require_trained_adapter", lambda *_args, **_kw: snapshot)
    monkeypatch.setattr(
        cli, "_load_splits", lambda _settings: ([object()], [], [object()])
    )
    monkeypatch.setattr(cli, "_intent_categories", lambda _records: (["intent"], {}))
    monkeypatch.setattr(cli, "_baseline_reports", lambda *_args, **_kw: {})
    monkeypatch.setattr(cli, "LocalMLXPredictor", lambda *_args, **_kw: object())
    monkeypatch.setattr(cli, "start_evaluation_session", lambda *_args: object())
    monkeypatch.setattr(
        cli,
        "build_local_mlx_inference_config",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(cli, "_model_predictions", lambda *_args, **_kw: ())
    monkeypatch.setattr(cli, "shared_adapter_lock", lambda _path: cli.nullcontext())

    def changed_adapter(_snapshot: object) -> None:
        raise cli.TrainingManifestError("manifest changed after inference")

    def unexpected_score(*_args: object, **_kwargs: object) -> None:
        pytest.fail("changed-adapter predictions were persisted")

    monkeypatch.setattr(cli, "recheck_training_snapshot", changed_adapter)
    monkeypatch.setattr(cli, "_score_predictions", unexpected_score)

    with pytest.raises(cli.TrainingManifestError, match="changed after inference"):
        cli._cmd_evaluate(
            Namespace(limit=1, max_tokens=32, methods="lora", track=False),
            Settings(),  # type: ignore[arg-type]
        )


def test_capstone_evaluation_rejects_adapter_change_after_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Settings:
        capstone_adapter_dir = tmp_path / "adapters" / "capstone"
        model_dir = tmp_path / "model"

    Settings.capstone_adapter_dir.mkdir(parents=True)
    (Settings.capstone_adapter_dir / "adapters.safetensors").write_bytes(b"adapter")
    snapshot = SimpleNamespace(manifest_sha256="b" * 64)

    monkeypatch.setattr(cli, "_require_current_flight_preparation", lambda _s: None)
    monkeypatch.setattr(cli, "require_assets", lambda _settings: [])
    monkeypatch.setattr(cli, "_generate_capstone", lambda: None)
    monkeypatch.setattr(cli, "_require_trained_adapter", lambda *_args, **_kw: snapshot)
    monkeypatch.setattr(cli, "load_capstone_records", lambda _path: [object()])
    monkeypatch.setattr(cli, "LocalMLXPredictor", lambda *_args, **_kw: object())
    monkeypatch.setattr(cli, "start_evaluation_session", lambda *_args: object())
    monkeypatch.setattr(
        cli,
        "build_local_mlx_inference_config",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(cli, "_capstone_model_predictions", lambda *_args, **_kw: ())
    monkeypatch.setattr(cli, "shared_adapter_lock", lambda _path: cli.nullcontext())

    def changed_adapter(_snapshot: object) -> None:
        raise cli.TrainingManifestError("manifest changed after inference")

    def unexpected_score(*_args: object, **_kwargs: object) -> None:
        pytest.fail("changed-adapter predictions were persisted")

    monkeypatch.setattr(cli, "recheck_training_snapshot", changed_adapter)
    monkeypatch.setattr(cli, "evaluate_capstone_predictions", unexpected_score)

    with pytest.raises(cli.TrainingManifestError, match="changed after inference"):
        cli._cmd_capstone_evaluate(
            Namespace(limit=1, max_tokens=32, methods="lora"),
            Settings(),  # type: ignore[arg-type]
        )


def test_clean_study_waits_for_every_cli_adapter_and_preserves_lock_inodes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(training, "PROJECT_ROOT", tmp_path)

    class Settings:
        processed_dir = tmp_path / "data" / "processed" / "bitext-v1"
        adapter_dir = tmp_path / "artifacts" / "adapters" / "support"
        capstone_adapter_dir = tmp_path / "artifacts" / "adapters" / "capstone"
        preflight_adapter_dir = tmp_path / "artifacts" / "adapters" / "preflight"

    settings = Settings()
    adapter_targets = cli._clean_study_adapter_targets(settings)  # type: ignore[arg-type]
    assert len(adapter_targets) == 5
    assert adapter_targets == tuple(
        sorted(adapter_targets, key=lambda path: path.as_posix())
    )
    for adapter in adapter_targets:
        adapter.mkdir(parents=True)
        (adapter / "adapters.safetensors").write_bytes(b"adapter")
    unrelated_artifact = tmp_path / "artifacts" / "evaluation" / "report.json"
    unrelated_artifact.parent.mkdir(parents=True)
    unrelated_artifact.write_text("{}\n", encoding="utf-8")

    shared_locks = {
        adapter: cli.shared_adapter_lock(adapter) for adapter in adapter_targets
    }
    active_locks: set[Path] = set()
    for adapter in reversed(adapter_targets):
        shared_locks[adapter].__enter__()
        active_locks.add(adapter)
    original_inodes = {
        training._adapter_lock_path(adapter): training._adapter_lock_path(adapter)
        .stat()
        .st_ino
        for adapter in adapter_targets
    }

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                cli._cmd_clean_study,
                Namespace(),
                settings,  # type: ignore[arg-type]
            )
            for adapter in adapter_targets[:-1]:
                with pytest.raises(TimeoutError):
                    future.result(timeout=0.15)
                assert unrelated_artifact.is_file()
                shared_locks[adapter].__exit__(None, None, None)
                active_locks.remove(adapter)
            with pytest.raises(TimeoutError):
                future.result(timeout=0.15)
            assert unrelated_artifact.is_file()
            final_adapter = adapter_targets[-1]
            shared_locks[final_adapter].__exit__(None, None, None)
            active_locks.remove(final_adapter)
            future.result(timeout=3)
    finally:
        for adapter in tuple(active_locks):
            shared_locks[adapter].__exit__(None, None, None)

    assert all(not adapter.exists() for adapter in adapter_targets)
    assert not unrelated_artifact.exists()
    for lock_path, original_inode in original_inodes.items():
        assert lock_path.is_file()
        assert lock_path.stat().st_ino == original_inode
