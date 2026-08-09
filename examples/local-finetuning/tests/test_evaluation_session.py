"""Regression tests for inference-wide source/runtime evidence sessions."""

from __future__ import annotations

import hashlib
import json
import os
from argparse import Namespace
from dataclasses import dataclass
from importlib.metadata import PathDistribution
from inspect import Parameter, signature
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from aai_local_finetuning import cli, training
from aai_local_finetuning.capstone import (
    DatasetSplit,
    build_records,
    deterministic_capstone_predictions,
    evaluate_capstone_predictions,
)
from aai_local_finetuning.evaluation import (
    DeterministicInferenceConfig,
    EvaluationRecord,
    Evaluator,
    Prediction,
    SupportOutput,
    evaluate_predictions,
    recheck_evaluation_session,
    start_evaluation_session,
)
from aai_local_finetuning.evaluation import session as evaluation_session_module


@dataclass(frozen=True)
class GovernedRuntime:
    project_root: Path
    source_path: Path
    metadata_path: Path
    package_payload_path: Path
    settings: object
    model_dir: Path
    model_runtime_path: Path
    revision_path: Path


def test_public_scorers_require_an_explicit_prestarted_session() -> None:
    for scorer in (
        Evaluator.evaluate,
        evaluate_predictions,
        evaluate_capstone_predictions,
    ):
        parameter = signature(scorer).parameters["evaluation_session"]
        assert parameter.kind is Parameter.KEYWORD_ONLY
        assert parameter.default is Parameter.empty


def _support_case() -> tuple[EvaluationRecord, Prediction]:
    target = SupportOutput(
        intent="recover_password",
        category="account",
        requires_escalation=False,
        response="I can help you recover access to your account.",
    )
    record = EvaluationRecord(
        example_id="session-support-1",
        input_text="I forgot my password.",
        target=target,
        source_dataset="session-fixture",
        source_version="1.0",
        system_prompt="Return valid JSON only.",
        metadata={"intent": target.intent, "category": target.category},
    )
    prediction = Prediction(
        example_id=record.example_id,
        raw_text=target.model_dump_json(),
        latency_ms=1.0,
        output_tokens=8,
        peak_memory_mb=16.0,
    )
    return record, prediction


@pytest.fixture
def governed_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> GovernedRuntime:
    """Provide the smallest governed tree and one installed distribution."""

    project_root = tmp_path / "project"
    source_path = project_root / "src" / "aai_local_finetuning" / "runtime.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text('"""Stable governed source."""\n', encoding="utf-8")
    scripts_dir = project_root / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "notebook_pedagogy.py").write_text(
        '"""Stable pedagogy source."""\n',
        encoding="utf-8",
    )
    (scripts_dir / "render_notebooks.py").write_text(
        '"""Stable notebook renderer."""\n',
        encoding="utf-8",
    )

    distribution, metadata_path, package_payload_path = _write_distribution(
        tmp_path / "site-packages",
        name="session-runtime",
        version="1.0.0",
    )

    model_dir = project_root / "models" / "tiny-model"
    model_dir.mkdir(parents=True)
    model_runtime_path = model_dir / "model.safetensors"
    model_runtime_path.write_bytes(b"small verified model runtime\n")
    tokenizer_path = model_dir / "tokenizer.json"
    tokenizer_path.write_bytes(b'{"version":"1.0"}\n')
    revision_path = model_dir / "LOCAL_REVISION"
    revision = "a" * 40
    revision_path.write_text(revision + "\n", encoding="utf-8")
    verified_runtime_files = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (model_runtime_path, tokenizer_path)
    }
    settings = SimpleNamespace(
        model=SimpleNamespace(
            repo="local/tiny-model",
            directory="models/tiny-model",
            revision=revision,
            verified_runtime_files=verified_runtime_files,
        )
    )

    monkeypatch.setattr(training, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(
        training.importlib.metadata,
        "distributions",
        lambda: (distribution,),
    )
    return GovernedRuntime(
        project_root=project_root,
        source_path=source_path,
        metadata_path=metadata_path,
        package_payload_path=package_payload_path,
        settings=settings,
        model_dir=model_dir,
        model_runtime_path=model_runtime_path,
        revision_path=revision_path,
    )


def _write_distribution(
    root: Path,
    *,
    name: str,
    version: str,
) -> tuple[PathDistribution, Path, Path]:
    normalized_name = name.replace("-", "_")
    directory_name = f"{normalized_name}-{version}.dist-info"
    distribution_dir = root / directory_name
    distribution_dir.mkdir(parents=True)
    package_dir = root / normalized_name
    package_dir.mkdir()
    package_payload_path = package_dir / "__init__.py"
    package_payload_path.write_text(
        f'"""Runtime payload for {name} {version}."""\n',
        encoding="utf-8",
    )
    metadata_path = distribution_dir / "METADATA"
    metadata_path.write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
        encoding="utf-8",
    )
    (distribution_dir / "RECORD").write_text(
        (
            f"{normalized_name}/__init__.py,,\n"
            f"{directory_name}/METADATA,,\n"
            f"{directory_name}/RECORD,,\n"
        ),
        encoding="utf-8",
    )
    return PathDistribution(distribution_dir), metadata_path, package_payload_path


def _replace_with_original_bytes(path: Path) -> None:
    """Change a file, then restore its bytes through a guaranteed new inode."""

    original = path.read_bytes()
    path.write_bytes(b"temporary drift that inference must not hide\n")
    restored = path.with_name(f".{path.name}.restored")
    restored.write_bytes(original)
    os.replace(restored, path)
    assert path.read_bytes() == original


def test_model_free_session_does_not_claim_or_recheck_model_lineage(
    governed_runtime: GovernedRuntime,
) -> None:
    session = start_evaluation_session()

    assert session.base_model_execution_contract is None
    assert session.base_model_execution_contract_sha256 is None
    _replace_with_original_bytes(governed_runtime.model_runtime_path)
    recheck_evaluation_session(session)


def test_model_session_contract_is_sorted_portable_and_content_addressed(
    governed_runtime: GovernedRuntime,
) -> None:
    session = start_evaluation_session(governed_runtime.settings)  # type: ignore[arg-type]
    contract = session.base_model_execution_contract

    assert contract is not None
    assert contract.repository == "local/tiny-model"
    assert contract.model_path == "models/tiny-model"
    assert contract.model_revision == "a" * 40
    assert tuple(item.path for item in contract.model_files) == (
        "LOCAL_REVISION",
        "model.safetensors",
        "tokenizer.json",
    )
    assert session.base_model_execution_contract_sha256 == (
        training.base_model_execution_contract_sha256(contract)
    )
    portable = json.dumps(contract.model_dump(mode="json"), sort_keys=True)
    assert str(governed_runtime.project_root) not in portable
    assert all(
        name not in portable
        for name in ("device", "inode", "modified_ns", "changed_ns")
    )
    assert repr(session) == "EvaluationSession()"


def test_model_session_rejects_runtime_bytes_outside_the_trusted_contract(
    governed_runtime: GovernedRuntime,
) -> None:
    governed_runtime.model_runtime_path.write_bytes(b"untrusted replacement\n")

    with pytest.raises(ValueError, match="runtime SHA-256 mismatch"):
        start_evaluation_session(governed_runtime.settings)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "relative_path",
    ("chat_template.jinja", "model-extra.safetensors"),
)
def test_model_session_rejects_unverified_loader_visible_files(
    governed_runtime: GovernedRuntime,
    relative_path: str,
) -> None:
    (governed_runtime.model_dir / relative_path).write_bytes(b"unverified runtime\n")

    with pytest.raises(ValueError, match="unverified entries"):
        start_evaluation_session(governed_runtime.settings)  # type: ignore[arg-type]


def test_model_session_rejects_unverified_nested_chat_templates(
    governed_runtime: GovernedRuntime,
) -> None:
    template_dir = governed_runtime.model_dir / "additional_chat_templates"
    template_dir.mkdir()
    (template_dir / "default.jinja").write_text(
        "{{ messages }}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unverified entries"):
        start_evaluation_session(governed_runtime.settings)  # type: ignore[arg-type]


@pytest.mark.parametrize("target", ("runtime", "revision"))
def test_model_session_rejects_file_drift_restored_to_the_same_bytes(
    governed_runtime: GovernedRuntime,
    target: str,
) -> None:
    session = start_evaluation_session(governed_runtime.settings)  # type: ignore[arg-type]
    path = (
        governed_runtime.model_runtime_path
        if target == "runtime"
        else governed_runtime.revision_path
    )

    _replace_with_original_bytes(path)

    with pytest.raises(RuntimeError, match="base-model.*changed"):
        recheck_evaluation_session(session)


def test_model_session_rejects_model_directory_replacement(
    governed_runtime: GovernedRuntime,
) -> None:
    session = start_evaluation_session(governed_runtime.settings)  # type: ignore[arg-type]
    prior_directory = governed_runtime.model_dir.with_name(".prior-model")
    os.replace(governed_runtime.model_dir, prior_directory)
    governed_runtime.model_dir.mkdir()
    for path in prior_directory.iterdir():
        os.replace(path, governed_runtime.model_dir / path.name)
    prior_directory.rmdir()

    with pytest.raises(RuntimeError, match="base-model.*changed"):
        recheck_evaluation_session(session)


def test_model_capture_is_sandwiched_by_execution_identity_rechecks(
    governed_runtime: GovernedRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture_model = evaluation_session_module.capture_base_model_snapshot

    def capture_then_drift(settings: object) -> training.BaseModelSnapshot:
        snapshot = capture_model(settings)  # type: ignore[arg-type]
        governed_runtime.source_path.write_text(
            '"""Drift during model capture."""\n',
            encoding="utf-8",
        )
        return snapshot

    monkeypatch.setattr(
        evaluation_session_module,
        "capture_base_model_snapshot",
        capture_then_drift,
    )

    with pytest.raises(RuntimeError, match="source code.*changed"):
        start_evaluation_session(governed_runtime.settings)  # type: ignore[arg-type]


def test_model_recheck_is_sandwiched_by_execution_identity_rechecks(
    governed_runtime: GovernedRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = start_evaluation_session(governed_runtime.settings)  # type: ignore[arg-type]
    recheck_model = evaluation_session_module.recheck_base_model_snapshot

    def recheck_then_drift(
        snapshot: training.BaseModelSnapshot,
    ) -> training.BaseModelSnapshot:
        checked = recheck_model(snapshot)
        governed_runtime.source_path.write_text(
            '"""Drift during model recheck."""\n',
            encoding="utf-8",
        )
        return checked

    monkeypatch.setattr(
        evaluation_session_module,
        "recheck_base_model_snapshot",
        recheck_then_drift,
    )

    with pytest.raises(RuntimeError, match="source code.*changed"):
        recheck_evaluation_session(session)


def test_prestarted_session_rejects_governed_source_drift_restored_before_scoring(
    governed_runtime: GovernedRuntime,
) -> None:
    record, prediction = _support_case()
    session = start_evaluation_session(governed_runtime.settings)  # type: ignore[arg-type]

    # This stands in for inference importing or rewriting code and restoring the
    # exact original bytes before the scorer starts.
    _replace_with_original_bytes(governed_runtime.source_path)

    with pytest.raises(RuntimeError, match="changed|drift"):
        evaluate_predictions(
            [record],
            [prediction],
            evaluation_session=session,
            inference_config=DeterministicInferenceConfig(method="session-fixture"),
        )


def test_prestarted_session_rejects_package_metadata_drift_restored_before_scoring(
    governed_runtime: GovernedRuntime,
) -> None:
    records = build_records(DatasetSplit.TEST, 1)
    predictions = deterministic_capstone_predictions(records)
    session = start_evaluation_session(governed_runtime.settings)  # type: ignore[arg-type]

    # The final name/version and bytes are unchanged; the session must still
    # notice that installed-package evidence changed during inference.
    _replace_with_original_bytes(governed_runtime.metadata_path)

    with pytest.raises(RuntimeError, match="changed|drift"):
        evaluate_capstone_predictions(
            records,
            predictions,
            evaluation_session=session,
            inference_config=DeterministicInferenceConfig(method="session-fixture"),
        )


def test_prestarted_session_rejects_installed_package_payload_mutation(
    governed_runtime: GovernedRuntime,
) -> None:
    session = start_evaluation_session(governed_runtime.settings)  # type: ignore[arg-type]
    package = session.execution_contract.runtime_packages[0]
    original = governed_runtime.package_payload_path.read_bytes()

    assert package.payload_file_count == 1
    assert package.payload_size_bytes == len(original)
    governed_runtime.package_payload_path.write_bytes(b"x" * len(original))

    with pytest.raises(RuntimeError, match="runtime package files changed"):
        recheck_evaluation_session(session)


def test_support_and_capstone_reports_bind_the_prestarted_session_hash(
    governed_runtime: GovernedRuntime,
) -> None:
    session = start_evaluation_session(governed_runtime.settings)  # type: ignore[arg-type]
    record, prediction = _support_case()
    capstone_records = build_records(DatasetSplit.TEST, 1)

    support_report = evaluate_predictions(
        [record],
        [prediction],
        evaluation_session=session,
        inference_config=DeterministicInferenceConfig(method="session-fixture"),
    )
    capstone_report = evaluate_capstone_predictions(
        capstone_records,
        deterministic_capstone_predictions(capstone_records),
        evaluation_session=session,
        inference_config=DeterministicInferenceConfig(method="session-fixture"),
    )

    assert (
        support_report.evaluation_execution_contract_sha256
        == session.execution_contract_sha256
    )
    assert (
        capstone_report.evaluation_execution_contract_sha256
        == session.execution_contract_sha256
    )


def test_session_preserves_distinct_vendored_versions_with_the_same_name(
    governed_runtime: GovernedRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = PathDistribution(governed_runtime.metadata_path.parent)
    vendored, _vendored_metadata, _vendored_payload = _write_distribution(
        governed_runtime.metadata_path.parents[2] / "provider" / "_vendor",
        name="session-runtime",
        version="0.9.0",
    )
    monkeypatch.setattr(
        training.importlib.metadata,
        "distributions",
        lambda: (primary, primary, vendored),
    )

    session = start_evaluation_session(governed_runtime.settings)  # type: ignore[arg-type]

    assert tuple(
        (package.name, package.version)
        for package in session.execution_contract.runtime_packages
    ) == (
        ("session-runtime", "0.9.0"),
        ("session-runtime", "1.0.0"),
    )
    recheck_evaluation_session(session)

    monkeypatch.setattr(
        training.importlib.metadata,
        "distributions",
        lambda: (primary,),
    )
    with pytest.raises(RuntimeError, match="changed"):
        recheck_evaluation_session(session)


@pytest.mark.parametrize("kind", ("support", "capstone"))
def test_cli_removes_prediction_and_report_after_post_persistence_session_failure(
    governed_runtime: GovernedRuntime,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    monkeypatch.setattr(cli, "PROJECT_ROOT", governed_runtime.project_root)
    session = start_evaluation_session(governed_runtime.settings)  # type: ignore[arg-type]
    if kind == "support":
        record, prediction = _support_case()
        name = "session-support"
        output_dir = governed_runtime.project_root / "artifacts" / "evaluation"
        invoke = lambda: cli._score_predictions(  # noqa: E731
            name=name,
            records=[record],
            predictions=[prediction],
            supported_intents=[record.target.intent],
            evaluation_session=session,
            inference_config=DeterministicInferenceConfig(method=name),
        )
    else:
        records = build_records(DatasetSplit.TEST, 1)
        predictions = deterministic_capstone_predictions(records)
        name = "session-capstone"
        output_dir = governed_runtime.project_root / "artifacts" / "capstone-evaluation"
        invoke = lambda: cli._score_capstone_predictions(  # noqa: E731
            name=name,
            records=records,
            predictions=predictions,
            evaluation_session=session,
            inference_config=DeterministicInferenceConfig(method=name),
        )
    prediction_path = output_dir / f"{name}-predictions.jsonl"
    report_path = output_dir / f"{name}-report.json"
    observed_persisted_evidence = False

    def reject_after_persistence(_session: object) -> None:
        nonlocal observed_persisted_evidence
        assert prediction_path.is_file()
        assert report_path.is_file()
        observed_persisted_evidence = True
        raise RuntimeError("post-persistence evaluation-session drift")

    monkeypatch.setattr(cli, "recheck_evaluation_session", reject_after_persistence)

    with pytest.raises(RuntimeError, match="post-persistence"):
        invoke()

    assert observed_persisted_evidence
    assert not prediction_path.exists()
    assert not report_path.exists()


@pytest.mark.parametrize("kind", ("support", "capstone"))
def test_cli_revalidates_training_lineage_before_persistence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    session = start_evaluation_session()
    invalid_snapshot = SimpleNamespace(
        manifest_sha256="a" * 64,
        manifest=SimpleNamespace(execution_contract_sha256="not-a-sha256"),
    )
    if kind == "support":
        record, prediction = _support_case()
        invoke = lambda: cli._score_predictions(  # noqa: E731
            name="validated-lineage",
            records=[record],
            predictions=[prediction],
            supported_intents=[record.target.intent],
            evaluation_session=session,
            inference_config=DeterministicInferenceConfig(method="fixture"),
            training_snapshot=invalid_snapshot,  # type: ignore[arg-type]
        )
    else:
        records = build_records(DatasetSplit.TEST, 1)
        invoke = lambda: cli._score_capstone_predictions(  # noqa: E731
            name="validated-lineage",
            records=records,
            predictions=deterministic_capstone_predictions(records),
            evaluation_session=session,
            inference_config=DeterministicInferenceConfig(method="fixture"),
            training_snapshot=invalid_snapshot,  # type: ignore[arg-type]
        )

    with pytest.raises(ValidationError):
        invoke()

    assert not (tmp_path / "artifacts").exists()


@pytest.mark.parametrize(
    ("methods", "artifact_names"),
    (
        ("basic", ("majority", "keyword-rule", "basic")),
        ("lora", ("majority", "keyword-rule", "lora-change")),
    ),
)
def test_support_cli_invalidates_requested_evidence_before_preconditions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    methods: str,
    artifact_names: tuple[str, ...],
) -> None:
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    output_dir = tmp_path / "artifacts" / "evaluation"
    output_dir.mkdir(parents=True)
    stale_paths = [
        output_dir / f"{name}-{suffix}"
        for name in artifact_names
        for suffix in ("predictions.jsonl", "report.json")
    ]
    promotion_path = output_dir / "promotion.json"
    stale_paths.append(promotion_path)
    for path in stale_paths:
        path.write_text("stale\n", encoding="utf-8")

    class Settings:
        processed_dir = tmp_path / "prepared"

    def reject(_path: Path) -> None:
        assert all(not path.exists() for path in stale_paths)
        raise cli.StudyCommandError("precondition failed after invalidation")

    monkeypatch.setattr(cli, "_require_prepared_split_integrity", reject)

    with pytest.raises(cli.StudyCommandError, match="after invalidation"):
        cli._cmd_evaluate(
            Namespace(limit=1, max_tokens=32, methods=methods, track=False),
            Settings(),  # type: ignore[arg-type]
        )

    assert all(not path.exists() for path in stale_paths)


def test_baseline_rerun_invalidates_prior_promotion_before_preconditions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    promotion_path = tmp_path / "artifacts" / "evaluation" / "promotion.json"
    promotion_path.parent.mkdir(parents=True)
    promotion_path.write_text("stale\n", encoding="utf-8")

    class Settings:
        processed_dir = tmp_path / "prepared"

    def reject(_path: Path) -> None:
        assert not promotion_path.exists()
        raise cli.StudyCommandError("precondition failed after invalidation")

    monkeypatch.setattr(cli, "_require_prepared_split_integrity", reject)

    with pytest.raises(cli.StudyCommandError, match="after invalidation"):
        cli._cmd_baselines(
            Namespace(track=False),
            Settings(),  # type: ignore[arg-type]
        )

    assert not promotion_path.exists()


@pytest.mark.parametrize("method", ("basic", "hybrid", "lora"))
def test_capstone_cli_invalidates_requested_evidence_before_preconditions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> None:
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    output_dir = tmp_path / "artifacts" / "capstone-evaluation"
    output_dir.mkdir(parents=True)
    artifact_name = "capstone-lora-change" if method == "lora" else method
    stale_paths = [
        output_dir / f"{artifact_name}-predictions.jsonl",
        output_dir / f"{artifact_name}-report.json",
    ]
    if method == "hybrid":
        stale_paths.append(output_dir / "hybrid-explanations.json")
    stale_paths.append(output_dir / "decision.json")
    for path in stale_paths:
        path.write_text("stale\n", encoding="utf-8")

    def reject(_settings: object) -> None:
        assert all(not path.exists() for path in stale_paths)
        raise cli.StudyCommandError("precondition failed after invalidation")

    monkeypatch.setattr(cli, "_require_current_flight_preparation", reject)

    with pytest.raises(cli.StudyCommandError, match="after invalidation"):
        cli._cmd_capstone_evaluate(
            Namespace(limit=1, max_tokens=32, methods=method),
            SimpleNamespace(),  # type: ignore[arg-type]
        )

    assert all(not path.exists() for path in stale_paths)


def _support_report_stub() -> SimpleNamespace:
    return SimpleNamespace(
        classification=SimpleNamespace(macro_f1=1.0),
        output_quality=SimpleNamespace(
            json_schema_validity_rate=1.0,
            response_policy_compliance_rate=1.0,
        ),
    )


def _capstone_report_stub() -> SimpleNamespace:
    return SimpleNamespace(
        aggregate=SimpleNamespace(
            exact_review_rate=1.0,
            status_accuracy=1.0,
            check_result_accuracy=1.0,
            schema_validity_rate=1.0,
        )
    )


def test_support_cli_starts_fresh_session_before_each_baseline_prediction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record, prediction = _support_case()
    events: list[tuple[str, object]] = []
    current_session: object | None = None

    class Settings:
        processed_dir = tmp_path / "prepared"

    class Method:
        def predict_many(self, _records: object) -> tuple[Prediction, ...]:
            assert current_session is not None
            events.append(("predict", current_session))
            return (prediction,)

    class Baseline:
        @staticmethod
        def fit(_records: object) -> Method:
            return Method()

    def start() -> object:
        nonlocal current_session
        current_session = object()
        events.append(("start", current_session))
        return current_session

    def score(**kwargs: object) -> SimpleNamespace:
        assert kwargs["evaluation_session"] is current_session
        assert current_session is not None
        events.append(("score", current_session))
        return _support_report_stub()

    monkeypatch.setattr(cli, "_load_splits", lambda _s: ([record], [], [record]))
    monkeypatch.setattr(cli, "MajorityBaseline", Baseline)
    monkeypatch.setattr(cli, "KeywordRuleBaseline", Baseline)
    monkeypatch.setattr(cli, "start_evaluation_session", start)
    monkeypatch.setattr(cli, "_score_predictions", score)

    cli._baseline_reports(Settings(), track=False)  # type: ignore[arg-type]

    assert [event for event, _session in events] == [
        "start",
        "predict",
        "score",
        "start",
        "predict",
        "score",
    ]
    assert events[0][1] is events[1][1] is events[2][1]
    assert events[3][1] is events[4][1] is events[5][1]
    assert events[0][1] is not events[3][1]


def test_support_cli_starts_session_before_model_prediction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record, prediction = _support_case()
    events: list[str] = []
    marker = object()
    inference_config = object()

    class Settings:
        processed_dir = tmp_path / "prepared"
        adapter_dir = tmp_path / "adapter"
        model_dir = tmp_path / "model"

    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cli, "_require_prepared_split_integrity", lambda _p: None)
    monkeypatch.setattr(cli, "_require_current_flight_preparation", lambda _s: None)
    monkeypatch.setattr(cli, "require_assets", lambda _s: [])
    monkeypatch.setattr(cli, "_load_splits", lambda _s: ([record], [], [record]))
    monkeypatch.setattr(cli, "_baseline_reports", lambda *_a, **_kw: {})
    monkeypatch.setattr(cli, "LocalMLXPredictor", lambda _path: object())

    def start(*_args: object) -> object:
        events.append("start")
        return marker

    def predict(*_args: object, **_kwargs: object) -> tuple[Prediction, ...]:
        assert events == ["start"]
        assert _kwargs["inference_config"] is inference_config
        events.append("predict")
        return (prediction,)

    def score(**kwargs: object) -> SimpleNamespace:
        assert kwargs["evaluation_session"] is marker
        assert kwargs["inference_config"] is inference_config
        events.append("score")
        return _support_report_stub()

    def build_config(session: object, **kwargs: object) -> object:
        assert session is marker
        assert kwargs["max_tokens"] == 32
        assert kwargs["method"] == "basic"
        assert kwargs["prompt_recipe"] == "basic"
        return inference_config

    monkeypatch.setattr(cli, "start_evaluation_session", start)
    monkeypatch.setattr(cli, "recheck_evaluation_session", lambda session: session)
    monkeypatch.setattr(
        cli,
        "build_local_mlx_inference_config",
        build_config,
    )
    monkeypatch.setattr(cli, "_model_predictions", predict)
    monkeypatch.setattr(cli, "_score_predictions", score)

    cli._cmd_evaluate(
        Namespace(limit=1, max_tokens=32, methods="basic", track=False),
        Settings(),  # type: ignore[arg-type]
    )

    assert events == ["start", "predict", "score"]


def test_capstone_cli_starts_session_before_model_prediction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = build_records(DatasetSplit.TRAIN, 1)
    records = build_records(DatasetSplit.TEST, 1)
    predictions = deterministic_capstone_predictions(records)
    events: list[str] = []
    inference_config = object()
    marker = SimpleNamespace(
        execution_contract_sha256="a" * 64,
        base_model_execution_contract=None,
    )

    class Settings:
        capstone_adapter_dir = tmp_path / "adapter"
        model_dir = tmp_path / "model"

    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cli, "_require_current_flight_preparation", lambda _s: None)
    monkeypatch.setattr(cli, "require_assets", lambda _s: [])
    monkeypatch.setattr(cli, "_generate_capstone", lambda: None)
    monkeypatch.setattr(
        cli,
        "load_capstone_records",
        lambda path: train if path.name == "train.jsonl" else records,
    )
    monkeypatch.setattr(cli, "LocalMLXPredictor", lambda _path: object())
    monkeypatch.setattr(cli, "_write_capstone_decision", lambda *_a, **_kw: None)

    def start(*_args: object) -> object:
        events.append("start")
        return marker

    def predict(*_args: object, **_kwargs: object) -> tuple[object, ...]:
        assert events == ["start"]
        assert _kwargs["inference_config"] is inference_config
        events.append("predict")
        return predictions

    def score(**kwargs: object) -> SimpleNamespace:
        assert kwargs["evaluation_session"] is marker
        assert kwargs["inference_config"] is inference_config
        events.append("score")
        report = _capstone_report_stub()
        report.inference_config = DeterministicInferenceConfig(method="basic")
        report.evaluation_execution_contract_sha256 = "a" * 64
        return report

    def build_config(session: object, **kwargs: object) -> object:
        assert session is marker
        assert kwargs["max_tokens"] == 32
        assert kwargs["method"] == "basic"
        assert kwargs["prompt_recipe"] == "basic"
        return inference_config

    monkeypatch.setattr(cli, "start_evaluation_session", start)
    monkeypatch.setattr(cli, "recheck_evaluation_session", lambda session: session)
    monkeypatch.setattr(
        cli,
        "build_local_mlx_inference_config",
        build_config,
    )
    monkeypatch.setattr(cli, "_capstone_model_predictions", predict)
    monkeypatch.setattr(cli, "_score_capstone_predictions", score)

    cli._cmd_capstone_evaluate(
        Namespace(limit=1, max_tokens=32, methods="basic"),
        Settings(),  # type: ignore[arg-type]
    )

    assert events == ["start", "predict", "score", "start"]
