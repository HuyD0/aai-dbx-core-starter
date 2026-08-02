"""Regression tests for inference-wide source/runtime evidence sessions."""

from __future__ import annotations

import os
from argparse import Namespace
from importlib.metadata import PathDistribution
from inspect import Parameter, signature
from pathlib import Path
from types import SimpleNamespace

import pytest

from aai_local_finetuning import cli, training
from aai_local_finetuning.capstone import (
    DatasetSplit,
    build_records,
    deterministic_capstone_predictions,
    evaluate_capstone_predictions,
)
from aai_local_finetuning.evaluation import (
    EvaluationRecord,
    Evaluator,
    Prediction,
    SupportOutput,
    evaluate_predictions,
    recheck_evaluation_session,
    start_evaluation_session,
)


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
) -> tuple[Path, Path, Path]:
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

    distribution, metadata_path = _write_distribution(
        tmp_path / "site-packages",
        name="session-runtime",
        version="1.0.0",
    )

    monkeypatch.setattr(training, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(
        training.importlib.metadata,
        "distributions",
        lambda: (distribution,),
    )
    return project_root, source_path, metadata_path


def _write_distribution(
    root: Path,
    *,
    name: str,
    version: str,
) -> tuple[PathDistribution, Path]:
    normalized_name = name.replace("-", "_")
    directory_name = f"{normalized_name}-{version}.dist-info"
    distribution_dir = root / directory_name
    distribution_dir.mkdir(parents=True)
    metadata_path = distribution_dir / "METADATA"
    metadata_path.write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
        encoding="utf-8",
    )
    (distribution_dir / "RECORD").write_text(
        f"{directory_name}/METADATA,,\n{directory_name}/RECORD,,\n",
        encoding="utf-8",
    )
    return PathDistribution(distribution_dir), metadata_path


def _replace_with_original_bytes(path: Path) -> None:
    """Change a file, then restore its bytes through a guaranteed new inode."""

    original = path.read_bytes()
    path.write_bytes(b"temporary drift that inference must not hide\n")
    restored = path.with_name(f".{path.name}.restored")
    restored.write_bytes(original)
    os.replace(restored, path)
    assert path.read_bytes() == original


def test_prestarted_session_rejects_governed_source_drift_restored_before_scoring(
    governed_runtime: tuple[Path, Path, Path],
) -> None:
    _project_root, source_path, _metadata_path = governed_runtime
    record, prediction = _support_case()
    session = start_evaluation_session()

    # This stands in for inference importing or rewriting code and restoring the
    # exact original bytes before the scorer starts.
    _replace_with_original_bytes(source_path)

    with pytest.raises(RuntimeError, match="changed|drift"):
        evaluate_predictions(
            [record],
            [prediction],
            evaluation_session=session,
        )


def test_prestarted_session_rejects_package_metadata_drift_restored_before_scoring(
    governed_runtime: tuple[Path, Path, Path],
) -> None:
    _project_root, _source_path, metadata_path = governed_runtime
    records = build_records(DatasetSplit.TEST, 1)
    predictions = deterministic_capstone_predictions(records)
    session = start_evaluation_session()

    # The final name/version and bytes are unchanged; the session must still
    # notice that installed-package evidence changed during inference.
    _replace_with_original_bytes(metadata_path)

    with pytest.raises(RuntimeError, match="changed|drift"):
        evaluate_capstone_predictions(
            records,
            predictions,
            evaluation_session=session,
        )


def test_support_and_capstone_reports_bind_the_prestarted_session_hash(
    governed_runtime: tuple[Path, Path, Path],
) -> None:
    _project_root, _source_path, _metadata_path = governed_runtime
    session = start_evaluation_session()
    record, prediction = _support_case()
    capstone_records = build_records(DatasetSplit.TEST, 1)

    support_report = evaluate_predictions(
        [record],
        [prediction],
        evaluation_session=session,
    )
    capstone_report = evaluate_capstone_predictions(
        capstone_records,
        deterministic_capstone_predictions(capstone_records),
        evaluation_session=session,
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
    governed_runtime: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _project_root, _source_path, metadata_path = governed_runtime
    primary = PathDistribution(metadata_path.parent)
    vendored, _vendored_metadata = _write_distribution(
        metadata_path.parents[2] / "provider" / "_vendor",
        name="session-runtime",
        version="0.9.0",
    )
    monkeypatch.setattr(
        training.importlib.metadata,
        "distributions",
        lambda: (primary, primary, vendored),
    )

    session = start_evaluation_session()

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
    governed_runtime: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    project_root, _source_path, _metadata_path = governed_runtime
    monkeypatch.setattr(cli, "PROJECT_ROOT", project_root)
    session = start_evaluation_session()
    if kind == "support":
        record, prediction = _support_case()
        name = "session-support"
        output_dir = project_root / "artifacts" / "evaluation"
        invoke = lambda: cli._score_predictions(  # noqa: E731
            name=name,
            records=[record],
            predictions=[prediction],
            supported_intents=[record.target.intent],
            evaluation_session=session,
        )
    else:
        records = build_records(DatasetSplit.TEST, 1)
        predictions = deterministic_capstone_predictions(records)
        name = "session-capstone"
        output_dir = project_root / "artifacts" / "capstone-evaluation"
        invoke = lambda: cli._score_capstone_predictions(  # noqa: E731
            name=name,
            records=records,
            predictions=predictions,
            evaluation_session=session,
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

    def start() -> object:
        events.append("start")
        return marker

    def predict(*_args: object, **_kwargs: object) -> tuple[Prediction, ...]:
        assert events == ["start"]
        events.append("predict")
        return (prediction,)

    def score(**kwargs: object) -> SimpleNamespace:
        assert kwargs["evaluation_session"] is marker
        events.append("score")
        return _support_report_stub()

    monkeypatch.setattr(cli, "start_evaluation_session", start)
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
    marker = object()

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
    monkeypatch.setattr(cli, "capture_execution_contract", lambda: object())
    monkeypatch.setattr(cli, "execution_contract_sha256", lambda _contract: "a" * 64)

    def start() -> object:
        events.append("start")
        return marker

    def predict(*_args: object, **_kwargs: object) -> tuple[object, ...]:
        assert events == ["start"]
        events.append("predict")
        return predictions

    def score(**kwargs: object) -> SimpleNamespace:
        assert kwargs["evaluation_session"] is marker
        events.append("score")
        return _capstone_report_stub()

    monkeypatch.setattr(cli, "start_evaluation_session", start)
    monkeypatch.setattr(cli, "_capstone_model_predictions", predict)
    monkeypatch.setattr(cli, "_score_capstone_predictions", score)

    cli._cmd_capstone_evaluate(
        Namespace(limit=1, max_tokens=32, methods="basic"),
        Settings(),  # type: ignore[arg-type]
    )

    assert events == ["start", "predict", "score"]
