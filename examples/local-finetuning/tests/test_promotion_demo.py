"""The committed synthetic fixtures must drive the real promotion gates."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from aai_local_finetuning import training
from aai_local_finetuning.evaluation import (
    PROMOTION_DEMO_METHODS,
    BaselineEvaluation,
    DeterministicInferenceConfig,
    LocalMLXInferenceConfig,
    PromotionDecision,
    PromotionThresholds,
    bind_promotion_demo_reports,
    decide_lora_promotion,
    degrade_schema_validity,
    load_promotion_demo_reports,
)
from aai_local_finetuning.evaluation import promotion as promotion_module

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "promotion-demo"


def _live_base_model() -> training.BaseModelExecutionContract:
    files = (
        training.TrainingFileEvidence(
            path="LOCAL_REVISION", sha256="4" * 64, size_bytes=41
        ),
        training.TrainingFileEvidence(
            path="config.json", sha256="5" * 64, size_bytes=10
        ),
    )
    return training.BaseModelExecutionContract(
        repository="local/live-model",
        model_path="models/live-model",
        model_revision="6" * 40,
        model_files=files,
        model_files_sha256=training._evidence_sequence_sha256(files),
    )


def _bound_demo_world():
    base_model = _live_base_model()
    live_execution = "a" * 64
    manifest_sha256 = "b" * 64
    training_execution = "c" * 64
    reports = bind_promotion_demo_reports(
        load_promotion_demo_reports(FIXTURE_DIR),
        evaluation_execution_contract_sha256=live_execution,
        base_model=base_model,
        training_manifest_sha256=manifest_sha256,
        training_execution_contract_sha256=training_execution,
    )
    session = SimpleNamespace(
        execution_contract_sha256=live_execution,
        base_model_execution_contract=base_model,
    )
    snapshot = SimpleNamespace(
        manifest_sha256=manifest_sha256,
        manifest=SimpleNamespace(
            execution_contract_sha256=training_execution,
            model_path=base_model.model_path,
            model_revision=base_model.model_revision,
            model_files=base_model.model_files,
        ),
    )
    return reports, session, snapshot


def test_fixture_reports_are_synthetic_and_comparable():
    reports = load_promotion_demo_reports(FIXTURE_DIR)

    assert set(reports) == set(PROMOTION_DEMO_METHODS)
    fingerprints = {report.evaluation_fingerprint for report in reports.values()}
    assert len(fingerprints) == 1
    assert {report.total_examples for report in reports.values()} == {6}
    supported = {report.supported_intents for report in reports.values()}
    assert len(supported) == 1
    # The license forbids committing dataset content: every intent is invented.
    assert all(intent.startswith("demo_") for intent in supported.pop())

    for method in ("majority", "keyword-rule"):
        assert isinstance(
            reports[method].inference_config, DeterministicInferenceConfig
        )
    model_configs = {
        method: reports[method].inference_config
        for method in ("basic", "strong", "few_shot", "lora-change")
    }
    assert all(
        isinstance(config, LocalMLXInferenceConfig) for config in model_configs.values()
    )
    assert (
        len({config.generation.model_dump_json() for config in model_configs.values()})
        == 1
    )
    assert model_configs["lora-change"].prompt_recipe == (
        model_configs["strong"].prompt_recipe
    )
    assert model_configs["lora-change"].few_shot_examples == (
        model_configs["strong"].few_shot_examples
    )
    for method in ("basic", "strong", "few_shot"):
        assert model_configs[method].adapter_manifest_sha256 is None
    assert model_configs["lora-change"].adapter_manifest_sha256 is not None
    assert reports["lora-change"].training_manifest_sha256 is not None

    # The intact fixture change passes every absolute gate and the gain gate.
    strongest = max(
        reports[method].classification.macro_f1
        for method in PROMOTION_DEMO_METHODS
        if method not in ("majority", "lora-change")
    )
    change = reports["lora-change"]
    assert change.classification.macro_f1 - strongest >= 0.01
    assert change.output_quality.json_schema_validity_rate >= 0.98
    assert change.output_quality.response_policy_compliance_rate >= 0.95
    assert change.output_quality.unsupported_intent_rate == 0.0


def test_load_reports_names_the_missing_fixture(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="majority-report.json"):
        load_promotion_demo_reports(tmp_path)


def test_bind_rebinds_identity_fields_and_nothing_else():
    raw = load_promotion_demo_reports(FIXTURE_DIR)
    bound, session, snapshot = _bound_demo_world()

    for method, report in bound.items():
        assert report.evaluation_execution_contract_sha256 == "a" * 64
        assert report.evaluation_fingerprint == (raw[method].evaluation_fingerprint)
        assert report.classification == raw[method].classification
        assert report.output_quality == raw[method].output_quality
        if isinstance(report.inference_config, LocalMLXInferenceConfig):
            assert report.inference_config.base_model == (
                session.base_model_execution_contract
            )
    lora = bound["lora-change"]
    assert lora.training_manifest_sha256 == snapshot.manifest_sha256
    assert lora.training_execution_contract_sha256 == (
        snapshot.manifest.execution_contract_sha256
    )
    assert lora.inference_config.adapter_manifest_sha256 == (snapshot.manifest_sha256)

    with pytest.raises(ValueError, match="missing: lora-change"):
        bind_promotion_demo_reports(
            {name: raw[name] for name in raw if name != "lora-change"},
            evaluation_execution_contract_sha256="a" * 64,
            base_model=_live_base_model(),
            training_manifest_sha256="b" * 64,
            training_execution_contract_sha256="c" * 64,
        )


def test_degrade_schema_validity_is_in_memory_only():
    reports = load_promotion_demo_reports(FIXTURE_DIR)
    original = reports["lora-change"]

    degraded = degrade_schema_validity(original, rate=0.5)

    assert degraded.output_quality.json_schema_validity_rate == 0.5
    assert original.output_quality.json_schema_validity_rate >= 0.98
    unchanged = degraded.output_quality.model_dump() | {
        "json_schema_validity_rate": (original.output_quality.json_schema_validity_rate)
    }
    assert unchanged == original.output_quality.model_dump()
    assert degraded.classification == original.classification
    for invalid in (-0.1, 1.5):
        with pytest.raises(ValueError, match="between"):
            degrade_schema_validity(original, rate=invalid)


def test_bound_fixtures_adopt_intact_and_reject_a_degraded_change(
    monkeypatch: pytest.MonkeyPatch,
):
    """The notebook 08 demo path: real gates, adopt then a named reject."""

    reports, session, snapshot = _bound_demo_world()
    monkeypatch.setattr(
        promotion_module, "recheck_evaluation_session", lambda value: value
    )
    monkeypatch.setattr(
        promotion_module, "recheck_training_snapshot", lambda value: value
    )
    baselines = [
        BaselineEvaluation(
            name=name,
            report=reports[name],
            meaningful=name != "majority",
        )
        for name in PROMOTION_DEMO_METHODS
        if name != "lora-change"
    ]
    thresholds = PromotionThresholds()

    intact = decide_lora_promotion(
        change_name="promotion-gate-demo",
        evaluation_session=session,  # type: ignore[arg-type]
        training_snapshot=snapshot,  # type: ignore[arg-type]
        change_report=reports["lora-change"],
        baselines=baselines,
        thresholds=thresholds,
    )
    assert intact.decision is PromotionDecision.ADOPT
    assert intact.baseline is not None
    assert intact.baseline.name == "strong"

    degraded = decide_lora_promotion(
        change_name="promotion-gate-demo-degraded",
        evaluation_session=session,  # type: ignore[arg-type]
        training_snapshot=snapshot,  # type: ignore[arg-type]
        change_report=degrade_schema_validity(reports["lora-change"], rate=0.5),
        baselines=baselines,
        thresholds=thresholds,
    )
    assert degraded.decision is PromotionDecision.REJECT
    assert degraded.result.passes_schema_threshold is False
    assert any("schema validity" in reason for reason in degraded.result.reasons)
