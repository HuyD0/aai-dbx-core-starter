"""Strict, portable inference configuration evidence."""

from __future__ import annotations

import hashlib
import json
from inspect import Parameter, signature
from types import SimpleNamespace

import pytest
from pydantic import TypeAdapter, ValidationError

from aai_local_finetuning.capstone import (
    DatasetSplit,
    build_records,
    deterministic_capstone_predictions,
    evaluate_capstone_predictions,
)
from aai_local_finetuning.capstone import evaluation as capstone_evaluation
from aai_local_finetuning.evaluation import (
    DeterministicInferenceConfig,
    EvaluationRecord,
    Evaluator,
    GenerationConfig,
    InferenceConfig,
    LocalMLXInferenceConfig,
    Prediction,
    SupportOutput,
    evaluate_predictions,
)
from aai_local_finetuning.evaluation import metrics as evaluation_metrics
from aai_local_finetuning.training import (
    BaseModelExecutionContract,
    TrainingFileEvidence,
)


def _base_model_contract() -> BaseModelExecutionContract:
    model_files = (
        TrainingFileEvidence(
            path="LOCAL_REVISION",
            sha256="a" * 64,
            size_bytes=41,
        ),
        TrainingFileEvidence(
            path="model.safetensors",
            sha256="b" * 64,
            size_bytes=1024,
        ),
    )
    canonical_files = json.dumps(
        [item.model_dump(mode="json") for item in model_files],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return BaseModelExecutionContract(
        repository="example/model",
        model_path="models/example",
        model_revision="c" * 40,
        model_files=model_files,
        model_files_sha256=hashlib.sha256(canonical_files).hexdigest(),
    )


def _local_config(
    base_model: BaseModelExecutionContract | None = None,
) -> LocalMLXInferenceConfig:
    return LocalMLXInferenceConfig(
        method="strong",
        prompt_recipe="support-structured-output-strong-v1",
        generation=GenerationConfig(max_tokens=37),
        base_model=base_model or _base_model_contract(),
    )


def test_public_scorers_require_explicit_inference_config() -> None:
    for scorer in (
        Evaluator.evaluate,
        evaluate_predictions,
        evaluate_capstone_predictions,
    ):
        parameter = signature(scorer).parameters["inference_config"]
        assert parameter.kind is Parameter.KEYWORD_ONLY
        assert parameter.default is Parameter.empty


def test_deterministic_config_rejects_model_generation_fields() -> None:
    config = DeterministicInferenceConfig(method="keyword-rule")

    assert config.model_dump(mode="json") == {
        "schema_version": "1.0.0",
        "mode": "deterministic",
        "method": "keyword-rule",
    }
    with pytest.raises(ValidationError, match="extra_forbidden"):
        DeterministicInferenceConfig.model_validate(
            {
                "mode": "deterministic",
                "method": "keyword-rule",
                "generation": {"max_tokens": 37},
            }
        )


def test_local_mlx_config_persists_complete_decoding_and_model_evidence() -> None:
    config = _local_config()

    payload = config.model_dump(mode="json")
    assert payload["generation"] == {
        "max_tokens": 37,
        "sampler": "greedy_argmax",
        "logits_processors": [],
        "draft_model": False,
        "max_kv_size": None,
        "prefill_step_size": 2048,
        "kv_bits": None,
        "kv_group_size": 64,
        "quantized_kv_start": 0,
    }
    assert payload["base_model"]["model_revision"] == "c" * 40
    assert {item["path"] for item in payload["base_model"]["model_files"]} == {
        "LOCAL_REVISION",
        "model.safetensors",
    }
    assert payload["adapter_manifest_sha256"] is None
    assert payload["few_shot_examples"] == 0


@pytest.mark.parametrize(
    "payload",
    (
        {
            "mode": "local_mlx",
            "method": "strong",
            "prompt_recipe": "support-strong-v1",
            "generation": {"max_tokens": 37},
        },
        {
            "mode": "local_mlx",
            "method": "strong",
            "prompt_recipe": "support-strong-v1",
            "generation": {"max_tokens": 0},
            "base_model": None,
        },
        {"mode": "unknown", "method": "fixture"},
    ),
)
def test_inference_union_rejects_missing_or_invalid_model_evidence(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(InferenceConfig).validate_python(payload)


def test_local_mlx_config_records_adapter_and_few_shot_lineage() -> None:
    config = LocalMLXInferenceConfig(
        method="lora-change",
        prompt_recipe="support-structured-output-few-shot-v1",
        few_shot_examples=4,
        generation=GenerationConfig(max_tokens=257),
        base_model=_base_model_contract(),
        adapter_manifest_sha256="d" * 64,
    )

    assert config.generation.max_tokens == 257
    assert config.few_shot_examples == 4
    assert config.adapter_manifest_sha256 == "d" * 64


def test_support_scorer_rejects_model_config_on_model_free_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = SupportOutput(
        intent="recover_password",
        category="account",
        requires_escalation=False,
        response="I can help recover access.",
    )
    record = EvaluationRecord(
        example_id="config-binding-1",
        input_text="I forgot my password.",
        target=target,
    )
    prediction = Prediction(
        example_id=record.example_id,
        raw_text=target.model_dump_json(),
        latency_ms=1.0,
        output_tokens=4,
        peak_memory_mb=8.0,
    )
    session = SimpleNamespace(
        execution_contract_sha256="e" * 64,
        base_model_execution_contract=None,
    )
    monkeypatch.setattr(
        evaluation_metrics,
        "recheck_evaluation_session",
        lambda _session: None,
    )

    with pytest.raises(ValueError, match="model-aware evaluation session"):
        evaluate_predictions(
            [record],
            [prediction],
            evaluation_session=session,  # type: ignore[arg-type]
            inference_config=_local_config(),
        )


def test_capstone_scorer_rechecks_model_contract_after_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_model = _base_model_contract()
    changed_model = expected_model.model_copy(update={"repository": "changed/model"})

    class Session:
        execution_contract_sha256 = "f" * 64
        reads = 0

        @property
        def base_model_execution_contract(self) -> BaseModelExecutionContract:
            self.reads += 1
            return expected_model if self.reads == 1 else changed_model

    session = Session()
    records = build_records(DatasetSplit.TEST, 1)
    monkeypatch.setattr(
        capstone_evaluation,
        "recheck_evaluation_session",
        lambda _session: None,
    )

    with pytest.raises(ValueError, match="does not match"):
        evaluate_capstone_predictions(
            records,
            deterministic_capstone_predictions(records),
            evaluation_session=session,  # type: ignore[arg-type]
            inference_config=_local_config(expected_model),
        )
    assert session.reads == 2
