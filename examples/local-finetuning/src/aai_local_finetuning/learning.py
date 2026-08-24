"""Small, explicit Python helpers for the narrative notebook curriculum.

The command-line interface remains useful for repeatable automation, but the
notebooks need inspectable Python objects at every stage.  This module exposes
that seam without importing MLflow or MLX until a learner chooses to use them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .data import require_valid_manifest
from .evaluation import (
    EvaluationRecord,
    EvaluationReport,
    LocalMLXInferenceConfig,
    Prediction,
    load_records_jsonl,
)
from .modeling import LocalGeneration, PromptStrategy, build_messages
from .settings import ProjectSettings, load_settings

COMPLETE_EVALUATION_SCOPE = "complete"


class SupportPredictor(Protocol):
    """The local generation behavior used by the teaching helpers."""

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 160,
    ) -> LocalGeneration: ...


@dataclass(frozen=True)
class SupportSplits:
    """In-memory views of the three portable evidence boundaries."""

    train: tuple[EvaluationRecord, ...]
    validation: tuple[EvaluationRecord, ...]
    test: tuple[EvaluationRecord, ...]


def load_support_splits(
    settings: ProjectSettings | None = None,
    *,
    include_test: bool = True,
) -> SupportSplits:
    """Load development records and optionally cross the frozen-test boundary."""

    project = settings or load_settings()
    require_valid_manifest(project.processed_dir)
    paths = {
        "train": project.processed_dir / "train.jsonl",
        "validation": project.processed_dir / "valid.jsonl",
        "test": project.processed_dir / "test.jsonl",
    }
    required = (
        ("train", "validation", "test")
        if include_test
        else (
            "train",
            "validation",
        )
    )
    missing = [str(paths[name]) for name in required if not paths[name].is_file()]
    if missing:
        raise FileNotFoundError(
            "prepared splits are missing; run `make prepare-flight` while online: "
            + ", ".join(missing)
        )
    return SupportSplits(
        train=tuple(load_records_jsonl(paths["train"])),
        validation=tuple(load_records_jsonl(paths["validation"])),
        test=(tuple(load_records_jsonl(paths["test"])) if include_test else ()),
    )


def support_contract(
    train_records: Sequence[EvaluationRecord],
) -> tuple[tuple[str, ...], dict[str, str]]:
    """Derive the supported labels and intent/category mapping from train only."""

    if not train_records:
        raise ValueError("train_records must not be empty")
    categories: dict[str, str] = {}
    for record in train_records:
        intent = record.target.intent
        category = record.target.category
        previous = categories.setdefault(intent, category)
        if previous != category:
            raise ValueError(
                f"training records map intent {intent!r} to multiple categories"
            )
    return tuple(sorted(categories)), dict(sorted(categories.items()))


def _validate_per_intent(per_intent: object) -> int:
    if isinstance(per_intent, bool) or not isinstance(per_intent, int):
        raise ValueError("per_intent must be an integer")
    if per_intent < 1:
        raise ValueError("per_intent must be positive")
    return per_intent


def stratified_evaluation_scope(per_intent: int) -> str:
    """Name the deterministic course-scale evaluation scope for report files."""

    return f"stratified-subsample-{_validate_per_intent(per_intent)}-per-intent"


def stratified_subsample(
    records: Sequence[EvaluationRecord],
    *,
    per_intent: int,
) -> tuple[EvaluationRecord, ...]:
    """Keep the first ``per_intent`` records of every intent, in frozen order.

    The selection is deterministic given the frozen split order: it walks the
    records once and keeps each record until its intent has ``per_intent``
    representatives.  Every intent present in ``records`` therefore keeps
    support, so macro-F1 is defined for each of them, unlike a first-N slice
    that leaves most intents with zero support.
    """

    limit = _validate_per_intent(per_intent)
    taken: dict[str, int] = {}
    selected: list[EvaluationRecord] = []
    for record in records:
        count = taken.get(record.target.intent, 0)
        if count < limit:
            taken[record.target.intent] = count + 1
            selected.append(record)
    return tuple(selected)


def select_few_shots(
    train_records: Sequence[EvaluationRecord],
    *,
    limit: int = 4,
) -> list[tuple[str, dict[str, object]]]:
    """Select deterministic demonstrations from train without touching evaluation."""

    if limit < 1:
        raise ValueError("limit must be positive")
    selected: list[tuple[str, dict[str, object]]] = []
    seen_intents: set[str] = set()
    for record in sorted(train_records, key=lambda item: item.example_id):
        if record.target.intent in seen_intents:
            continue
        selected.append(
            (
                record.input_text,
                record.target.model_dump(mode="json"),
            )
        )
        seen_intents.add(record.target.intent)
        if len(selected) == limit:
            break
    if len(selected) < limit:
        raise ValueError(
            f"only {len(selected)} distinct training intents are available"
        )
    return selected


def generate_support_predictions(
    predictor: SupportPredictor,
    records: Sequence[EvaluationRecord],
    *,
    strategy: PromptStrategy,
    train_records: Sequence[EvaluationRecord],
    inference_config: LocalMLXInferenceConfig,
    few_shot_limit: int = 4,
) -> tuple[Prediction, ...]:
    """Generate measured predictions while keeping prompt evidence train-derived."""

    if not records:
        raise ValueError("records must not be empty")
    allowed_intents, category_by_intent = support_contract(train_records)
    demonstrations = (
        select_few_shots(train_records, limit=few_shot_limit)
        if strategy == "few_shot"
        else None
    )
    if inference_config.prompt_recipe != strategy:
        raise ValueError(
            "inference prompt recipe does not match the requested strategy"
        )
    if inference_config.few_shot_examples != len(demonstrations or ()):
        raise ValueError("inference few-shot count does not match the rendered prompt")
    predictions = []
    for record in records:
        messages = build_messages(
            record.input_text,
            strategy=strategy,
            allowed_intents=list(allowed_intents),
            category_by_intent=category_by_intent,
            few_shot=demonstrations,
        )
        generated = predictor.generate(
            messages,
            max_tokens=inference_config.generation.max_tokens,
        )
        predictions.append(
            Prediction(
                example_id=record.example_id,
                raw_text=generated.text,
                latency_ms=generated.latency_ms,
                output_tokens=generated.output_tokens,
                peak_memory_mb=generated.peak_memory_mb,
            )
        )
    return tuple(predictions)


def report_row(name: str, report: EvaluationReport) -> dict[str, object]:
    """Return the compact comparison row used across evaluation notebooks."""

    return {
        "method": name,
        "examples": report.total_examples,
        "macro_f1": round(report.classification.macro_f1, 3),
        "weighted_f1": round(report.classification.weighted_f1, 3),
        "category_accuracy": round(report.classification.category_accuracy, 3),
        "json_parse_rate": round(report.output_quality.json_parse_rate, 3),
        "schema_validity": round(report.output_quality.json_schema_validity_rate, 3),
        "unsupported_intent_rate": round(
            report.output_quality.unsupported_intent_rate, 3
        ),
        "response_policy": round(
            report.output_quality.response_policy_compliance_rate, 3
        ),
        "mean_latency_ms": round(report.performance.latency_ms.mean, 1),
        "mean_output_tokens": round(report.performance.output_tokens.mean, 1),
        "peak_memory_mb": round(report.performance.peak_memory_mb.maximum, 1),
    }


def load_report(path: str | Path) -> EvaluationReport:
    """Load one persisted evaluation report through its strict schema."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    return EvaluationReport.model_validate_json(source.read_text(encoding="utf-8"))


def report_inventory(directory: str | Path) -> Mapping[str, Path]:
    """List available report artifacts without assuming every experiment ran."""

    source = Path(directory)
    if not source.is_dir():
        return {}
    return {
        path.name.removesuffix("-report.json"): path
        for path in sorted(source.glob("*-report.json"))
    }


@dataclass(frozen=True)
class LoraParameterBudget:
    """Trainable-versus-frozen parameter arithmetic for the pinned LoRA change."""

    total_parameters: int
    lora_trainable_parameters: int
    trainable_fraction: float
    adapted_layers: int
    adapted_projections: tuple[str, ...]
    rank: int


def _positive_config_int(config: Mapping[str, Any], key: str, label: str) -> int:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} {key} must be a positive integer")
    return value


def lora_parameter_budget(
    model_config: Mapping[str, Any],
    lora_config: Mapping[str, Any],
) -> LoraParameterBudget:
    """Compute total and LoRA-trainable parameter counts for the pinned model.

    ``model_config`` is the local checkpoint's ``config.json`` and
    ``lora_config`` is the parsed training YAML.  The arithmetic implements the
    Qwen2 layout used by this course exactly — embeddings, biased q/k/v
    projections, bias-free o/MLP projections, RMSNorms, and tied or untied
    output embeddings — and fails closed on any other architecture rather than
    reporting a plausible wrong number.  Quantization changes storage bytes,
    not the parameter count.
    """

    model_type = model_config.get("model_type")
    if model_type != "qwen2":
        raise ValueError(
            "lora_parameter_budget implements the qwen2 layout only; "
            f"got model_type {model_type!r}"
        )
    hidden = _positive_config_int(model_config, "hidden_size", "model")
    layers = _positive_config_int(model_config, "num_hidden_layers", "model")
    intermediate = _positive_config_int(model_config, "intermediate_size", "model")
    vocabulary = _positive_config_int(model_config, "vocab_size", "model")
    heads = _positive_config_int(model_config, "num_attention_heads", "model")
    kv_heads = (
        _positive_config_int(model_config, "num_key_value_heads", "model")
        if "num_key_value_heads" in model_config
        else heads
    )
    if "head_dim" in model_config:
        head_dim = _positive_config_int(model_config, "head_dim", "model")
    else:
        if hidden % heads:
            raise ValueError(
                "hidden_size is not divisible by num_attention_heads and "
                "config.json provides no head_dim"
            )
        head_dim = hidden // heads
    tied_embeddings = bool(model_config.get("tie_word_embeddings", False))

    attention_dim = heads * head_dim
    kv_dim = kv_heads * head_dim
    per_layer = (
        (hidden * attention_dim + attention_dim)  # q_proj weight + bias
        + 2 * (hidden * kv_dim + kv_dim)  # k_proj and v_proj weight + bias
        + attention_dim * hidden  # o_proj, no bias
        + 3 * hidden * intermediate  # gate_proj, up_proj, down_proj, no bias
        + 2 * hidden  # input and post-attention RMSNorm weights
    )
    total = vocabulary * hidden + layers * per_layer + hidden
    if not tied_embeddings:
        total += vocabulary * hidden

    lora_parameters = lora_config.get("lora_parameters")
    if not isinstance(lora_parameters, Mapping):
        raise ValueError("training configuration must provide lora_parameters")
    rank = lora_parameters.get("rank")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ValueError("lora_parameters.rank must be a positive integer")
    keys = lora_parameters.get("keys")
    if (
        not isinstance(keys, Sequence)
        or isinstance(keys, (str, bytes))
        or not keys
        or any(not isinstance(key, str) for key in keys)
    ):
        raise ValueError("lora_parameters.keys must be a non-empty list of strings")
    adapted_layers = lora_config.get("num_layers")
    if isinstance(adapted_layers, bool) or not isinstance(adapted_layers, int):
        raise ValueError("training configuration num_layers must be an integer")
    if adapted_layers == -1:
        adapted_layers = layers
    if not 1 <= adapted_layers <= layers:
        raise ValueError(
            f"num_layers must be -1 or between 1 and {layers}; got {adapted_layers}"
        )

    projection_dims = {
        "self_attn.q_proj": (hidden, attention_dim),
        "self_attn.k_proj": (hidden, kv_dim),
        "self_attn.v_proj": (hidden, kv_dim),
        "self_attn.o_proj": (attention_dim, hidden),
        "mlp.gate_proj": (hidden, intermediate),
        "mlp.up_proj": (hidden, intermediate),
        "mlp.down_proj": (intermediate, hidden),
    }
    per_layer_trainable = 0
    for key in keys:
        if key not in projection_dims:
            raise ValueError(f"unsupported LoRA projection key: {key!r}")
        in_features, out_features = projection_dims[key]
        # One adapter is two low-rank matrices: (in x rank) and (rank x out).
        per_layer_trainable += rank * (in_features + out_features)
    trainable = adapted_layers * per_layer_trainable

    return LoraParameterBudget(
        total_parameters=total,
        lora_trainable_parameters=trainable,
        trainable_fraction=trainable / total,
        adapted_layers=adapted_layers,
        adapted_projections=tuple(keys),
        rank=rank,
    )
