"""Portable JSONL readers and writers for offline evaluation."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .models import EvaluationRecord, EvaluationReport, Prediction, SupportOutput


class EvaluationDataError(ValueError):
    """A portable record or prediction artifact is invalid."""


def parse_portable_record(payload: Mapping[str, Any]) -> EvaluationRecord:
    """Normalize a conversational training record without framework imports.

    The preferred input is the data-pipeline schema: the user request is
    ``messages[1].content`` and the final assistant message is a JSON target.
    Role lookup is retained as a safe fallback for records without a system
    message.
    """

    example_id = payload.get("example_id")
    if not isinstance(example_id, str) or not example_id.strip():
        raise EvaluationDataError("example_id must be a non-blank string")

    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, Sequence) or isinstance(raw_messages, (str, bytes)):
        raise EvaluationDataError(f"{example_id}: messages must be an array")
    messages = [_message(value, example_id=example_id) for value in raw_messages]

    user_message = None
    if len(messages) > 1 and messages[1]["role"] == "user":
        user_message = messages[1]
    if user_message is None:
        user_message = next(
            (message for message in messages if message["role"] == "user"), None
        )
    if user_message is None:
        raise EvaluationDataError(f"{example_id}: no user message was found")

    assistant_messages = [
        message for message in messages if message["role"] == "assistant"
    ]
    if not assistant_messages:
        raise EvaluationDataError(f"{example_id}: no assistant target was found")
    target = _support_output(assistant_messages[-1]["content"], example_id=example_id)

    metadata_value = payload.get("metadata", {})
    if not isinstance(metadata_value, Mapping):
        raise EvaluationDataError(f"{example_id}: metadata must be an object")
    raw_metadata = dict(metadata_value)
    _validate_metadata_label(raw_metadata, "intent", target.intent, example_id)
    _validate_metadata_label(raw_metadata, "category", target.category, example_id)

    flags = _extract_flags(raw_metadata)
    difficulty_value = raw_metadata.get(
        "difficulty", raw_metadata.get("difficulty_level")
    )
    difficulty = (
        difficulty_value.strip()
        if isinstance(difficulty_value, str) and difficulty_value.strip()
        else "unspecified"
    )
    metadata = {
        key: value
        for key, value in raw_metadata.items()
        if key
        not in {
            "flags",
            "slice_flags",
            "quality_flags",
            "difficulty",
            "difficulty_level",
        }
    }
    system_prompt = next(
        (
            str(message["content"])
            for message in messages
            if message["role"] == "system" and isinstance(message["content"], str)
        ),
        None,
    )

    try:
        return EvaluationRecord(
            example_id=example_id,
            input_text=_text_content(user_message["content"], example_id=example_id),
            target=target,
            source_dataset=_optional_string(payload.get("source_dataset")),
            source_version=_optional_string(payload.get("source_version")),
            system_prompt=system_prompt,
            flags=flags,
            difficulty=difficulty,
            metadata=metadata,
        )
    except ValidationError as error:
        raise EvaluationDataError(
            f"{example_id}: invalid evaluation record: {error}"
        ) from error


def load_records_jsonl(path: str | Path) -> list[EvaluationRecord]:
    """Load portable conversational records, reporting the failing line."""

    return [
        parse_portable_record(payload)
        for payload in _read_json_objects(path, artifact="record")
    ]


def write_records_jsonl(
    path: str | Path,
    records: Iterable[EvaluationRecord],
) -> None:
    """Write normalized records back to the shared conversational schema."""

    output: list[dict[str, Any]] = []
    for record in records:
        metadata = dict(record.metadata)
        metadata.update(
            {
                "intent": record.target.intent,
                "category": record.target.category,
            }
        )
        if record.flags:
            metadata["flags"] = list(record.flags)
        if record.difficulty != "unspecified":
            metadata["difficulty"] = record.difficulty
        messages: list[dict[str, str]] = []
        if record.system_prompt is not None:
            messages.append({"role": "system", "content": record.system_prompt})
        messages.extend(
            [
                {"role": "user", "content": record.input_text},
                {
                    "role": "assistant",
                    "content": record.target.model_dump_json(),
                },
            ]
        )
        payload: dict[str, Any] = {
            "example_id": record.example_id,
            "messages": messages,
            "metadata": metadata,
        }
        if record.source_dataset is not None:
            payload["source_dataset"] = record.source_dataset
        if record.source_version is not None:
            payload["source_version"] = record.source_version
        output.append(payload)
    _write_json_objects(path, output)


def parse_prediction(payload: Mapping[str, Any]) -> Prediction:
    """Normalize common local-inference evidence names to the canonical model."""

    example_id = payload.get("example_id")
    raw_text = payload.get("raw_text", payload.get("output", payload.get("text")))
    latency_ms = payload.get("latency_ms")
    if latency_ms is None and "latency_seconds" in payload:
        latency_ms = _number(payload["latency_seconds"], "latency_seconds") * 1000.0
    output_tokens = payload.get("output_tokens", payload.get("generated_tokens"))
    peak_memory_mb = payload.get("peak_memory_mb")
    if peak_memory_mb is None and "peak_memory_gb" in payload:
        peak_memory_mb = _number(payload["peak_memory_gb"], "peak_memory_gb") * 1024.0
    if peak_memory_mb is None and "peak_memory_bytes" in payload:
        peak_memory_mb = _number(payload["peak_memory_bytes"], "peak_memory_bytes") / (
            1024.0 * 1024.0
        )
    try:
        return Prediction(
            example_id=example_id,
            raw_text=raw_text,
            latency_ms=latency_ms,
            output_tokens=output_tokens,
            peak_memory_mb=peak_memory_mb,
        )
    except ValidationError as error:
        display_id = example_id if isinstance(example_id, str) else "<unknown>"
        raise EvaluationDataError(
            f"{display_id}: invalid prediction evidence: {error}"
        ) from error


def load_predictions_jsonl(path: str | Path) -> list[Prediction]:
    return [
        parse_prediction(payload)
        for payload in _read_json_objects(path, artifact="prediction")
    ]


def write_predictions_jsonl(
    path: str | Path,
    predictions: Iterable[Prediction],
) -> None:
    _write_json_objects(
        path,
        (prediction.model_dump(mode="json") for prediction in predictions),
    )


def write_report_json(path: str | Path, report: EvaluationReport) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")


def _message(value: Any, *, example_id: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise EvaluationDataError(f"{example_id}: each message must be an object")
    role = value.get("role")
    if role not in {"system", "user", "assistant"}:
        raise EvaluationDataError(f"{example_id}: unsupported message role {role!r}")
    if "content" not in value:
        raise EvaluationDataError(f"{example_id}: message content is missing")
    return {"role": role, "content": value["content"]}


def _support_output(content: Any, *, example_id: str) -> SupportOutput:
    if isinstance(content, str):
        try:
            value = json.loads(content)
        except json.JSONDecodeError as error:
            raise EvaluationDataError(
                f"{example_id}: assistant target is not valid JSON: {error.msg}"
            ) from error
    else:
        value = content
    try:
        return SupportOutput.model_validate(value, strict=True)
    except ValidationError as error:
        raise EvaluationDataError(
            f"{example_id}: assistant target violates SupportOutput: {error}"
        ) from error


def _text_content(content: Any, *, example_id: str) -> str:
    if isinstance(content, str) and content.strip():
        return content
    raise EvaluationDataError(f"{example_id}: user content must be a non-blank string")


def _extract_flags(metadata: Mapping[str, Any]) -> tuple[str, ...]:
    raw = metadata.get(
        "flags",
        metadata.get("slice_flags", metadata.get("quality_flags", ())),
    )
    if isinstance(raw, str):
        values = [raw]
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        values = list(raw)
    else:
        values = []
    flags = {
        value.strip() for value in values if isinstance(value, str) and value.strip()
    }
    return tuple(sorted(flags))


def _validate_metadata_label(
    metadata: Mapping[str, Any],
    field: str,
    expected: str,
    example_id: str,
) -> None:
    value = metadata.get(field)
    if value is not None and value != expected:
        raise EvaluationDataError(
            f"{example_id}: metadata {field} {value!r} does not match "
            f"assistant target {expected!r}"
        )


def _read_json_objects(path: str | Path, *, artifact: str) -> list[dict[str, Any]]:
    source = Path(path)
    values: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise EvaluationDataError(
                f"{source}:{line_number}: invalid JSON: {error.msg}"
            ) from error
        if not isinstance(value, dict):
            raise EvaluationDataError(
                f"{source}:{line_number}: {artifact} must be a JSON object"
            )
        values.append(value)
    if not values:
        raise EvaluationDataError(f"{source}: no {artifact} objects were found")
    return values


def _write_json_objects(path: str | Path, values: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for value in values
    ]
    target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationDataError(f"{name} must be a number")
    return float(value)
