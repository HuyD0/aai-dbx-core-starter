"""Compact model contract, MLX rendering, and capstone evaluation."""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import Field, ValidationError, model_validator

from .schemas import (
    CapstoneRecord,
    CheckOutcome,
    ReadinessReview,
    ReadinessStatus,
    Severity,
    StrictFrozenModel,
)

CAPSTONE_OUTPUT_CONTRACT_VERSION = "1.0.0"
CAPSTONE_SYSTEM_PROMPT = (
    "Review the application manifest. Return one JSON object only with status "
    "and checks. Include only non-pass checks. Every check must contain exactly "
    "name, result, severity, and remediation_id. Do not invent external facts."
)


class CompactReadinessCheck(StrictFrozenModel):
    """One actionable, non-pass check exposed to the tiny model."""

    name: str = Field(min_length=1)
    result: CheckOutcome
    severity: Severity
    remediation_id: str | None = None

    @model_validator(mode="after")
    def _reject_pass_checks(self) -> CompactReadinessCheck:
        if self.result is CheckOutcome.PASS:
            raise ValueError("compact output must omit pass checks")
        return self


class CompactReadinessReview(StrictFrozenModel):
    """Bounded model-facing projection of the full deterministic review."""

    schema_version: str = CAPSTONE_OUTPUT_CONTRACT_VERSION
    status: ReadinessStatus
    checks: tuple[CompactReadinessCheck, ...]

    @model_validator(mode="after")
    def _require_unique_checks(self) -> CompactReadinessReview:
        names = [check.name for check in self.checks]
        if len(names) != len(set(names)):
            raise ValueError("compact checks must have unique names")
        return self


class CapstonePrediction(StrictFrozenModel):
    """Persistable local model output and performance evidence."""

    example_id: str = Field(pattern=r"^capstone-[a-f0-9]{16}$")
    raw_text: str
    latency_ms: float = Field(ge=0.0)
    output_tokens: int = Field(ge=0)
    peak_memory_mb: float = Field(ge=0.0)


class CapstoneScoreMetrics(StrictFrozenModel):
    count: int = Field(ge=1)
    json_parse_rate: float = Field(ge=0.0, le=1.0)
    schema_validity_rate: float = Field(ge=0.0, le=1.0)
    status_accuracy: float = Field(ge=0.0, le=1.0)
    check_result_accuracy: float = Field(ge=0.0, le=1.0)
    check_severity_accuracy: float = Field(ge=0.0, le=1.0)
    missing_check_rate: float = Field(ge=0.0, le=1.0)
    extra_check_rate: float = Field(ge=0.0, le=1.0)
    exact_review_rate: float = Field(ge=0.0, le=1.0)


class CapstonePerformance(StrictFrozenModel):
    mean_latency_ms: float = Field(ge=0.0)
    p95_latency_ms: float = Field(ge=0.0)
    mean_output_tokens: float = Field(ge=0.0)
    maximum_peak_memory_mb: float = Field(ge=0.0)


class CapstoneErrorExample(StrictFrozenModel):
    example_id: str
    slices: tuple[str, ...]
    issues: tuple[str, ...]
    output_preview: str


class CapstoneErrorAnalysis(StrictFrozenModel):
    total_errors: int = Field(ge=0)
    examples: tuple[CapstoneErrorExample, ...]
    truncated: bool


class CapstoneEvaluationReport(StrictFrozenModel):
    total_examples: int = Field(ge=1)
    evaluation_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_manifest_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    aggregate: CapstoneScoreMetrics
    by_slice: dict[str, CapstoneScoreMetrics]
    performance: CapstonePerformance
    error_analysis: CapstoneErrorAnalysis

    def flat_metrics(self) -> dict[str, float]:
        return {
            "capstone/json_parse_rate": self.aggregate.json_parse_rate,
            "capstone/schema_validity_rate": self.aggregate.schema_validity_rate,
            "capstone/status_accuracy": self.aggregate.status_accuracy,
            "capstone/check_result_accuracy": self.aggregate.check_result_accuracy,
            "capstone/check_severity_accuracy": (
                self.aggregate.check_severity_accuracy
            ),
            "capstone/missing_check_rate": self.aggregate.missing_check_rate,
            "capstone/extra_check_rate": self.aggregate.extra_check_rate,
            "capstone/exact_review_rate": self.aggregate.exact_review_rate,
            "capstone/latency_mean_ms": self.performance.mean_latency_ms,
            "capstone/latency_p95_ms": self.performance.p95_latency_ms,
            "capstone/output_tokens_mean": self.performance.mean_output_tokens,
            "capstone/memory_peak_max_mb": (self.performance.maximum_peak_memory_mb),
        }


class CapstoneTrainingArtifact(StrictFrozenModel):
    path: str
    record_count: int = Field(ge=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CapstoneTrainingManifest(StrictFrozenModel):
    schema_version: str = "1.0.0"
    source_dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_contract_version: str = CAPSTONE_OUTPUT_CONTRACT_VERSION
    system_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    splits: dict[str, CapstoneTrainingArtifact]


class _Scored:
    def __init__(
        self,
        *,
        record: CapstoneRecord,
        prediction: CapstonePrediction,
        parsed: bool,
        output: CompactReadinessReview | None,
        issues: tuple[str, ...],
    ) -> None:
        self.record = record
        self.prediction = prediction
        self.parsed = parsed
        self.output = output
        self.issues = issues


def compact_review(review: ReadinessReview) -> CompactReadinessReview:
    """Project a full deterministic review to the bounded model contract."""

    checks = tuple(
        CompactReadinessCheck(
            name=check.name,
            result=check.result,
            severity=check.severity,
            remediation_id=check.remediation_id,
        )
        for check in review.checks
        if check.result is not CheckOutcome.PASS
    )
    return CompactReadinessReview(
        status=review.status,
        checks=checks,
    )


def compact_expected(record: CapstoneRecord) -> CompactReadinessReview:
    """Project deterministic ground truth to the bounded model contract."""

    return compact_review(record.expected_output)


def load_capstone_records(path: str | Path) -> tuple[CapstoneRecord, ...]:
    source = Path(path)
    records: list[CapstoneRecord] = []
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            records.append(CapstoneRecord.model_validate_json(line))
        except ValidationError as error:
            raise ValueError(
                f"invalid capstone record at {source}:{line_number}"
            ) from error
    if not records:
        raise ValueError(f"no capstone records found in {source}")
    return tuple(records)


def render_capstone_mlx_dataset(
    source_dir: str | Path,
    output_dir: str | Path,
) -> CapstoneTrainingManifest:
    """Render generated policy data to portable MLX conversational JSONL."""

    source = Path(source_dir)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    names = {
        "train": ("train.jsonl", "train.jsonl"),
        "validation": ("validation.jsonl", "valid.jsonl"),
        "test": ("test.jsonl", "test.jsonl"),
    }
    artifacts: dict[str, CapstoneTrainingArtifact] = {}
    fingerprint = hashlib.sha256()
    source_hash = hashlib.sha256()
    for split, (source_name, output_name) in names.items():
        source_bytes = (source / source_name).read_bytes()
        source_hash.update(split.encode())
        source_hash.update(b"\0")
        source_hash.update(source_bytes)
        records = load_capstone_records(source / source_name)
        lines = []
        for record in records:
            expected = compact_expected(record)
            payload = {
                "example_id": record.example_id,
                "source_dataset": record.source_dataset,
                "source_version": record.source_version,
                "messages": [
                    {"role": "system", "content": CAPSTONE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": _canonical_json(record.manifest),
                    },
                    {
                        "role": "assistant",
                        "content": expected.model_dump_json(),
                    },
                ],
                "metadata": {
                    "split": record.metadata.split.value,
                    "slices": list(record.metadata.slices),
                    "policy_version": record.metadata.policy_version,
                },
            }
            lines.append(_canonical_json(payload))
            fingerprint.update(record.example_id.encode())
            fingerprint.update(b"\0")
            fingerprint.update(expected.model_dump_json().encode())
            fingerprint.update(b"\0")
        content = ("\n".join(lines) + "\n").encode()
        output_path = destination / output_name
        output_path.write_bytes(content)
        artifacts[split] = CapstoneTrainingArtifact(
            path=output_name,
            record_count=len(records),
            sha256=hashlib.sha256(content).hexdigest(),
        )
    manifest = CapstoneTrainingManifest(
        source_dataset_sha256=source_hash.hexdigest(),
        system_prompt_sha256=hashlib.sha256(
            CAPSTONE_SYSTEM_PROMPT.encode()
        ).hexdigest(),
        dataset_fingerprint=fingerprint.hexdigest(),
        splits=artifacts,
    )
    (destination / "manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def deterministic_capstone_predictions(
    records: Sequence[CapstoneRecord],
) -> tuple[CapstonePrediction, ...]:
    """Serialize the policy ceiling through the same prediction interface."""

    predictions = []
    for record in records:
        started = time.perf_counter()
        raw_text = compact_expected(record).model_dump_json()
        predictions.append(
            CapstonePrediction(
                example_id=record.example_id,
                raw_text=raw_text,
                latency_ms=(time.perf_counter() - started) * 1000,
                output_tokens=0,
                peak_memory_mb=0.0,
            )
        )
    return tuple(predictions)


def evaluate_capstone_predictions(
    records: Sequence[CapstoneRecord],
    predictions: Sequence[CapstonePrediction],
    *,
    error_limit: int = 20,
) -> CapstoneEvaluationReport:
    if not records:
        raise ValueError("records must not be empty")
    if error_limit < 0:
        raise ValueError("error_limit must be non-negative")
    ordered = _align(records, predictions)
    scored = tuple(
        _score(record, prediction)
        for record, prediction in zip(records, ordered, strict=True)
    )
    aggregate = _metrics(scored)
    by_slice_items: dict[str, list[_Scored]] = defaultdict(list)
    for item in scored:
        for slice_name in item.record.metadata.slices:
            by_slice_items[slice_name].append(item)
    errors = [item for item in scored if item.issues]
    return CapstoneEvaluationReport(
        total_examples=len(scored),
        evaluation_fingerprint=_fingerprint(records),
        aggregate=aggregate,
        by_slice={
            name: _metrics(tuple(items))
            for name, items in sorted(by_slice_items.items())
        },
        performance=CapstonePerformance(
            mean_latency_ms=statistics.fmean(
                item.prediction.latency_ms for item in scored
            ),
            p95_latency_ms=_p95([item.prediction.latency_ms for item in scored]),
            mean_output_tokens=statistics.fmean(
                item.prediction.output_tokens for item in scored
            ),
            maximum_peak_memory_mb=max(
                item.prediction.peak_memory_mb for item in scored
            ),
        ),
        error_analysis=CapstoneErrorAnalysis(
            total_errors=len(errors),
            examples=tuple(
                CapstoneErrorExample(
                    example_id=item.record.example_id,
                    slices=item.record.metadata.slices,
                    issues=item.issues,
                    output_preview=item.prediction.raw_text[:240],
                )
                for item in errors[:error_limit]
            ),
            truncated=len(errors) > error_limit,
        ),
    )


def _score(record: CapstoneRecord, prediction: CapstonePrediction) -> _Scored:
    parsed = False
    try:
        json.loads(prediction.raw_text)
        parsed = True
    except (json.JSONDecodeError, TypeError):
        pass
    output = None
    if parsed:
        try:
            output = CompactReadinessReview.model_validate_json(prediction.raw_text)
        except ValidationError:
            pass
    expected = compact_expected(record)
    issues: list[str] = []
    if not parsed:
        issues.append("json_parse")
    elif output is None:
        issues.append("schema")
    else:
        if output.status is not expected.status:
            issues.append("status")
        expected_by_name = {check.name: check for check in expected.checks}
        output_by_name = {check.name: check for check in output.checks}
        missing = sorted(set(expected_by_name) - set(output_by_name))
        extra = sorted(set(output_by_name) - set(expected_by_name))
        if missing:
            issues.append("missing_checks:" + ",".join(missing))
        if extra:
            issues.append("extra_checks:" + ",".join(extra))
        if any(
            output_by_name[name].result is not expected_by_name[name].result
            for name in set(expected_by_name) & set(output_by_name)
        ):
            issues.append("check_result")
        if any(
            output_by_name[name].severity is not expected_by_name[name].severity
            for name in set(expected_by_name) & set(output_by_name)
        ):
            issues.append("check_severity")
        if any(
            output_by_name[name].remediation_id != expected_by_name[name].remediation_id
            for name in set(expected_by_name) & set(output_by_name)
        ):
            issues.append("remediation_id")
    return _Scored(
        record=record,
        prediction=prediction,
        parsed=parsed,
        output=output,
        issues=tuple(issues),
    )


def _metrics(items: Sequence[_Scored]) -> CapstoneScoreMetrics:
    parse_count = sum(item.parsed for item in items)
    valid_count = sum(item.output is not None for item in items)
    status_correct = 0
    result_correct = 0
    severity_correct = 0
    check_denominator = 0
    expected_total = 0
    predicted_total = 0
    missing_total = 0
    extra_total = 0
    exact = 0
    for item in items:
        expected = compact_expected(item.record)
        output = item.output
        if output is not None and output.status is expected.status:
            status_correct += 1
        expected_by_name = {check.name: check for check in expected.checks}
        output_by_name = (
            {check.name: check for check in output.checks} if output is not None else {}
        )
        union = set(expected_by_name) | set(output_by_name)
        denominator = max(1, len(union))
        check_denominator += denominator
        for name in union:
            expected_check = expected_by_name.get(name)
            output_check = output_by_name.get(name)
            if (
                expected_check is not None
                and output_check is not None
                and expected_check.result is output_check.result
            ):
                result_correct += 1
            if (
                expected_check is not None
                and output_check is not None
                and expected_check.severity is output_check.severity
            ):
                severity_correct += 1
        if not union and output is not None:
            result_correct += 1
            severity_correct += 1
        expected_total += len(expected_by_name)
        predicted_total += len(output_by_name)
        missing_total += len(set(expected_by_name) - set(output_by_name))
        extra_total += len(set(output_by_name) - set(expected_by_name))
        if output == expected:
            exact += 1
    count = len(items)
    return CapstoneScoreMetrics(
        count=count,
        json_parse_rate=parse_count / count,
        schema_validity_rate=valid_count / count,
        status_accuracy=status_correct / count,
        check_result_accuracy=result_correct / check_denominator,
        check_severity_accuracy=severity_correct / check_denominator,
        missing_check_rate=missing_total / max(1, expected_total),
        extra_check_rate=extra_total / max(1, predicted_total),
        exact_review_rate=exact / count,
    )


def _align(
    records: Sequence[CapstoneRecord],
    predictions: Sequence[CapstonePrediction],
) -> tuple[CapstonePrediction, ...]:
    record_ids = [record.example_id for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("records contain duplicate example IDs")
    by_id: dict[str, CapstonePrediction] = {}
    for prediction in predictions:
        if prediction.example_id in by_id:
            raise ValueError("predictions contain duplicate example IDs")
        by_id[prediction.example_id] = prediction
    missing = sorted(set(record_ids) - set(by_id))
    extra = sorted(set(by_id) - set(record_ids))
    if missing or extra:
        raise ValueError(
            f"prediction IDs do not align (missing={missing[:3]}, extra={extra[:3]})"
        )
    return tuple(by_id[example_id] for example_id in record_ids)


def _fingerprint(records: Sequence[CapstoneRecord]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(record.example_id.encode())
        digest.update(b"\0")
        digest.update(compact_expected(record).model_dump_json().encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, int(0.95 * len(ordered) + 0.999999) - 1)]


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
