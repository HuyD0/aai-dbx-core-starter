"""What the run actually cost, normalized by successful completion.

Mean cost per call is the trap metric: it prices attempts while the money
is spent per outcome.  A failed row's spend — including every retry the
trace recorded — still has to be paid for by the rows that succeeded, so
the decision metric is total known spend divided by successful
completions, and the tail (p95) rather than the mean is what reveals a
retry loop re-sending grown context.  This module harvests per-row
operational observations from the traces a scored run already produced,
aggregates them coverage-first ("unknown cost is not zero cost"), and
emits synthetic run-level metrics beside the quality scorers.

Success here is *execution* success: the row produced an answer and its
trace did not end in ERROR.  Judge and scorer failures are deliberately
excluded — they measure the instrument, already fail the gate through
``<scorer>/error_count``, and folding them in would move the cost-per-
success signal every time a judge endpoint flaps.  Quality remains the
gate's own concern; economics says what each outcome cost.

Everything is report-only by default.  A project opts into enforcement
through the ordinary ``thresholds``/``regression_budget`` grammar, so an
economics rule persists in ``policy_rules`` and replays exactly like any
other rule.  No price table ships here: cost is trace-supplied or comes
from a project-configured rate pair, and rows with neither stay unknown.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from aai_core.agentkit._values import is_missing_scalar
from aai_core.agentkit.datasets import _trace_document
from aai_core.agentkit.statistics import _quantile
from aai_core.contracts import ContractModel

_ECONOMICS_PREFIX = "economics/"
# `cost/coverage` deliberately matches the gate engine's default
# ``GatePolicy.cost_coverage_metric`` and the convention the analytics
# template already emits, so one metric name means one thing everywhere.
_COVERAGE_METRICS = frozenset({"cost/coverage", "tokens/coverage"})
_HIGHER_IS_BETTER = frozenset({*_COVERAGE_METRICS, "economics/success_rate"})

# Where pinned MLflow 3.15 records aggregated usage: trace metadata holds
# JSON strings under these keys (`TraceInfo.token_usage`/`.cost` parse
# them), and provider autologging stamps the per-span attributes.
_TRACE_TOKEN_USAGE_KEY = "mlflow.trace.tokenUsage"
_TRACE_COST_KEY = "mlflow.trace.cost"
_SPAN_TOKEN_USAGE_KEY = "mlflow.chat.tokenUsage"
_SPAN_COST_KEY = "mlflow.llm.cost"
_SPAN_INPUT_TOKENS_KEY = "gen_ai.usage.input_tokens"
_SPAN_OUTPUT_TOKENS_KEY = "gen_ai.usage.output_tokens"
_SPAN_TYPE_ATTRIBUTE = "mlflow.spanType"
_SPAN_TYPE_KEYS = ("type", "span_type", "spanType")
_LLM_SPAN_TYPES = frozenset({"LLM", "CHAT_MODEL"})
# Embedding tokens are billed at the embedding model's rate, which is not the
# configured pair. They are excluded from the priced sum rather than from the
# trace: the span still carries them under `gen_ai.usage.input_tokens`.
_UNPRICED_SPAN_TYPES = frozenset({"EMBEDDING"})

# Enough segments to see every intent that matters, few enough that a
# free-text stratum cannot turn the evidence into a row-per-row dump.
_MAX_SEGMENT_VALUES = 20


class EconomicsConfig(ContractModel):
    """Project-owned policy for the run-economics evidence.

    Reporting is on by default and adds no gate rules.  The price pair is
    a project-supplied rate for the *agent's* model — never a shipped
    table — used only for rows whose trace carries token usage but no
    recorded cost.  Input and output are priced separately because a
    retry loop grows input tokens far faster than output tokens.
    """

    enabled: bool = True
    price_per_1m_input_tokens: float | None = Field(default=None, gt=0.0)
    price_per_1m_output_tokens: float | None = Field(default=None, gt=0.0)

    @field_validator(
        "price_per_1m_input_tokens", "price_per_1m_output_tokens", mode="before"
    )
    @classmethod
    def coerce_price(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, int):
            return value
        return float(value)

    @model_validator(mode="after")
    def require_price_pair(self) -> EconomicsConfig:
        # Half a rate silently prices half of every request; a partial
        # pair is a configuration mistake, not a smaller estimate.
        if (self.price_per_1m_input_tokens is None) != (
            self.price_per_1m_output_tokens is None
        ):
            raise ValueError(
                "economics prices come as a pair: set both "
                "price_per_1m_input_tokens and price_per_1m_output_tokens, "
                "or neither"
            )
        return self


class EconomicsSegment(ContractModel):
    """One stratum's economics — the per-intent routing evidence.

    Per-success ratios appear only at complete in-segment cost coverage
    and with at least one success, exactly like the run-level metrics.
    """

    key: str = Field(min_length=1)
    value: str
    rows: int = Field(ge=1)
    successes: int = Field(ge=0)
    success_rate: float = Field(ge=0.0, le=1.0)
    tokens_known: int = Field(ge=0)
    cost_known: int = Field(ge=0)
    duration_known: int = Field(ge=0)
    tokens_total: float | None = None
    cost_total_usd: float | None = None
    tokens_per_success: float | None = None
    cost_per_success_usd: float | None = None
    tokens_p95: float | None = None
    cost_p95_usd: float | None = None
    latency_p95_seconds: float | None = None
    llm_calls_p95: float | None = None


class EconomicsEvidence(ContractModel):
    """Coverage-first spend evidence for one scored run.

    The genai-lifecycle rule is the schema: observation count, known
    count, known subtotal, and coverage travel together, and a total is
    treated as valid only at complete coverage.  The per-row tuples are
    content-free nullable numbers in dataset order — the same
    reproducibility contract ``metric_samples`` follows — kept here
    rather than in ``metric_samples`` so the statistics module does not
    wrap mean confidence intervals around a distribution whose story is
    its tail.
    """

    schema_version: Literal[1] = 1
    rows: int = Field(ge=1)
    successes: int = Field(ge=0)
    tokens_known: int = Field(ge=0)
    cost_known: int = Field(ge=0)
    duration_known: int = Field(ge=0)
    # Subtotals cover the known rows only; ``None`` when nothing is
    # known, because writing 0.0 would turn unknown cost into zero cost.
    tokens_total: float | None = None
    cost_total_usd: float | None = None
    cost_source: Literal["trace", "configured-price", "mixed", "none"] = "none"
    price_per_1m_input_tokens: float | None = None
    price_per_1m_output_tokens: float | None = None
    percentile_method: Literal["linear-interpolation-v1"] = "linear-interpolation-v1"
    segments: tuple[EconomicsSegment, ...] = ()
    input_tokens: tuple[int | None, ...] = ()
    output_tokens: tuple[int | None, ...] = ()
    total_tokens: tuple[int | None, ...] = ()
    cost_usd: tuple[float | None, ...] = ()
    duration_ms: tuple[int | None, ...] = ()
    llm_calls: tuple[int | None, ...] = ()
    success: tuple[bool, ...] = ()

    @field_validator(
        "segments",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cost_usd",
        "duration_ms",
        "llm_calls",
        "success",
        mode="before",
    )
    @classmethod
    def coerce_sequences(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("cost_usd", mode="before")
    @classmethod
    def coerce_costs(cls, value: Any) -> Any:
        # Strict mode refuses an int where a float belongs, and JSON
        # round-trips 2.0 faithfully but a hand-authored 2 arrives int.
        if isinstance(value, (list, tuple)):
            return tuple(
                (
                    float(item)
                    if isinstance(item, int) and not isinstance(item, bool)
                    else item
                )
                for item in value
            )
        return value

    @model_validator(mode="after")
    def per_row_tuples_align_with_rows(self) -> EconomicsEvidence:
        # The tuples are evidence only because their positions are the
        # dataset's rows; a misaligned tuple pins spend to the wrong row.
        for name in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cost_usd",
            "duration_ms",
            "llm_calls",
            "success",
        ):
            values = getattr(self, name)
            if values and len(values) != self.rows:
                raise ValueError(
                    f"{name} must align with rows ({self.rows}) in dataset order"
                )
        return self


def is_economics_metric(metric: str) -> bool:
    return metric.startswith(_ECONOMICS_PREFIX) or metric in _COVERAGE_METRICS


def economics_direction(metric: str) -> str | None:
    """The improvement direction for an economics metric, else ``None``.

    The registry answers "higher" for any metric it does not know, which
    would make a falling cost count as a regression; the gate resolves
    economics metrics here first.
    """

    if not is_economics_metric(metric):
        return None
    return "higher" if metric in _HIGHER_IS_BETTER else "lower"


@dataclass(frozen=True)
class _Observation:
    """One row's operational readings, each ``None`` when unknown."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None
    cost_from_trace: bool = False
    duration_ms: int | None = None
    state_error: bool = False
    llm_calls: int | None = None


def build_economics_evidence(
    rows: Sequence[Mapping[str, Any]],
    traces: Sequence[Any],
    error_flags: Sequence[bool],
    *,
    strata: Sequence[str],
    config: EconomicsConfig,
) -> tuple[EconomicsEvidence | None, dict[str, float], list[str]]:
    """Build persisted evidence and the synthetic metrics consumed by gates.

    ``traces`` and ``error_flags`` are aligned with ``rows`` in dataset
    order; a missing trace simply leaves that row's readings unknown.
    """

    if not config.enabled or not rows:
        return None, {}, []
    observations = [_observe(trace, config) for trace in _aligned(traces, len(rows))]
    flags = list(_aligned(error_flags, len(rows)))
    success = tuple(
        not bool(flag) and not observation.state_error
        for observation, flag in zip(observations, flags, strict=True)
    )

    row_count = len(rows)
    successes = sum(success)
    known_tokens = [o.total_tokens for o in observations if o.total_tokens is not None]
    known_costs = [o.cost_usd for o in observations if o.cost_usd is not None]
    known_durations = [o.duration_ms for o in observations if o.duration_ms is not None]

    metrics: dict[str, float] = {
        "tokens/coverage": len(known_tokens) / row_count,
        "cost/coverage": len(known_costs) / row_count,
        "economics/success_rate": successes / row_count,
    }
    if known_durations:
        seconds = [value / 1000.0 for value in known_durations]
        metrics["economics/latency_p50_seconds"] = _percentile(seconds, 0.5)
        metrics["economics/latency_p95_seconds"] = _percentile(seconds, 0.95)
    if known_tokens:
        token_values = [float(value) for value in known_tokens]
        metrics["economics/tokens_p50"] = _percentile(token_values, 0.5)
        metrics["economics/tokens_p95"] = _percentile(token_values, 0.95)
    if known_costs:
        metrics["economics/cost_p50_usd"] = _percentile(known_costs, 0.5)
        metrics["economics/cost_p95_usd"] = _percentile(known_costs, 0.95)

    warnings: list[str] = []
    tokens_total = float(sum(known_tokens)) if known_tokens else None
    cost_total = float(sum(known_costs)) if known_costs else None
    # The numerator is *all* known spend — failed rows included, because
    # their spend is exactly what the mean-per-call view hides — but the
    # ratio is honest only when every row's spend is known.
    if len(known_tokens) == row_count and successes:
        assert tokens_total is not None
        metrics["economics/tokens_per_success"] = tokens_total / successes
    if len(known_costs) == row_count and successes:
        assert cost_total is not None
        metrics["economics/cost_per_success_usd"] = cost_total / successes
    if not successes:
        warnings.append(
            "no successful completions in this run, so per-success "
            "economics are not reported"
        )
    else:
        for name, known in (
            ("economics/cost_per_success_usd", len(known_costs)),
            ("economics/tokens_per_success", len(known_tokens)),
        ):
            if known < row_count:
                warnings.append(
                    f"{name} is not reported: the spend is known for "
                    f"{known} of {row_count} rows, and a total is valid "
                    "only at complete coverage"
                )

    segments, segment_warnings = _segments(rows, observations, success, strata)
    warnings.extend(segment_warnings)

    evidence = EconomicsEvidence(
        rows=row_count,
        successes=successes,
        tokens_known=len(known_tokens),
        cost_known=len(known_costs),
        duration_known=len(known_durations),
        tokens_total=tokens_total,
        cost_total_usd=cost_total,
        cost_source=_cost_source(observations),
        price_per_1m_input_tokens=config.price_per_1m_input_tokens,
        price_per_1m_output_tokens=config.price_per_1m_output_tokens,
        segments=segments,
        input_tokens=tuple(o.input_tokens for o in observations),
        output_tokens=tuple(o.output_tokens for o in observations),
        total_tokens=tuple(o.total_tokens for o in observations),
        cost_usd=tuple(o.cost_usd for o in observations),
        duration_ms=tuple(o.duration_ms for o in observations),
        llm_calls=tuple(o.llm_calls for o in observations),
        success=success,
    )
    return evidence, metrics, warnings


def _aligned(values: Sequence[Any], count: int) -> Sequence[Any]:
    if len(values) == count:
        return values
    # A frame shorter than the dataset marks the tail unknown rather than
    # shifting readings onto the wrong rows.
    padded = list(values)[:count]
    padded.extend([None] * (count - len(padded)))
    return padded


def _segments(
    rows: Sequence[Mapping[str, Any]],
    observations: Sequence[_Observation],
    success: Sequence[bool],
    strata: Sequence[str],
) -> tuple[tuple[EconomicsSegment, ...], list[str]]:
    segments: list[EconomicsSegment] = []
    warnings: list[str] = []
    for key in strata:
        groups: dict[str, list[int]] = {}
        for index, row in enumerate(rows):
            inputs = row.get("inputs")
            source = inputs if isinstance(inputs, Mapping) else {}
            # The same reading `smoke_sample` stratifies by, so segment
            # economics and stratified sampling agree on what a stratum is.
            groups.setdefault(str(source.get(key, "")), []).append(index)
        ordered = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
        if len(ordered) > _MAX_SEGMENT_VALUES:
            warnings.append(
                f"strata key {key!r} has {len(ordered)} distinct values; "
                f"segment economics reports the {_MAX_SEGMENT_VALUES} "
                "largest"
            )
            ordered = ordered[:_MAX_SEGMENT_VALUES]
        for value, indices in sorted(ordered):
            segments.append(_build_segment(key, value, indices, observations, success))
    return tuple(segments), warnings


def _build_segment(
    key: str,
    value: str,
    indices: Sequence[int],
    observations: Sequence[_Observation],
    success: Sequence[bool],
) -> EconomicsSegment:
    group = [observations[index] for index in indices]
    group_success = sum(1 for index in indices if success[index])
    tokens = [o.total_tokens for o in group if o.total_tokens is not None]
    costs = [o.cost_usd for o in group if o.cost_usd is not None]
    durations = [o.duration_ms for o in group if o.duration_ms is not None]
    calls = [o.llm_calls for o in group if o.llm_calls is not None]
    tokens_total = float(sum(tokens)) if tokens else None
    cost_total = float(sum(costs)) if costs else None
    full_tokens = len(tokens) == len(group) and group_success > 0
    full_costs = len(costs) == len(group) and group_success > 0
    return EconomicsSegment(
        key=key,
        value=value,
        rows=len(group),
        successes=group_success,
        success_rate=group_success / len(group),
        tokens_known=len(tokens),
        cost_known=len(costs),
        duration_known=len(durations),
        tokens_total=tokens_total,
        cost_total_usd=cost_total,
        tokens_per_success=(
            tokens_total / group_success
            if full_tokens and tokens_total is not None
            else None
        ),
        cost_per_success_usd=(
            cost_total / group_success
            if full_costs and cost_total is not None
            else None
        ),
        tokens_p95=(
            _percentile([float(item) for item in tokens], 0.95) if tokens else None
        ),
        cost_p95_usd=_percentile(costs, 0.95) if costs else None,
        latency_p95_seconds=(
            _percentile([item / 1000.0 for item in durations], 0.95)
            if durations
            else None
        ),
        llm_calls_p95=(
            _percentile([float(item) for item in calls], 0.95) if calls else None
        ),
    )


def _cost_source(
    observations: Sequence[_Observation],
) -> Literal["trace", "configured-price", "mixed", "none"]:
    sources: set[Literal["trace", "configured-price"]] = {
        "trace" if observation.cost_from_trace else "configured-price"
        for observation in observations
        if observation.cost_usd is not None
    }
    if not sources:
        return "none"
    if len(sources) > 1:
        return "mixed"
    return next(iter(sources))


def _percentile(values: Sequence[float], quantile: float) -> float:
    """Linear-interpolation percentile — the tail as recorded, not modelled.

    The same ``(n - 1) * q`` positional arithmetic the statistics module
    reads bootstrap bounds with; one quantile definition serves both.
    """

    return _quantile(sorted(values), quantile)


def _observe(trace: Any, config: EconomicsConfig) -> _Observation:
    """One row's readings, from whatever trace form the run produced.

    Preference order per reading: the live ``TraceInfo`` properties, the
    serialized envelope's trace metadata, then a sum over the LLM spans.
    Every reading degrades independently to ``None`` — a trace that
    records duration but no usage still contributes its latency.
    """

    if trace is None or is_missing_scalar(trace):
        return _Observation()
    document = _trace_document(trace)
    info_attr = getattr(trace, "info", None) if not isinstance(trace, Mapping) else None
    info_doc = document.get("info") if isinstance(document, Mapping) else None
    if not isinstance(info_doc, Mapping):
        info_doc = {}
    spans = _document_spans(document)

    usage = _usage_mapping(info_attr, info_doc)
    input_tokens = _token_count(usage.get("input_tokens")) if usage else None
    output_tokens = _token_count(usage.get("output_tokens")) if usage else None
    total_tokens = _token_count(usage.get("total_tokens")) if usage else None
    if input_tokens is None and output_tokens is None and total_tokens is None:
        input_tokens, output_tokens, total_tokens = _span_usage(spans)
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    cost = _cost_mapping(info_attr, info_doc)
    cost_usd = _cost_value(cost) if cost else None
    if cost_usd is None:
        cost_usd = _span_cost(spans)
    cost_from_trace = cost_usd is not None
    if cost_usd is None:
        cost_usd = _configured_cost(input_tokens, output_tokens, config)

    return _Observation(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cost_usd=cost_usd,
        cost_from_trace=cost_from_trace,
        duration_ms=_duration_ms(info_attr, info_doc),
        state_error=_state_is_error(info_attr, info_doc),
        llm_calls=_llm_call_count(spans) if spans else None,
    )


def _document_spans(document: Any) -> list[Mapping[str, Any]]:
    if not isinstance(document, Mapping):
        return []
    data = document.get("data")
    spans: Any = data.get("spans") if isinstance(data, Mapping) else None
    if not isinstance(spans, Sequence) or isinstance(spans, (str, bytes)):
        return []
    return [span for span in spans if isinstance(span, Mapping)]


def _usage_mapping(info_attr: Any, info_doc: Mapping[str, Any]) -> Mapping[str, Any]:
    if info_attr is not None:
        try:
            usage = getattr(info_attr, "token_usage", None)
        except Exception:
            usage = None
        if isinstance(usage, Mapping):
            return usage
    return _metadata_json(info_doc, _TRACE_TOKEN_USAGE_KEY)


def _cost_mapping(info_attr: Any, info_doc: Mapping[str, Any]) -> Mapping[str, Any]:
    if info_attr is not None:
        try:
            cost = getattr(info_attr, "cost", None)
        except Exception:
            cost = None
        if isinstance(cost, Mapping):
            return cost
    return _metadata_json(info_doc, _TRACE_COST_KEY)


def _metadata_json(info_doc: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    # v3 envelopes store trace metadata as a string map whose interesting
    # values are themselves JSON documents; v2 called it request_metadata.
    for field in ("trace_metadata", "request_metadata"):
        metadata = info_doc.get(field)
        if not isinstance(metadata, Mapping):
            continue
        decoded = _decoded(metadata.get(key))
        if isinstance(decoded, Mapping):
            return decoded
    return {}


def _span_usage(
    spans: Sequence[Mapping[str, Any]],
) -> tuple[int | None, int | None, int | None]:
    """Summed LLM-span usage for traces without aggregated metadata.

    Each span contributes its own effective total — the explicit total
    when it recorded one, otherwise the sum of its sides — so a trace
    mixing both span shapes is not under-counted.

    Spans billed at a rate other than the configured pair are skipped. A
    RAG query embeds before it retrieves, and pricing those tokens at the
    agent model's rate would inflate every retrieval row by a model the
    project never priced. A span whose type cannot be read still counts:
    dropping unlabelled spans would silently under-count instead.
    """

    input_total = output_total = total_total = None
    for span in spans:
        if _span_type(span) in _UNPRICED_SPAN_TYPES:
            continue
        attributes = _span_attributes(span)
        usage = _decoded(attributes.get(_SPAN_TOKEN_USAGE_KEY))
        if isinstance(usage, Mapping):
            span_input = _token_count(_decoded(usage.get("input_tokens")))
            span_output = _token_count(_decoded(usage.get("output_tokens")))
            span_total = _token_count(_decoded(usage.get("total_tokens")))
        else:
            span_input = _token_count(_decoded(attributes.get(_SPAN_INPUT_TOKENS_KEY)))
            span_output = _token_count(
                _decoded(attributes.get(_SPAN_OUTPUT_TOKENS_KEY))
            )
            span_total = None
        if span_total is None and span_input is not None and span_output is not None:
            span_total = span_input + span_output
        input_total = _add(input_total, span_input)
        output_total = _add(output_total, span_output)
        total_total = _add(total_total, span_total)
    return input_total, output_total, total_total


def _span_cost(spans: Sequence[Mapping[str, Any]]) -> float | None:
    total: float | None = None
    for span in spans:
        cost = _decoded(_span_attributes(span).get(_SPAN_COST_KEY))
        if not isinstance(cost, Mapping):
            continue
        value = _cost_value(cost)
        if value is not None:
            total = value if total is None else total + value
    return total


def _span_attributes(span: Mapping[str, Any]) -> Mapping[str, Any]:
    attributes = span.get("attributes")
    return attributes if isinstance(attributes, Mapping) else {}


def _llm_call_count(spans: Sequence[Mapping[str, Any]]) -> int:
    return sum(1 for span in spans if _span_type(span) in _LLM_SPAN_TYPES)


def _span_type(span: Mapping[str, Any]) -> str | None:
    for key in _SPAN_TYPE_KEYS:
        value = span.get(key)
        if isinstance(value, str) and value.strip():
            return _normalized_name(value)
    value = _span_attributes(span).get(_SPAN_TYPE_ATTRIBUTE)
    if isinstance(value, str) and value.strip():
        return _normalized_name(value)
    return None


def _normalized_name(value: str) -> str:
    # MLflow stores enum-valued attributes JSON-encoded, so the quotes are
    # part of the string, and an enum repr arrives dotted.
    return value.strip().strip('"').rpartition(".")[2].upper()


def _duration_ms(info_attr: Any, info_doc: Mapping[str, Any]) -> int | None:
    if info_attr is not None:
        candidate = _token_count(getattr(info_attr, "execution_duration", None))
        if candidate is not None:
            return candidate
    for key in ("execution_duration_ms", "execution_duration", "execution_time_ms"):
        candidate = _token_count(info_doc.get(key))
        if candidate is not None:
            return candidate
    return None


def _state_is_error(info_attr: Any, info_doc: Mapping[str, Any]) -> bool:
    """Whether the trace definitely ended in ERROR.

    Only a readable ERROR marks the row failed; an unreadable or absent
    state leaves success to the error-column evidence, because guessing a
    failure would deflate the denominator the routing decision divides by.
    """

    if info_attr is not None:
        state = getattr(info_attr, "state", None)
        if state is not None:
            return _normalized_name(str(state)) == "ERROR"
    for key in ("state", "status"):
        value = info_doc.get(key)
        if isinstance(value, str) and value.strip():
            return _normalized_name(value) == "ERROR"
    return False


def _configured_cost(
    input_tokens: int | None,
    output_tokens: int | None,
    config: EconomicsConfig,
) -> float | None:
    if config.price_per_1m_input_tokens is None:
        return None
    if config.price_per_1m_output_tokens is None:
        return None
    if input_tokens is None or output_tokens is None:
        # A one-sided count priced at the pair would silently halve the
        # row's spend; the row stays unknown instead.
        return None
    return (
        input_tokens * config.price_per_1m_input_tokens
        + output_tokens * config.price_per_1m_output_tokens
    ) / 1_000_000.0


def _add(total: int | None, value: int | None) -> int | None:
    if value is None:
        return total
    return value if total is None else total + value


def _token_count(value: Any) -> int | None:
    if isinstance(value, bool) or is_missing_scalar(value):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float):
        if not math.isfinite(value) or value < 0 or value != int(value):
            return None
        return int(value)
    return None


def _cost_value(cost: Mapping[str, Any]) -> float | None:
    total = _finite_amount(cost.get("total_cost"))
    if total is not None:
        return total
    input_cost = _finite_amount(cost.get("input_cost"))
    output_cost = _finite_amount(cost.get("output_cost"))
    if input_cost is None and output_cost is None:
        return None
    return (input_cost or 0.0) + (output_cost or 0.0)


def _finite_amount(value: Any) -> float | None:
    if isinstance(value, bool) or is_missing_scalar(value):
        return None
    if not isinstance(value, (int, float)):
        return None
    amount = float(value)
    if not math.isfinite(amount) or amount < 0:
        return None
    return amount


def _decoded(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value
