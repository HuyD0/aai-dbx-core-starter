"""MLflow tracing helpers with a no-op fallback for non-GenAI installs."""

from __future__ import annotations

import hashlib
import inspect
import json
import re
from asyncio import CancelledError
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from contextlib import aclosing, contextmanager, suppress
from contextvars import ContextVar, Token
from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import StrEnum
from functools import wraps
from math import isfinite
from types import MappingProxyType
from typing import Any, TypeVar, cast, overload

from pydantic import Field

from aai_core._sensitive import is_sensitive_name, matches_protected_name
from aai_core.agents import AgentDecision
from aai_core.contracts import ContractModel
from aai_core.tags import DataClassification, ResourceContext

__all__ = [
    "GovernedSpan",
    "TraceCaptureMode",
    "TraceIntegration",
    "TracePolicy",
    "TraceState",
    "configure_tracing",
    "current_trace_state",
    "native_active_span",
    "native_mlflow",
    "provider_span",
    "record_agent_decision",
    "sanitize_trace_payload",
    "set_trace_context",
    "set_trace_resource_context",
    "set_trace_session",
    "trace_context",
    "traced",
]

F = TypeVar("F", bound=Callable[..., Any])
_CONTROLLED_TRACE_KEYS = frozenset(
    f"aai.{field_name}" for field_name in ResourceContext.model_fields
)
_SAFE_OPERATIONAL_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+/-]{0,127}$")
_CREDENTIAL_SHAPED_IDENTIFIER = re.compile(
    r"^(?:dapi|github_pat_|gh[oprsu]_|sk-|eyj)",
    re.IGNORECASE,
)
_OPERATIONAL_IDENTIFIER_ATTRIBUTES = frozenset(
    {
        "aai.logical_name",
        "aai.model",
        "aai.provider",
        "aai.retrieval_mode",
        "gen_ai.operation.name",
        "gen_ai.output.type",
        "gen_ai.provider.name",
        "gen_ai.request.model",
        "gen_ai.response.model",
        "gen_ai.tool.name",
        "gen_ai.tool.type",
        "mlflow.llm.model",
        "mlflow.llm.provider",
        "mlflow.message.format",
    }
)
_DECISION_IDENTIFIER_ATTRIBUTES = frozenset(
    {
        "agent.decision.type",
        "agent.decision.selected_action",
    }
)
_DECISION_CONFIDENCE_ATTRIBUTE = "agent.decision.confidence"
_OPENAI_AUTOLOG_MODEL_SPAN_TYPES = frozenset({"CHAT_MODEL", "EMBEDDING", "LLM"})
_TOKEN_USAGE_ATTRIBUTES = frozenset(
    {
        "gen_ai.usage.input_tokens",
        "gen_ai.usage.output_tokens",
    }
)
_TOKEN_USAGE_KEYS = frozenset(
    {
        "completion_tokens",
        "input_tokens",
        "output_tokens",
        "prompt_tokens",
        "total_tokens",
    }
)
_COST_KEYS = frozenset({"input_cost", "output_cost", "total_cost"})
_NATIVE_SPAN_INPUTS = "mlflow.spanInputs"
_NATIVE_SPAN_OUTPUTS = "mlflow.spanOutputs"
_NATIVE_STRUCTURAL_ATTRIBUTES = frozenset(
    {
        "mlflow.experimentId",
        "mlflow.spanFunctionName",
        "mlflow.spanLogLevel",
        "mlflow.spanStartTimeNs",
        "mlflow.spanType",
        "mlflow.traceRequestId",
    }
)
_NATIVE_LINEAGE_ATTRIBUTES = frozenset(
    {
        "mlflow.gateway.linkedTraceId",
        "mlflow.linkedPrompts",
        "session.id",
    }
)
_LINEAGE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,254}$")
_HASHED_SESSION_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
_OPAQUE_TRACE_METADATA_KEYS = frozenset({"correlation_id", "request_id"})


class TraceCaptureMode(StrEnum):
    """How application payloads may be attached to MLflow spans."""

    OFF = "off"
    BOUNDED = "bounded"
    FULL = "full"
    REDACTED = "redacted"
    METADATA_ONLY = "metadata_only"


class TraceIntegration(StrEnum):
    """The single library responsible for instrumenting an operation."""

    SDK = "sdk"
    MLFLOW_OPENAI = "mlflow_openai"
    MLFLOW_LANGCHAIN = "mlflow_langchain"
    MLFLOW_AGENT_SERVER = "mlflow_agent_server"


class TracePolicy(ContractModel):
    """Validated trace-data boundary for SDK and native instrumentation.

    ``BOUNDED`` is the safe paved-road default. It redacts configured mapping
    keys and limits content size/depth before SDK adapters attach values.
    Applications with richer PII rules can install a native MLflow processor via
    :attr:`native_mlflow`; autologging remains an explicit opt-in because it
    may capture arguments before an SDK adapter can sanitize them.
    """

    capture_mode: TraceCaptureMode = TraceCaptureMode.BOUNDED
    redacted_keys: tuple[str, ...] = (
        "access_token",
        "api_key",
        "authorization",
        "client_secret",
        "password",
        "refresh_token",
        "secret",
        "token",
    )
    max_payload_depth: int = Field(default=8, ge=1, le=32)
    max_string_length: int = Field(default=4_096, ge=0, le=1_000_000)
    max_collection_items: int = Field(default=100, ge=0, le=100_000)


@dataclass(frozen=True)
class TraceState:
    """Execution-local trace configuration.

    This is runtime state, not a persisted contract, so it deliberately keeps
    native values out of Pydantic serialization.
    """

    metadata: Mapping[str, str]
    policy: TracePolicy
    integration: TraceIntegration = TraceIntegration.SDK
    experiment_name: str | None = None
    session_id: str | None = None
    controlled_metadata: Mapping[str, str] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )
        object.__setattr__(
            self,
            "controlled_metadata",
            MappingProxyType(dict(self.controlled_metadata)),
        )


class GovernedSpan:
    """Thin policy boundary over an otherwise native MLflow span."""

    def __init__(
        self,
        native_span: Any,
        policy: TracePolicy,
        *,
        sanitize_at_export: bool = False,
    ) -> None:
        self.native_span = native_span
        self.policy = policy
        self.sanitize_at_export = sanitize_at_export

    def set_inputs(self, value: Any) -> None:
        payload = (
            value
            if self.sanitize_at_export
            else sanitize_trace_payload(value, policy=self.policy)
        )
        self.native_span.set_inputs(payload)

    def set_outputs(self, value: Any) -> None:
        payload = (
            value
            if self.sanitize_at_export
            else sanitize_trace_payload(value, policy=self.policy)
        )
        self.native_span.set_outputs(payload)

    def set_attribute(self, key: str, value: Any) -> None:
        if self.policy.capture_mode is TraceCaptureMode.OFF:
            return
        if self.policy.capture_mode is TraceCaptureMode.METADATA_ONLY:
            if not _is_metadata_only_operational_attribute(key):
                return
            value = _metadata_only_span_attribute(
                key,
                value,
                policy=self.policy,
            )
            self.native_span.set_attribute(key, value)
            return
        sanitized = sanitize_trace_payload({key: value}, policy=self.policy)
        if isinstance(sanitized, Mapping) and key in sanitized:
            value = sanitized[key]
        self.native_span.set_attribute(key, value)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.native_span, name)


_DEFAULT_TRACE_STATE = TraceState(
    metadata={},
    policy=TracePolicy(capture_mode=TraceCaptureMode.OFF),
)
_TRACE_STATE: ContextVar[TraceState | None] = ContextVar(
    "aai_core_trace_state",
    default=None,
)
_PROCESS_TRACE_CONFIGURATION: dict[str, tuple[Any, ...]] = {}


def configure_tracing(
    context: ResourceContext,
    *,
    experiment_name: str,
    tracking_uri: str | None = None,
    integration: TraceIntegration = TraceIntegration.SDK,
    policy: TracePolicy | None = None,
    autolog_options: Mapping[str, Any] | None = None,
) -> TraceState:
    """Configure one process-wide tracing owner during application startup."""

    global _DEFAULT_TRACE_STATE

    if not experiment_name.strip():
        raise ValueError("experiment_name must not be blank")
    if not isinstance(integration, TraceIntegration):
        raise TypeError("integration must be a TraceIntegration")
    mlflow = _require_mlflow()
    selected_policy = policy or _default_trace_policy(context)
    if context.data_classification in {
        DataClassification.CONFIDENTIAL,
        DataClassification.RESTRICTED,
    } and selected_policy.capture_mode not in {
        TraceCaptureMode.OFF,
        TraceCaptureMode.METADATA_ONLY,
    }:
        raise ValueError(
            "confidential and restricted data classifications permit only "
            "metadata_only or off tracing; application code cannot weaken "
            "this platform boundary"
        )
    if (
        integration
        in {
            TraceIntegration.MLFLOW_OPENAI,
            TraceIntegration.MLFLOW_LANGCHAIN,
        }
        and selected_policy.capture_mode is not TraceCaptureMode.FULL
    ):
        raise ValueError(
            f"{integration.value} requires TraceCaptureMode.FULL because native "
            "autologging can capture provider/framework payloads before the SDK "
            "can sanitize them."
        )
    controlled_metadata = _sanitize_trace_metadata(
        context.for_trace(),
        policy=selected_policy,
        allow_new_controlled=True,
    )
    state = TraceState(
        metadata=controlled_metadata,
        policy=selected_policy,
        integration=integration,
        experiment_name=experiment_name,
        controlled_metadata=controlled_metadata,
    )
    options = dict(autolog_options or {})
    signature = (
        tuple(controlled_metadata.items()),
        experiment_name,
        tracking_uri,
        integration,
        selected_policy,
        tuple(sorted(options.items())),
    )
    configured_signature = _PROCESS_TRACE_CONFIGURATION.get("signature")
    if configured_signature is not None:
        if signature != configured_signature:
            raise RuntimeError(
                "Tracing is already configured for this process with a "
                "different experiment, integration, policy, or resource context. "
                "Configure tracing once during process startup."
            )
        _TRACE_STATE.set(_DEFAULT_TRACE_STATE)
        return _DEFAULT_TRACE_STATE

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    if integration is TraceIntegration.MLFLOW_AGENT_SERVER:
        _configure_native_span_policy(mlflow, selected_policy)
    if integration is TraceIntegration.MLFLOW_OPENAI:
        mlflow.openai.autolog(**options)
    elif integration is TraceIntegration.MLFLOW_LANGCHAIN:
        mlflow.langchain.autolog(**options)

    _PROCESS_TRACE_CONFIGURATION["signature"] = signature
    _DEFAULT_TRACE_STATE = state
    _TRACE_STATE.set(state)
    return state


@overload
def traced(
    function: F,
    *,
    name: str | None = None,
    span_type: str | None = None,
) -> F:
    raise NotImplementedError


@overload
def traced(
    function: None = None,
    *,
    name: str | None = None,
    span_type: str | None = None,
) -> Callable[[F], F]:
    raise NotImplementedError


def traced(
    function: F | None = None,
    *,
    name: str | None = None,
    span_type: str | None = None,
) -> F | Callable[[F], F]:
    """Trace a call under the active policy, or no-op without MLflow.

    Bounded modes use native spans with SDK-controlled input/output capture.
    ``FULL`` delegates to MLflow's decorator so explicitly opted-in callers
    retain the complete native behavior.
    """

    def decorate(target: F) -> F:
        try:
            import mlflow
        except ImportError:
            return target
        operation_name = name or target.__name__
        operation_type = span_type or "UNKNOWN"
        if inspect.isasyncgenfunction(target):
            return cast(
                F,
                _decorate_async_generator(
                    target,
                    mlflow=mlflow,
                    name=operation_name,
                    span_type=operation_type,
                ),
            )
        if inspect.iscoroutinefunction(target):
            return cast(
                F,
                _decorate_coroutine(
                    target,
                    mlflow=mlflow,
                    name=operation_name,
                    span_type=operation_type,
                ),
            )
        return cast(
            F,
            _decorate_sync(
                target,
                mlflow=mlflow,
                name=operation_name,
                span_type=operation_type,
            ),
        )

    if function is None:
        return decorate
    return decorate(function)


def _decorate_async_generator(
    target: Callable[..., Any],
    *,
    mlflow: Any,
    name: str,
    span_type: str,
) -> Callable[..., AsyncIterator[Any]]:
    @wraps(target)
    async def dispatch(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        state = current_trace_state()
        if state.policy.capture_mode is TraceCaptureMode.OFF:
            async for item in _iterate_async_generator(target, args, kwargs):
                yield item
            return

        token = _TRACE_STATE.set(state)
        try:
            if not hasattr(mlflow, "start_span"):
                _apply_trace_state(state)
                async for item in _iterate_async_generator(target, args, kwargs):
                    yield item
                return
            if state.policy.capture_mode is TraceCaptureMode.FULL:
                async for item in _full_async_generator_span(
                    target,
                    args,
                    kwargs,
                    mlflow=mlflow,
                    name=name,
                    span_type=span_type,
                    state=state,
                ):
                    yield item
                return
            async for item in _bounded_async_generator_span(
                target,
                args,
                kwargs,
                name=name,
                span_type=span_type,
            ):
                yield item
        finally:
            _TRACE_STATE.reset(token)

    return dispatch


async def _iterate_async_generator(
    target: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> AsyncIterator[Any]:
    try:
        async with aclosing(target(*args, **kwargs)) as iterator:
            async for item in iterator:
                yield item
    except GeneratorExit:
        return


async def _full_async_generator_span(
    target: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    *,
    mlflow: Any,
    name: str,
    span_type: str,
    state: TraceState,
) -> AsyncIterator[Any]:
    with mlflow.start_span(name=name, span_type=span_type) as native_span:
        _apply_trace_state(state)
        span = GovernedSpan(native_span, state.policy)
        span.set_inputs(_bound_inputs(target, args, kwargs))
        item_count = 0
        async for item in _iterate_async_generator(target, args, kwargs):
            item_count += 1
            yield item
        span.set_outputs({"stream_items": item_count})


async def _bounded_async_generator_span(
    target: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    *,
    name: str,
    span_type: str,
) -> AsyncIterator[Any]:
    with provider_span(name, span_type=span_type) as span:
        if span is not None:
            span.set_inputs(_bound_inputs(target, args, kwargs))
        item_count = 0
        async for item in _iterate_async_generator(target, args, kwargs):
            item_count += 1
            yield item
        if span is not None:
            span.set_outputs({"stream_items": item_count})


def _decorate_coroutine(
    target: Callable[..., Any],
    *,
    mlflow: Any,
    name: str,
    span_type: str,
) -> Callable[..., Any]:
    @wraps(target)
    async def invoke_native(*args: Any, **kwargs: Any) -> Any:
        state = current_trace_state()
        token = _TRACE_STATE.set(state)
        try:
            _apply_trace_state(state)
            return await target(*args, **kwargs)
        finally:
            _TRACE_STATE.reset(token)

    native_traced = mlflow.trace(name=name, span_type=span_type)(invoke_native)

    @wraps(target)
    async def dispatch(*args: Any, **kwargs: Any) -> Any:
        state = current_trace_state()
        if state.policy.capture_mode is TraceCaptureMode.OFF:
            return await target(*args, **kwargs)
        if not hasattr(mlflow, "start_span"):
            return await native_traced(*args, **kwargs)
        token = _TRACE_STATE.set(state)
        try:
            if state.policy.capture_mode is TraceCaptureMode.FULL:
                return await _full_coroutine_span(
                    target,
                    args,
                    kwargs,
                    mlflow=mlflow,
                    name=name,
                    span_type=span_type,
                    state=state,
                )
            return await _bounded_coroutine_span(
                target,
                args,
                kwargs,
                name=name,
                span_type=span_type,
            )
        finally:
            _TRACE_STATE.reset(token)

    return dispatch


async def _full_coroutine_span(
    target: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    *,
    mlflow: Any,
    name: str,
    span_type: str,
    state: TraceState,
) -> Any:
    # A synchronous native span context closes before cancellation propagates.
    with mlflow.start_span(name=name, span_type=span_type) as native_span:
        _apply_trace_state(state)
        span = GovernedSpan(native_span, state.policy)
        span.set_inputs(_bound_inputs(target, args, kwargs))
        result = await target(*args, **kwargs)
        span.set_outputs(result)
        return result


async def _bounded_coroutine_span(
    target: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    *,
    name: str,
    span_type: str,
) -> Any:
    with provider_span(name, span_type=span_type) as span:
        if span is not None:
            span.set_inputs(_bound_inputs(target, args, kwargs))
        result = await target(*args, **kwargs)
        if span is not None:
            span.set_outputs(result)
        return result


def _decorate_sync(
    target: Callable[..., Any],
    *,
    mlflow: Any,
    name: str,
    span_type: str,
) -> Callable[..., Any]:
    @wraps(target)
    def invoke_native(*args: Any, **kwargs: Any) -> Any:
        state = current_trace_state()
        token = _TRACE_STATE.set(state)
        try:
            _apply_trace_state(state)
            return target(*args, **kwargs)
        finally:
            _TRACE_STATE.reset(token)

    native_traced = mlflow.trace(name=name, span_type=span_type)(invoke_native)

    @wraps(target)
    def dispatch(*args: Any, **kwargs: Any) -> Any:
        state = current_trace_state()
        if state.policy.capture_mode is TraceCaptureMode.OFF:
            return target(*args, **kwargs)
        if state.policy.capture_mode is TraceCaptureMode.FULL or not hasattr(
            mlflow, "start_span"
        ):
            return native_traced(*args, **kwargs)
        token = _TRACE_STATE.set(state)
        try:
            return _bounded_sync_span(
                target,
                args,
                kwargs,
                name=name,
                span_type=span_type,
            )
        finally:
            _TRACE_STATE.reset(token)

    return dispatch


def _bounded_sync_span(
    target: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    *,
    name: str,
    span_type: str,
) -> Any:
    with provider_span(name, span_type=span_type) as span:
        if span is not None:
            span.set_inputs(_bound_inputs(target, args, kwargs))
        result = target(*args, **kwargs)
        if span is not None:
            span.set_outputs(result)
        return result


@contextmanager
def provider_span(
    name: str,
    *,
    span_type: str,
    attributes: Mapping[str, Any] | None = None,
) -> Iterator[GovernedSpan | None]:
    """Create one policy-bound span under the selected instrumentation owner.

    OpenAI autologging owns native model/chat/embedding calls, while explicit
    application TOOL and RETRIEVER spans remain authoritative. LangChain
    autologging owns the complete framework trajectory, including its tool and
    retriever spans, so all manual provider spans are suppressed in that mode.
    """

    state = current_trace_state()
    native_owned = state.integration is TraceIntegration.MLFLOW_LANGCHAIN or (
        state.integration is TraceIntegration.MLFLOW_OPENAI
        and str(span_type).strip().upper() in _OPENAI_AUTOLOG_MODEL_SPAN_TYPES
    )
    if state.policy.capture_mode is TraceCaptureMode.OFF or native_owned:
        yield None
        return
    try:
        import mlflow
    except ImportError:
        yield None
        return

    state = current_trace_state()
    policy = state.policy
    failure: tuple[BaseException, Any] | None = None
    with mlflow.start_span(name=name, span_type=span_type) as native_span:
        _apply_trace_state(state)
        span = GovernedSpan(
            native_span,
            policy,
            sanitize_at_export=(
                state.integration is TraceIntegration.MLFLOW_AGENT_SERVER
                and policy.capture_mode is not TraceCaptureMode.FULL
            ),
        )
        for key, value in (attributes or {}).items():
            span.set_attribute(key, value)
        try:
            yield span
        except (
            Exception,
            CancelledError,
            GeneratorExit,
            KeyboardInterrupt,
            SystemExit,
        ) as caught_error:
            # MLflow records exception messages and complete chained
            # tracebacks when an exception exits start_span. Those strings can
            # contain prompts, tool inputs, or provider responses. Exit the
            # native context normally with a generic status, then re-raise the
            # original failure outside the telemetry boundary.
            failure = (caught_error, caught_error.__traceback__)
            with suppress(Exception):
                native_span.set_status("ERROR")
    if failure is not None:
        failure_error, failure_traceback = failure
        raise failure_error.with_traceback(failure_traceback) from None


@contextmanager
def _application_semantic_span(
    name: str,
    *,
    span_type: str,
    attributes: Mapping[str, Any] | None = None,
) -> Iterator[GovernedSpan | None]:
    """Create a governed application span only inside an existing MLflow trace.

    Application semantics such as decisions are independent of the library that
    owns provider/framework instrumentation. Unlike :func:`provider_span`, this
    path is therefore available under every ``TraceIntegration``. Requiring an
    active parent prevents best-effort telemetry from creating disconnected
    decision-only traces.
    """

    state = current_trace_state()
    if state.policy.capture_mode is TraceCaptureMode.OFF:
        yield None
        return
    try:
        import mlflow
    except ImportError:
        yield None
        return

    active_span = getattr(mlflow, "get_current_active_span", None)
    parent = active_span() if callable(active_span) else None
    if parent is None:
        yield None
        return
    parent_attribute = getattr(parent, "get_attribute", None)
    if callable(parent_attribute):
        trace_request_id = parent_attribute("mlflow.traceRequestId")
        if not isinstance(trace_request_id, str) or not trace_request_id.strip():
            yield None
            return

    policy = state.policy
    with mlflow.start_span(name=name, span_type=span_type) as native_span:
        _apply_trace_state(state)
        span = GovernedSpan(
            native_span,
            policy,
            sanitize_at_export=(
                state.integration is TraceIntegration.MLFLOW_AGENT_SERVER
                and policy.capture_mode is not TraceCaptureMode.FULL
            ),
        )
        for key, value in (attributes or {}).items():
            span.set_attribute(key, value)
        yield span


def record_agent_decision(decision: AgentDecision) -> None:
    """Best-effort recording of an application decision as an MLflow AGENT span.

    A decision record is concise application evidence, not provider reasoning
    or hidden chain-of-thought. The span closes before any selected action is
    executed, so subsequent TOOL/RETRIEVER/LLM spans remain authoritative for
    execution, failures, retries, and outputs. Without an active application or
    agent trace this helper is a no-op rather than creating disconnected
    evidence. Telemetry failures deliberately do not alter application behavior.
    """

    if not isinstance(decision, AgentDecision):
        raise TypeError("decision must be an AgentDecision")

    attributes: dict[str, Any] = {
        "agent.decision.type": decision.decision_type.value,
        "agent.decision.goal": decision.goal,
        "agent.decision.selected_action": decision.selected_action,
        "agent.decision.reason": decision.reason,
        "agent.decision.evidence_refs": list(decision.evidence_refs),
    }
    if decision.alternatives_considered:
        attributes["agent.decision.alternatives"] = list(
            decision.alternatives_considered
        )
    if decision.confidence is not None:
        attributes[_DECISION_CONFIDENCE_ATTRIBUTE] = decision.confidence
    if decision.expected_result is not None:
        attributes["agent.decision.expected_result"] = decision.expected_result

    try:
        with _application_semantic_span(
            f"decision.{decision.decision_type.value}",
            span_type="AGENT",
            attributes=attributes,
        ):
            pass
    except Exception:
        # Instrumentation is evidence about application behavior, never a
        # dependency of that behavior. BaseException subclasses such as task
        # cancellation, KeyboardInterrupt, and SystemExit are not suppressed.
        return


def set_trace_context(metadata: Mapping[str, str]) -> None:
    """Merge bounded request metadata into the active trace.

    Canonical ``aai.*`` ownership, cost, and lifecycle fields come from
    :class:`ResourceContext` and cannot be replaced through this request-level
    API. Sensitive-looking metadata values are redacted even when full payload
    capture is explicitly enabled.
    """

    current = current_trace_state()
    sanitized = _sanitize_trace_metadata(
        metadata,
        policy=current.policy,
        controlled_metadata=current.controlled_metadata,
    )
    merged = {
        **current.metadata,
        **sanitized,
    }
    _TRACE_STATE.set(replace(current, metadata=merged))
    if current.policy.capture_mode is not TraceCaptureMode.OFF:
        _update_native_trace(metadata=sanitized)


def set_trace_resource_context(context: ResourceContext) -> None:
    """Attach canonical resource metadata as controlled trace fields.

    This is intended for native servers, such as MLflow Agent Server, that
    create the root trace before application initialization. A different
    resource context cannot replace one already bound to the execution.
    """

    current = current_trace_state()
    controlled = _sanitize_trace_metadata(
        context.for_trace(),
        policy=current.policy,
        allow_new_controlled=True,
    )
    conflicts = {
        key
        for key, value in controlled.items()
        if key in current.controlled_metadata
        and current.controlled_metadata[key] != value
    }
    if conflicts:
        raise ValueError(
            "Trace resource context conflicts with controlled metadata: "
            + ", ".join(sorted(conflicts))
        )
    combined_controlled = {**current.controlled_metadata, **controlled}
    state = replace(
        current,
        metadata={**current.metadata, **controlled},
        controlled_metadata=combined_controlled,
    )
    _TRACE_STATE.set(state)
    if current.policy.capture_mode is not TraceCaptureMode.OFF:
        _update_native_trace(metadata=controlled)


def set_trace_session(session_id: str) -> None:
    """Group the active trace into an MLflow conversation session.

    Uses MLflow's dedicated ``session_id`` field rather than ordinary trace
    metadata. Calling this when no trace is active is a no-op, matching
    :func:`set_trace_context`.
    """

    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must not be blank")
    current = current_trace_state()
    sanitized_session_id = _sanitize_trace_session_id(
        session_id,
        policy=current.policy,
    )
    if current.policy.capture_mode is TraceCaptureMode.OFF:
        _TRACE_STATE.set(replace(current, session_id=sanitized_session_id))
        return
    try:
        import mlflow
    except ImportError:
        return
    try:
        mlflow.update_current_trace(session_id=sanitized_session_id)
    except Exception:
        # Session context can only be attached while a trace is active.
        return
    _TRACE_STATE.set(replace(current, session_id=sanitized_session_id))


def current_trace_state() -> TraceState:
    """Return the execution-local policy and metadata snapshot."""

    state = _TRACE_STATE.get()
    return state if state is not None else _DEFAULT_TRACE_STATE


@contextmanager
def trace_context(
    *,
    metadata: Mapping[str, str] | None = None,
    session_id: str | None = None,
) -> Iterator[TraceState]:
    """Bind task-local request metadata without changing startup policy."""

    current = current_trace_state()
    sanitized = _sanitize_trace_metadata(
        metadata or {},
        policy=current.policy,
        controlled_metadata=current.controlled_metadata,
    )
    merged = dict(current.metadata)
    merged.update(sanitized)
    sanitized_session_id = None
    if session_id is not None:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must not be blank")
        sanitized_session_id = _sanitize_trace_session_id(
            session_id,
            policy=current.policy,
        )
    state = TraceState(
        metadata=merged,
        policy=current.policy,
        integration=current.integration,
        experiment_name=current.experiment_name,
        session_id=(
            sanitized_session_id if session_id is not None else current.session_id
        ),
        controlled_metadata=current.controlled_metadata,
    )
    token: Token[TraceState | None] = _TRACE_STATE.set(state)
    try:
        yield state
    finally:
        _TRACE_STATE.reset(token)


def sanitize_trace_payload(
    value: Any,
    *,
    policy: TracePolicy | None = None,
) -> Any:
    """Apply the active capture policy before attaching a payload to a span."""

    selected = policy or current_trace_state().policy
    if selected.capture_mode is TraceCaptureMode.OFF:
        return None
    if selected.capture_mode is TraceCaptureMode.FULL:
        return value
    if selected.capture_mode is TraceCaptureMode.METADATA_ONLY:
        return _payload_shape(
            value,
            depth=0,
            max_depth=selected.max_payload_depth,
            max_collection_items=selected.max_collection_items,
        )
    redacted = {key.casefold() for key in selected.redacted_keys}
    return _redact_payload(
        value,
        redacted_keys=redacted,
        depth=0,
        max_depth=selected.max_payload_depth,
        max_string_length=(
            selected.max_string_length
            if selected.capture_mode is TraceCaptureMode.BOUNDED
            else None
        ),
        max_collection_items=(
            selected.max_collection_items
            if selected.capture_mode is TraceCaptureMode.BOUNDED
            else None
        ),
    )


def _default_trace_policy(context: ResourceContext) -> TracePolicy:
    """Select the safest default capture policy for the data classification."""

    if context.data_classification in {
        DataClassification.CONFIDENTIAL,
        DataClassification.RESTRICTED,
    }:
        return TracePolicy(capture_mode=TraceCaptureMode.METADATA_ONLY)
    return TracePolicy()


def native_mlflow() -> Any:
    """Return the native MLflow module for unsupported tracing features."""

    return _require_mlflow()


def native_active_span() -> Any | None:
    """Return MLflow's active span without wrapping or mirroring it."""

    mlflow = _require_mlflow()
    getter = getattr(mlflow, "get_current_active_span", None)
    return getter() if getter is not None else None


def _configure_native_span_policy(mlflow: Any, policy: TracePolicy) -> None:
    """Enforce capture policy on spans created by native framework servers.

    MLflow Agent Server owns its root span and writes outputs only after the
    application handler returns. A pre-export processor is therefore the only
    supported boundary that can apply the same policy to both SDK-created
    children and that framework-owned root.
    """

    tracing_api = getattr(mlflow, "tracing", None)
    configure = getattr(tracing_api, "configure", None)
    if callable(configure) and policy.capture_mode is not TraceCaptureMode.FULL:
        config_api = getattr(tracing_api, "config", None)
        get_config = getattr(config_api, "get_config", None)
        if not callable(get_config):
            raise RuntimeError(
                "The certified MLflow tracing runtime must expose "
                "tracing.config.get_config so AAI can compose with existing "
                "span processors without replacing them."
            )
        existing = list(getattr(get_config(), "span_processors", ()))
        processors = [
            processor
            for processor in existing
            if getattr(processor, "__name__", "") != "aai_core_trace_capture_policy"
        ]
        processors.append(_native_span_policy_processor(policy))
        configure(span_processors=processors)

    if policy.capture_mode is TraceCaptureMode.OFF:
        disable = getattr(tracing_api, "disable", None)
        if callable(disable):
            disable()
    else:
        enable = getattr(tracing_api, "enable", None)
        if callable(enable):
            enable()


def _native_span_policy_processor(
    policy: TracePolicy,
) -> Callable[[Any], None]:
    def apply_policy(span: Any) -> None:
        inputs = _read_native_span_value(span, "inputs")
        outputs = _read_native_span_value(span, "outputs")
        _replace_native_span_payload(span, "set_inputs", inputs, policy=policy)
        _replace_native_span_payload(span, "set_outputs", outputs, policy=policy)
        _clear_native_status_description(span)
        attributes = _native_span_attributes(span)
        for key, value in attributes.items():
            if key in {
                _NATIVE_SPAN_INPUTS,
                _NATIVE_SPAN_OUTPUTS,
                *_NATIVE_STRUCTURAL_ATTRIBUTES,
            }:
                continue
            _replace_native_span_attribute(span, key, value, policy=policy)

    apply_policy.__name__ = "aai_core_trace_capture_policy"
    return apply_policy


def _read_native_span_value(span: Any, name: str) -> Any:
    try:
        return getattr(span, name, None)
    except Exception:
        return None


def _native_span_attributes(span: Any) -> dict[str, Any]:
    try:
        return dict(getattr(span, "attributes", {}))
    except Exception:
        return {}


def _clear_native_status_description(span: Any) -> None:
    """Remove provider exception text while retaining the error status code."""

    try:
        status = span.status
        if getattr(status, "description", ""):
            span.set_status(status.status_code)
    except Exception:
        return


def _replace_native_span_attribute(
    span: Any,
    key: str,
    value: Any,
    *,
    policy: TracePolicy,
) -> None:
    # Establish a safe value first because MLflow intentionally swallows
    # processor exceptions; computing before replacement would fail open.
    try:
        span.set_attribute(key, {"type": "suppressed"})
    except Exception:
        return
    try:
        sanitized = _sanitize_native_span_attribute(key, value, policy=policy)
    except Exception:
        return
    with suppress(Exception):
        span.set_attribute(key, sanitized)


def _sanitize_native_span_attribute(
    key: str,
    value: Any,
    *,
    policy: TracePolicy,
) -> Any:
    if policy.capture_mode is TraceCaptureMode.METADATA_ONLY:
        if key in _NATIVE_LINEAGE_ATTRIBUTES:
            return _metadata_only_lineage_attribute(key, value)
        if _is_metadata_only_operational_attribute(key):
            return _metadata_only_span_attribute(key, value, policy=policy)
        return _payload_shape(
            value,
            depth=0,
            max_depth=policy.max_payload_depth,
            max_collection_items=policy.max_collection_items,
        )
    if policy.capture_mode is TraceCaptureMode.OFF:
        return {"type": "suppressed"}
    candidate = sanitize_trace_payload({key: value}, policy=policy)
    if isinstance(candidate, Mapping) and key in candidate:
        return candidate[key]
    return None


def _replace_native_span_payload(
    span: Any,
    setter_name: str,
    value: Any,
    *,
    policy: TracePolicy,
) -> None:
    """Replace one framework-owned payload without a fail-open window."""

    setter = getattr(span, setter_name, None)
    if not callable(setter):
        return
    try:
        setter(None)
    except Exception:
        return
    try:
        sanitized = sanitize_trace_payload(value, policy=policy)
    except Exception:
        return
    try:
        setter(sanitized)
    except Exception:
        with suppress(Exception):
            setter(None)


def _metadata_only_lineage_attribute(key: str, value: Any) -> Any:
    if key == "session.id" and isinstance(value, str) and value.strip():
        return _sanitize_trace_session_id(
            value,
            policy=TracePolicy(capture_mode=TraceCaptureMode.METADATA_ONLY),
        )
    if key == "mlflow.gateway.linkedTraceId" and isinstance(value, str):
        identifier = value.strip()
        if _LINEAGE_IDENTIFIER.fullmatch(identifier):
            return identifier
    if key == "mlflow.linkedPrompts" and isinstance(value, str):
        try:
            entries = json.loads(value)
        except (TypeError, ValueError):
            entries = None
        if isinstance(entries, list) and len(entries) <= 32:
            normalized: list[dict[str, str]] = []
            for entry in entries:
                if not isinstance(entry, Mapping) or set(entry) != {"name", "version"}:
                    break
                name = entry.get("name")
                version = entry.get("version")
                if (
                    not isinstance(name, str)
                    or not _LINEAGE_IDENTIFIER.fullmatch(name.strip())
                    or not isinstance(version, str)
                    or not version.isascii()
                    or not version.isdigit()
                    or version.startswith("0")
                ):
                    break
                normalized.append({"name": name.strip(), "version": version})
            else:
                return json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return {"type": "suppressed"}


def _sanitize_trace_session_id(session_id: str, *, policy: TracePolicy) -> str:
    normalized = session_id.strip()
    if _HASHED_SESSION_ID.fullmatch(normalized):
        return normalized
    if policy.capture_mode in {
        TraceCaptureMode.METADATA_ONLY,
        TraceCaptureMode.OFF,
    }:
        return _opaque_identifier(normalized)
    return normalized


def _opaque_identifier(value: str) -> str:
    normalized = value.strip()
    if _HASHED_SESSION_ID.fullmatch(normalized):
        return normalized
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _apply_trace_state(state: TraceState) -> None:
    if state.policy.capture_mode is TraceCaptureMode.OFF:
        return
    if state.metadata:
        _update_native_trace(metadata=dict(state.metadata))
    if state.session_id:
        _update_native_trace(session_id=state.session_id)


def _update_native_trace(**options: Any) -> None:
    try:
        import mlflow
    except ImportError:
        return
    try:
        mlflow.update_current_trace(**options)
    except Exception:
        # Context can be configured outside a trace and applied by a later
        # traced invocation.
        return


def _sanitize_trace_metadata(
    metadata: Mapping[str, str],
    *,
    policy: TracePolicy,
    controlled_metadata: Mapping[str, str] | None = None,
    allow_new_controlled: bool = False,
) -> dict[str, str]:
    """Validate, redact, and bound low-cardinality MLflow trace metadata."""

    protected = controlled_metadata or {}
    redacted_keys = {key.casefold() for key in policy.redacted_keys}
    sanitized: dict[str, str] = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or not key.strip():
            raise TypeError("Trace metadata keys must be non-empty strings")
        if not isinstance(value, str):
            raise TypeError(f"Trace metadata value for {key!r} must be a string")
        if key in _CONTROLLED_TRACE_KEYS:
            if allow_new_controlled:
                # ResourceContext has already validated these platform-owned
                # identifiers. Preserve them exactly so tags remain joinable.
                sanitized[key] = value
                continue
            expected = protected.get(key)
            if expected is None or value != expected:
                raise ValueError(
                    f"Trace metadata cannot override controlled field {key!r}; "
                    "bind a ResourceContext instead."
                )
            sanitized[key] = value
            continue
        if policy.capture_mode in {
            TraceCaptureMode.METADATA_ONLY,
            TraceCaptureMode.OFF,
        }:
            if key in _OPAQUE_TRACE_METADATA_KEYS:
                sanitized[key] = _opaque_identifier(value)
            # Sensitive classifications expose no arbitrary trace-metadata
            # channel. Platform-owned aai.* values were handled above.
            continue
        if _is_redacted_key(key, redacted_keys):
            sanitized[key] = "[REDACTED]"
        elif len(value) > policy.max_string_length:
            sanitized[key] = value[: policy.max_string_length] + "<truncated>"
        else:
            sanitized[key] = value
    return sanitized


def _redact_payload(
    value: Any,
    *,
    redacted_keys: set[str],
    depth: int,
    max_depth: int,
    max_string_length: int | None,
    max_collection_items: int | None,
) -> Any:
    if depth >= max_depth:
        return "<max-depth>"
    if is_dataclass(value) and not isinstance(value, type):
        value = {
            item.name: getattr(value, item.name)
            for item in fields(value)
            if not item.name.startswith("_")
        }
    elif callable(getattr(value, "model_dump", None)):
        try:
            value = value.model_dump(mode="python")
        except Exception:
            return {"type": type(value).__name__}
    if isinstance(value, Mapping):
        return _redact_mapping(
            value,
            redacted_keys=redacted_keys,
            depth=depth,
            max_depth=max_depth,
            max_string_length=max_string_length,
            max_collection_items=max_collection_items,
        )
    if isinstance(value, (list, tuple)):
        return _redact_sequence(
            value,
            redacted_keys=redacted_keys,
            depth=depth,
            max_depth=max_depth,
            max_string_length=max_string_length,
            max_collection_items=max_collection_items,
        )
    if (
        isinstance(value, str)
        and max_string_length is not None
        and len(value) > max_string_length
    ):
        return value[:max_string_length] + "<truncated>"
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return {"type": "bytes", "length": len(value)}
    return {"type": type(value).__name__}


def _redact_mapping(
    value: Mapping[Any, Any],
    *,
    redacted_keys: set[str],
    depth: int,
    max_depth: int,
    max_string_length: int | None,
    max_collection_items: int | None,
) -> dict[str, Any]:
    items = list(value.items())
    truncated = max_collection_items is not None and len(items) > max_collection_items
    if max_collection_items is not None:
        items = items[:max_collection_items]
    result = {
        str(key): (
            "[REDACTED]"
            if _is_redacted_key(str(key), redacted_keys)
            else _redact_payload(
                item,
                redacted_keys=redacted_keys,
                depth=depth + 1,
                max_depth=max_depth,
                max_string_length=max_string_length,
                max_collection_items=max_collection_items,
            )
        )
        for key, item in items
    }
    if truncated:
        result["<truncated>"] = len(value) - len(items)
    return result


def _redact_sequence(
    value: list[Any] | tuple[Any, ...],
    *,
    redacted_keys: set[str],
    depth: int,
    max_depth: int,
    max_string_length: int | None,
    max_collection_items: int | None,
) -> list[Any]:
    items = list(value)
    truncated = max_collection_items is not None and len(items) > max_collection_items
    if max_collection_items is not None:
        items = items[:max_collection_items]
    result = [
        _redact_payload(
            item,
            redacted_keys=redacted_keys,
            depth=depth + 1,
            max_depth=max_depth,
            max_string_length=max_string_length,
            max_collection_items=max_collection_items,
        )
        for item in items
    ]
    if truncated:
        result.append({"<truncated>": len(value) - len(items)})
    return result


def _is_redacted_key(key: str, redacted_keys: set[str]) -> bool:
    return is_sensitive_name(key) or matches_protected_name(key, redacted_keys)


def _bound_inputs(
    target: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        arguments = dict(
            inspect.signature(target).bind_partial(*args, **kwargs).arguments
        )
    except (TypeError, ValueError):
        arguments = {"args": list(args), "kwargs": dict(kwargs)}
    # Bound methods often carry a large object graph and credentials on self.
    arguments.pop("self", None)
    arguments.pop("cls", None)
    return arguments


def _payload_shape(
    value: Any,
    *,
    depth: int,
    max_depth: int,
    max_collection_items: int,
) -> Any:
    if depth >= max_depth:
        return {"type": type(value).__name__}
    if isinstance(value, Mapping):
        return {
            "type": "mapping",
            "size": len(value),
            "truncated": len(value) > max_collection_items,
        }
    if isinstance(value, (list, tuple)):
        items = list(value[:max_collection_items])
        return {
            "type": "sequence",
            "size": len(value),
            "item_types": sorted({type(item).__name__ for item in items}),
            "truncated": len(value) > max_collection_items,
        }
    if isinstance(value, str):
        return {"type": "str", "length": len(value)}
    return {"type": type(value).__name__}


def _metadata_only_span_attribute(
    key: str,
    value: Any,
    *,
    policy: TracePolicy,
) -> Any:
    """Retain only typed, low-cardinality operational evidence.

    Inputs and outputs remain shape-only for sensitive classifications. A small
    allowlist preserves the model/tool identifiers and token/cost values needed
    for reliability and spend monitoring without opening an arbitrary attribute
    channel for application payloads.
    """

    if key in (
        _OPERATIONAL_IDENTIFIER_ATTRIBUTES | _DECISION_IDENTIFIER_ATTRIBUTES
    ) and isinstance(value, str):
        identifier = value.strip()
        if _SAFE_OPERATIONAL_IDENTIFIER.fullmatch(
            identifier
        ) and not _CREDENTIAL_SHAPED_IDENTIFIER.match(identifier):
            return identifier
    elif key == _DECISION_CONFIDENCE_ATTRIBUTE:
        if (
            isinstance(value, int | float)
            and not isinstance(value, bool)
            and isfinite(value)
            and 0.0 <= value <= 1.0
        ):
            return value
    elif key in _TOKEN_USAGE_ATTRIBUTES:
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    elif key == "mlflow.chat.tokenUsage":
        usage = _bounded_numeric_mapping(value, allowed_keys=_TOKEN_USAGE_KEYS)
        if usage is not None and all(isinstance(item, int) for item in usage.values()):
            return usage
    elif key == "mlflow.llm.cost":
        cost = _bounded_numeric_mapping(value, allowed_keys=_COST_KEYS)
        if cost is not None:
            return cost

    return _payload_shape(
        value,
        depth=0,
        max_depth=policy.max_payload_depth,
        max_collection_items=policy.max_collection_items,
    )


def _is_metadata_only_operational_attribute(key: str) -> bool:
    return (
        key in _OPERATIONAL_IDENTIFIER_ATTRIBUTES
        or key in _DECISION_IDENTIFIER_ATTRIBUTES
        or key == _DECISION_CONFIDENCE_ATTRIBUTE
        or key in _TOKEN_USAGE_ATTRIBUTES
        or key in {"mlflow.chat.tokenUsage", "mlflow.llm.cost"}
    )


def _bounded_numeric_mapping(
    value: Any,
    *,
    allowed_keys: frozenset[str],
) -> dict[str, int | float] | None:
    if not isinstance(value, Mapping) or not value:
        return None
    if not set(value).issubset(allowed_keys):
        return None
    result: dict[str, int | float] = {}
    for key, item in value.items():
        if isinstance(item, bool) or not isinstance(item, (int, float)) or item < 0:
            return None
        # NaN and infinity cannot become portable JSON evidence.
        if isinstance(item, float) and not isfinite(item):
            return None
        result[str(key)] = item
    return result


def _require_mlflow() -> Any:
    try:
        import mlflow
    except ImportError as error:
        raise RuntimeError(
            "MLflow support requires the `genai` extra. From an aai-core "
            "checkout run `make examples-install` and use `.venv/bin/python`; "
            "in a consuming environment install `aai-core[genai]`."
        ) from error
    return mlflow
