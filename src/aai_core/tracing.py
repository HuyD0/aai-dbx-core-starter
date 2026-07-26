"""MLflow tracing helpers with a no-op fallback for non-GenAI installs."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import StrEnum
from functools import wraps
from types import MappingProxyType
from typing import Any, TypeVar, cast

from pydantic import Field

from aai_core.contracts import ContractModel
from aai_core.tags import ResourceContext

F = TypeVar("F", bound=Callable[..., Any])
_CONTROLLED_TRACE_KEYS = frozenset(
    f"aai.{field_name}" for field_name in ResourceContext.model_fields
)


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

    def __init__(self, native_span: Any, policy: TracePolicy) -> None:
        self.native_span = native_span
        self.policy = policy

    def set_inputs(self, value: Any) -> None:
        self.native_span.set_inputs(sanitize_trace_payload(value, policy=self.policy))

    def set_outputs(self, value: Any) -> None:
        self.native_span.set_outputs(sanitize_trace_payload(value, policy=self.policy))

    def set_attribute(self, key: str, value: Any) -> None:
        sanitized = sanitize_trace_payload({key: value}, policy=self.policy)
        if isinstance(sanitized, Mapping) and key in sanitized:
            value = sanitized[key]
        elif self.policy.capture_mode is TraceCaptureMode.METADATA_ONLY:
            value = _payload_shape(
                value,
                depth=0,
                max_depth=self.policy.max_payload_depth,
                max_collection_items=self.policy.max_collection_items,
            )
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
_PROCESS_TRACE_CONFIGURATION: tuple[Any, ...] | None = None


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

    global _DEFAULT_TRACE_STATE, _PROCESS_TRACE_CONFIGURATION

    if not experiment_name.strip():
        raise ValueError("experiment_name must not be blank")
    if not isinstance(integration, TraceIntegration):
        raise TypeError("integration must be a TraceIntegration")
    mlflow = _require_mlflow()
    selected_policy = policy or TracePolicy()
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
    if _PROCESS_TRACE_CONFIGURATION is not None:
        if signature != _PROCESS_TRACE_CONFIGURATION:
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
    if integration is TraceIntegration.MLFLOW_OPENAI:
        mlflow.openai.autolog(**options)
    elif integration is TraceIntegration.MLFLOW_LANGCHAIN:
        mlflow.langchain.autolog(**options)

    _PROCESS_TRACE_CONFIGURATION = signature
    _DEFAULT_TRACE_STATE = state
    _TRACE_STATE.set(state)
    return state


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

        if inspect.iscoroutinefunction(target):

            @wraps(target)
            async def invoke_traced_async(*args: Any, **kwargs: Any) -> Any:
                state = current_trace_state()
                token = _TRACE_STATE.set(state)
                try:
                    _apply_trace_state(state)
                    return await target(*args, **kwargs)
                finally:
                    _TRACE_STATE.reset(token)

            native_traced_async = mlflow.trace(name=name, span_type=span_type)(
                invoke_traced_async
            )

            @wraps(target)
            async def dispatch_async(*args: Any, **kwargs: Any) -> Any:
                state = current_trace_state()
                if state.policy.capture_mode is TraceCaptureMode.OFF:
                    return await target(*args, **kwargs)
                if not hasattr(mlflow, "start_span"):
                    return await native_traced_async(*args, **kwargs)
                token = _TRACE_STATE.set(state)
                try:
                    if state.policy.capture_mode is TraceCaptureMode.FULL:
                        # MLflow's async decorator may not finalize a trace when
                        # asyncio.CancelledError (a BaseException) escapes on
                        # some supported versions. A native span context exits
                        # synchronously during cancellation, so the trace is
                        # closed before cancellation propagates.
                        with mlflow.start_span(
                            name=name or target.__name__,
                            span_type=span_type or "UNKNOWN",
                        ) as native_span:
                            _apply_trace_state(state)
                            span = GovernedSpan(native_span, state.policy)
                            span.set_inputs(_bound_inputs(target, args, kwargs))
                            result = await target(*args, **kwargs)
                            span.set_outputs(result)
                            return result
                    with provider_span(
                        name or target.__name__,
                        span_type=span_type or "UNKNOWN",
                    ) as span:
                        if span is not None:
                            span.set_inputs(_bound_inputs(target, args, kwargs))
                        result = await target(*args, **kwargs)
                        if span is not None:
                            span.set_outputs(result)
                        return result
                finally:
                    _TRACE_STATE.reset(token)

            return cast(F, dispatch_async)

        @wraps(target)
        def invoke_traced(*args: Any, **kwargs: Any) -> Any:
            state = current_trace_state()
            token = _TRACE_STATE.set(state)
            try:
                _apply_trace_state(state)
                return target(*args, **kwargs)
            finally:
                _TRACE_STATE.reset(token)

        native_traced = mlflow.trace(name=name, span_type=span_type)(invoke_traced)

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
                with provider_span(
                    name or target.__name__,
                    span_type=span_type or "UNKNOWN",
                ) as span:
                    if span is not None:
                        span.set_inputs(_bound_inputs(target, args, kwargs))
                    result = target(*args, **kwargs)
                    if span is not None:
                        span.set_outputs(result)
                    return result
            finally:
                _TRACE_STATE.reset(token)

        return cast(F, dispatch)

    if function is None:
        return decorate
    return decorate(function)


@contextmanager
def provider_span(
    name: str,
    *,
    span_type: str,
    attributes: Mapping[str, Any] | None = None,
) -> Iterator[GovernedSpan | None]:
    """Create a policy-bound native provider span if MLflow is available."""

    state = current_trace_state()
    if state.policy.capture_mode is TraceCaptureMode.OFF or state.integration not in {
        TraceIntegration.SDK,
        TraceIntegration.MLFLOW_AGENT_SERVER,
    }:
        yield None
        return
    try:
        import mlflow
    except ImportError:
        yield None
        return

    policy = current_trace_state().policy
    with mlflow.start_span(name=name, span_type=span_type) as native_span:
        _apply_trace_state(current_trace_state())
        span = GovernedSpan(native_span, policy)
        for key, value in (attributes or {}).items():
            span.set_attribute(key, value)
        yield span


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

    if not session_id.strip():
        raise ValueError("session_id must not be blank")
    try:
        import mlflow
    except ImportError:
        return
    try:
        mlflow.update_current_trace(session_id=session_id)
    except Exception:
        # Session context can only be attached while a trace is active.
        return
    current = current_trace_state()
    _TRACE_STATE.set(replace(current, session_id=session_id))


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
    if session_id is not None and not session_id.strip():
        raise ValueError("session_id must not be blank")
    state = TraceState(
        metadata=merged,
        policy=current.policy,
        integration=current.integration,
        experiment_name=current.experiment_name,
        session_id=session_id if session_id is not None else current.session_id,
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


def native_mlflow() -> Any:
    """Return the native MLflow module for unsupported tracing features."""

    return _require_mlflow()


def native_active_span() -> Any | None:
    """Return MLflow's active span without wrapping or mirroring it."""

    mlflow = _require_mlflow()
    getter = getattr(mlflow, "get_current_active_span", None)
    return getter() if getter is not None else None


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
        items = list(value.items())
        truncated = (
            max_collection_items is not None and len(items) > max_collection_items
        )
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
    if isinstance(value, (list, tuple)):
        items = list(value)
        truncated = (
            max_collection_items is not None and len(items) > max_collection_items
        )
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
    if isinstance(value, str) and max_string_length is not None:
        if len(value) > max_string_length:
            return value[:max_string_length] + "<truncated>"
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return {"type": "bytes", "length": len(value)}
    return {"type": type(value).__name__}


def _is_redacted_key(key: str, redacted_keys: set[str]) -> bool:
    normalized = key.casefold().replace("-", "_").replace(".", "_")
    return any(
        normalized == protected or normalized.endswith(f"_{protected}")
        for protected in redacted_keys
    )


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
        keys = sorted(str(key) for key in value)
        return {
            "type": "mapping",
            "keys": keys[:max_collection_items],
            "size": len(value),
            "truncated": len(keys) > max_collection_items,
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


def _require_mlflow():
    try:
        import mlflow
    except ImportError as error:
        raise RuntimeError(
            "MLflow support requires the `genai` extra. From an aai-core "
            "checkout run `make examples-install` and use `.venv/bin/python`; "
            "in a consuming environment install `aai-core[genai]`."
        ) from error
    return mlflow
