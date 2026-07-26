"""MLflow tracing helpers with a no-op fallback for non-GenAI installs."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from functools import wraps
from typing import Any, TypeVar, cast

from aai_core.tags import ResourceContext

F = TypeVar("F", bound=Callable[..., Any])
_TRACE_METADATA: dict[str, str] = {}


def configure_tracing(
    context: ResourceContext,
    *,
    experiment_name: str,
    tracking_uri: str | None = None,
    enable_openai_autolog: bool = False,
    enable_langchain_autolog: bool = False,
) -> None:
    """Configure governed tracing integrations for an experiment.

    OpenAI and LangChain autologging are opt-in because framework integrations
    can capture raw SDK/framework arguments. The stable provider adapters emit
    their own bounded spans and token usage; do not combine their calls with
    OpenAI autologging because duplicate spans also double-count usage. Enable
    an autologger for direct native/framework calls only after confirming its
    inputs comply with the application's trace data policy.
    """

    global _TRACE_METADATA

    mlflow = _require_mlflow()
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    if enable_openai_autolog:
        mlflow.openai.autolog()
    if enable_langchain_autolog:
        mlflow.langchain.autolog()
    _TRACE_METADATA = dict(context.for_trace())


def traced(
    function: F | None = None,
    *,
    name: str | None = None,
    span_type: str | None = None,
) -> F | Callable[[F], F]:
    """Apply ``mlflow.trace`` when MLflow is installed, otherwise do nothing."""

    def decorate(target: F) -> F:
        try:
            import mlflow
        except ImportError:
            return target

        if inspect.iscoroutinefunction(target):

            @wraps(target)
            async def invoke_async(*args: Any, **kwargs: Any) -> Any:
                if _TRACE_METADATA:
                    set_trace_context(_TRACE_METADATA)
                return await target(*args, **kwargs)

            return cast(
                F,
                mlflow.trace(name=name, span_type=span_type)(invoke_async),
            )

        @wraps(target)
        def invoke(*args: Any, **kwargs: Any) -> Any:
            if _TRACE_METADATA:
                set_trace_context(_TRACE_METADATA)
            return target(*args, **kwargs)

        return cast(F, mlflow.trace(name=name, span_type=span_type)(invoke))

    if function is None:
        return decorate
    return decorate(function)


@contextmanager
def provider_span(
    name: str,
    *,
    span_type: str,
    attributes: Mapping[str, Any] | None = None,
) -> Iterator[Any]:
    """Create a provider span if MLflow is available."""

    try:
        import mlflow
    except ImportError:
        yield None
        return

    with mlflow.start_span(name=name, span_type=span_type) as span:
        if _TRACE_METADATA:
            set_trace_context(_TRACE_METADATA)
        for key, value in (attributes or {}).items():
            span.set_attribute(key, value)
        yield span


def set_trace_context(metadata: Mapping[str, str]) -> None:
    try:
        import mlflow
    except ImportError:
        return
    try:
        mlflow.update_current_trace(metadata=dict(metadata))
    except Exception:
        # No trace is active. Configuration remains valid and the metadata will
        # be applied by callers when a trace starts.
        return


def set_trace_session(session_id: str) -> None:
    """Group the active trace into an MLflow conversation session.

    Uses MLflow's dedicated ``session_id`` field rather than ordinary trace
    metadata. Calling this when no trace is active is a no-op, matching
    :func:`set_trace_context`.
    """

    try:
        import mlflow
    except ImportError:
        return
    try:
        mlflow.update_current_trace(session_id=session_id)
    except Exception:
        # Session context can only be attached while a trace is active.
        return


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
