"""MLflow tracing helpers with a no-op fallback for non-GenAI installs."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any, TypeVar

from aai_core.tags import ResourceContext

F = TypeVar("F", bound=Callable[..., Any])


def configure_tracing(
    context: ResourceContext,
    *,
    experiment_name: str,
    tracking_uri: str | None = None,
    enable_openai_autolog: bool = True,
) -> None:
    mlflow = _require_mlflow()
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    if enable_openai_autolog:
        mlflow.openai.autolog()
    set_trace_context(context.for_trace())


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
        return mlflow.trace(name=name, span_type=span_type)(target)

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


def _require_mlflow():
    try:
        import mlflow
    except ImportError as error:
        raise RuntimeError(
            "MLflow support requires `pip install 'aai-core[genai]'`"
        ) from error
    return mlflow
