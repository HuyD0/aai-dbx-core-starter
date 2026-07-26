"""Agent deployment adapter: Databricks Model Serving behind one entry point.

Keeping deployment behind this seam means a future target (for example
Databricks Apps) is an SDK change, not a template change. Requires the
``databricks-agents`` package at call time (generated projects pin it in
requirements.lock; an ``aai-core[serving]`` extra lands with the next lock
regeneration).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from aai_core.exceptions import AaiCoreError

if TYPE_CHECKING:
    from aai_core.context import PlatformContext
    from aai_core.runtime import PlatformSettings


class ServingError(AaiCoreError):
    code = "aai_core.serving.error"


def agent_resources(
    settings: PlatformSettings,
    *,
    models: Sequence[str] = (),
    retrievers: Sequence[str] = (),
    uc_functions: Sequence[str] = (),
) -> list[Any]:
    """Translate logical names into MLflow resource declarations.

    Passed to ``mlflow.pyfunc.log_model(resources=...)`` so Model Serving
    grants the deployed agent authenticated access to exactly the endpoints,
    indexes, and UC functions it uses — never hand-maintain these.
    """

    for logical_name in models:
        config = settings.models.get(logical_name) or {}
        if config.get("provider") != "databricks":
            raise ServingError(
                f"Model {logical_name!r} is not a Databricks serving endpoint",
                remediation="Deployed agents must call gateway-fronted "
                "Databricks endpoints (external models for Foundry); update "
                "aai-platform.yml.",
            )

    from mlflow.models.resources import (
        DatabricksFunction,
        DatabricksServingEndpoint,
        DatabricksVectorSearchIndex,
    )

    resources: list[Any] = []
    for logical_name in models:
        config = settings.models.get(logical_name) or {}
        resources.append(DatabricksServingEndpoint(endpoint_name=config["deployment"]))
    for logical_name in retrievers:
        config = settings.retrievers.get(logical_name) or {}
        if config.get("provider") == "databricks_ai_search":
            resources.append(DatabricksVectorSearchIndex(index_name=config["index"]))
    for function_name in uc_functions:
        resources.append(DatabricksFunction(function_name=function_name))
    return resources


def deploy_agent(
    context: PlatformContext,
    *,
    uc_model_name: str,
    version: int | str,
    scale_to_zero: bool = True,
    agents_module: Any | None = None,
) -> Any:
    """Deploy a UC-registered agent model version to Model Serving.

    Sets the experiment first (deployment tracing lands in the project's
    experiment, not the folder default), then delegates to
    ``databricks.agents.deploy``. Runs on the credentialed path only, after
    the release gate has passed.
    """

    if agents_module is None:
        try:
            from databricks import agents as agents_module  # type: ignore[no-redef]
        except ImportError as error:
            raise ServingError(
                "Agent deployment requires the databricks-agents package",
                remediation="pip install databricks-agents (generated "
                "projects pin it in requirements.lock).",
            ) from error

    import mlflow

    mlflow.set_experiment(context.settings.effective_experiment_name)
    return agents_module.deploy(
        uc_model_name,
        int(version) if str(version).isdigit() else version,
        scale_to_zero=scale_to_zero,
        tags=dict(context.tags.for_databricks()),
    )
