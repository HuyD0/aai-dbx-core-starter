"""LLM judge configuration (tier 2 only — needs model access).

Judges are routed through the platform's gateway-fronted judge endpoint (the
`judge-model` logical name in aai-platform.yml). Calibrate judges against
human labels before trusting them in the gate — see
notebooks/01_align_judge.py.
"""

from __future__ import annotations

from aai_core.providers.types import ProviderConfigurationError


def judge_model_uri(settings) -> str:
    """Resolve the judge endpoint into an mlflow judge model URI."""

    config = settings.models.get("judge-model")
    if not config:
        raise ProviderConfigurationError(
            "aai-platform.yml has no judge-model entry",
            remediation="Add providers.models.judge-model with the serving "
            "endpoint the LLM judges should use.",
        )
    if config.get("provider") == "databricks":
        return f"endpoints:/{config['deployment']}"
    raise ProviderConfigurationError(
        "LLM judges need a Databricks serving endpoint",
        remediation="Route the judge model through a (gateway-enabled) "
        "Databricks serving endpoint — for Foundry models, use an external "
        "model endpoint — and set provider: databricks on judge-model.",
    )


def judge_scorers(settings) -> list:
    """Built-in judges pinned to the configured judge endpoint. Add
    Guidelines-based judges here for domain rubrics, e.g.
    Guidelines(name="tone", guidelines="...", model=judge_model_uri(settings)).
    """

    from mlflow.genai.scorers import Correctness, Safety

    model = judge_model_uri(settings)
    return [Correctness(model=model), Safety(model=model)]
