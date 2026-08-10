# Databricks notebook source
# Production monitoring (Beta): register sampled scorers on the deployed
# agent's traces. scorer.register()/.start() must run FROM A NOTEBOOK (the
# service serializes notebook code), which is why this is not wrapped in
# aai-core or automated in CI. Mind the per-experiment scorer cap and keep
# sample rates low — judges cost money on every sampled trace.
#
# Managed preflight is mandatory before register()/start():
# 1. Confirm Production Monitoring (Beta) is enabled for this workspace.
# 2. Confirm this is the experiment receiving the deployed agent's traces and
#    that an approved serverless budget policy covers sampled scorer execution.
# 3. If the experiment stores traces in Unity Catalog, confirm an available SQL
#    warehouse and the required catalog, schema, and trace-table permissions.
# This notebook does not enable Beta, attach budget policy, create a warehouse,
# or grant permissions. The platform owner verifies those managed prerequisites.
#
# Registered-production-scorer input contract:
# - A sampled invocation receives the trace; it does not receive a benchmark
#   row's reviewed expectations.
# - decision_action_consistency can reuse the release scorer's semantic rule,
#   but its registered implementation must be self-contained @scorer code
#   defined in this notebook. Registration serializes notebook code; importing
#   app.tool_scoring's factory does not serialize that factory and its helpers.
# - decision_tool_appropriateness requires reviewed expected_tool_calls, so it
#   cannot be registered unchanged. Keep it in dataset-backed release evaluation
#   and curate reviewed production failures into that dataset first.

# COMMAND ----------

import mlflow
from mlflow.genai.scorers import Safety, ScorerSamplingConfig

from aai_core import bootstrap
from aai_core.providers.types import ProviderConfigurationError

context = bootstrap()  # discovers aai-platform.yml (env override / upward search)
print({"experiment": context.settings.effective_experiment_name})


def judge_model_uri(settings) -> str:
    config = settings.models.get("judge-model")
    if not config or config.get("provider") != "databricks":
        raise ProviderConfigurationError(
            "judge-model must resolve to a governed Databricks serving endpoint"
        )
    # Registered production scorers require the Databricks judge URI scheme;
    # endpoints:/ is valid for direct evaluation but is rejected on register().
    return f"databricks:/{config['deployment']}"


MANAGED_MONITORING_PREFLIGHT_COMPLETE = False


def require_managed_monitoring_preflight(confirmed: bool) -> None:
    """Fail closed until the platform owner completes the managed checklist."""

    if not confirmed:
        raise RuntimeError(
            "Before register/start, confirm Production Monitoring Beta, the "
            "traced experiment and serverless budget policy, and, for Unity "
            "Catalog trace storage, a SQL warehouse plus catalog/schema/table "
            "permissions. This notebook does not provision them."
        )


# COMMAND ----------

require_managed_monitoring_preflight(MANAGED_MONITORING_PREFLIGHT_COMPLETE)
mlflow.set_experiment(context.settings.effective_experiment_name)

safety = Safety(model=judge_model_uri(context.settings)).register(
    name="production_safety"
)
safety = safety.start(sampling_config=ScorerSamplingConfig(sample_rate=0.1))
print("registered production_safety at 10% sampling")

# COMMAND ----------

# Attach end-user/expert feedback to the originating trace with the thin
# native helpers in app.feedback. Only human-reviewed expectations are marked
# as curation-ready for evals/data/release_cases.json:
#
#   from app.feedback import record_human_feedback
#   record_human_feedback(trace_id=..., name="user_helpful", value=False,
#                         source_id="support-reviewers",
#                         rationale="answer ignored the order id")
