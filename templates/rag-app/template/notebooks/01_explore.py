# Databricks notebook source
# MAGIC %md
# MAGIC # Explore the Agentic RAG paved road
# MAGIC
# MAGIC This notebook is for interactive investigation. Move reusable logic into
# MAGIC `src/app` and add evaluation cases before deployment.

# COMMAND ----------

from aai_core import bootstrap
from aai_core.tracing import configure_tracing

ctx = bootstrap()  # discovers aai-platform.yml (env override / upward search)
configure_tracing(
    ctx.tags,
    experiment_name=ctx.settings.effective_experiment_name,
)

display(  # noqa: F821 - supplied by the Databricks notebook runtime
    {
        "application": ctx.tags.application,
        "environment": ctx.tags.environment,
        "model": "general-chat",
        "retriever": "product-knowledge",
    }
)
