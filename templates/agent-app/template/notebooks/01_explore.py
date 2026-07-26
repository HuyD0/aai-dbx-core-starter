# Databricks notebook source
# ruff: noqa: F704, PLE1142
# Exploration only — production logic lives in src/app and runs as jobs.

# COMMAND ----------

from aai_core import bootstrap
from aai_core.agents import AgentRequest
from aai_core.tracing import TraceIntegration
from app.agent import ToolAgent

context = bootstrap()  # discovers aai-platform.yml (env override / upward search)
context.configure_tracing(integration=TraceIntegration.SDK)
print({"application": context.tags.application})

# COMMAND ----------

agent = ToolAgent(context)
try:
    response = await agent.ainvoke(
        AgentRequest(messages=[{"role": "user", "content": "Where is order A-1001?"}])
    )
    print(response.content)
    print(response.metadata)
finally:
    await agent.aclose()
