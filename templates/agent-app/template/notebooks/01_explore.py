# Databricks notebook source
# Exploration only — production logic lives in src/app and runs as jobs.

# COMMAND ----------

from pathlib import Path

from aai_core import bootstrap
from aai_core.agents import AgentRequest
from app.agent import ToolAgent

context = bootstrap(Path.cwd().parent / "aai-platform.yml")
print({"application": context.tags.application})

# COMMAND ----------

agent = ToolAgent(context)
response = agent.invoke(
    AgentRequest(messages=[{"role": "user", "content": "Where is order A-1001?"}])
)
print(response.content)
print(response.metadata)
