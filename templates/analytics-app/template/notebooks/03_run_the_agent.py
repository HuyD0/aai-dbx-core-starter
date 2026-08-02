# Databricks notebook source
# The runbook loop end to end. Offline by default: a scripted async client
# plays the model role so the tool trajectory, provenance footer, and token
# accounting are all visible deterministically. The scripted client uses the
# same injection seam (async_client=) the tests use — swap in the live path
# at the bottom to run against real endpoints.

# COMMAND ----------

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from aai_core.testing import dev_context, fake_tool_call
from app.agent import AnalyticsAgent
from app.knowledge import KnowledgeRouter
from app.semantics.executor import FakeWarehouseExecutor
from app.semantics.models import load_semantic_model

ROOT = next(
    parent
    for parent in [Path.cwd(), *Path.cwd().parents]
    if (parent / "semantics" / "semantic_model.yml").exists()
)

# COMMAND ----------


class ScriptedAsyncClient:
    """OpenAI-shaped fake: returns queued responses, records requests."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **request):
        self.requests.append(request)
        return self.responses.pop(0)

    async def aclose(self):
        return None


def response(content="", tool_calls=(), tokens=0):
    message = SimpleNamespace(content=content, tool_calls=list(tool_calls))
    usage = SimpleNamespace(
        prompt_tokens=tokens, completion_tokens=0, total_tokens=tokens
    )
    return SimpleNamespace(
        model="scripted", choices=[SimpleNamespace(message=message)], usage=usage
    )


# COMMAND ----------

# Script the runbook: the "model" first queries the semantic layer, then
# returns a structured final answer. The application appends the footer.
plan = {
    "metrics": ["revenue"],
    "filters": [{"dimension": "order_date", "value": "2024-03", "grain": "month"}],
}
final = json.dumps(
    {
        "answer": "Revenue for March 2024 was 600.75.",
        "caveats": ["Excludes cancelled orders."],
    }
)
client = ScriptedAsyncClient(
    [
        response(tool_calls=[fake_tool_call("query_metrics", plan)], tokens=900),
        response(tokens=400),
        response(content=final, tokens=250),
    ]
)

model_fake = SimpleNamespace(
    provider="fake", logical_name="general-chat", model="scripted"
)
context = dev_context()
context.providers.register_model("general-chat", model_fake)

agent = AnalyticsAgent(
    context,
    semantic_model=load_semantic_model(ROOT / "semantics" / "semantic_model.yml"),
    knowledge=KnowledgeRouter(ROOT / "knowledge"),
    executor=FakeWarehouseExecutor(ROOT / "evals" / "data" / "seed_data.json"),
    system_messages=tuple(
        json.loads((ROOT / "prompts" / "system_prompt.json").read_text())["messages"]
    ),
    enable_review=False,
    async_client=client,
)
answer = asyncio.run(agent.aanswer("What was total revenue in March 2024?"))
print(answer.answer)

# COMMAND ----------

# Tokenomics per answer, split by pass. Adversarial review adds a "review"
# pass — the published tradeoff is roughly +6% accuracy for +32% tokens and
# +72% latency, which is why it ships opt-in and off by default.
print(
    {
        "tools_used": answer.tools_used,
        "usage": answer.usage.model_dump(),
        "reviewed": answer.reviewed,
        "tiers": [record.tier.value for record in answer.records],
    }
)

# COMMAND ----------

# Live path (requires configured providers and warehouse grants). Run the
# same question with review off and on, and compare usage.review_tokens:
#
# from aai_core import bootstrap
# from app.config import DEMO_CATALOG, DEMO_SCHEMA, resolve_warehouse_id
# from app.semantics.executor import DatabricksWarehouseExecutor
#
# context = bootstrap()
# executor = DatabricksWarehouseExecutor(
#     warehouse_id=resolve_warehouse_id(None),
#     catalog=DEMO_CATALOG,
#     schema=DEMO_SCHEMA,
# )
# for review in (False, True):
#     agent = AnalyticsAgent.from_project(ROOT, context, executor=executor,
#                                         enable_review=review)
#     try:
#         live = asyncio.run(agent.aanswer("What was revenue in March 2024?"))
#         print({"review": review, "usage": live.usage.model_dump()})
#     finally:
#         asyncio.run(agent.aclose())
