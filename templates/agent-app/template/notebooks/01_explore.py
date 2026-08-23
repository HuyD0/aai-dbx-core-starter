# Databricks notebook source
# ruff: noqa: F704, PLE1142
# Exploration only — production logic lives in src/app and runs as jobs.

# COMMAND ----------

import mlflow

from aai_core import bootstrap
from aai_core.agents import AgentRequest
from aai_core.tracing import TraceIntegration, traced
from app.agent import ToolAgent
from app.tool_scoring import trace_execution_success_scorer

context = bootstrap()  # discovers aai-platform.yml (env override / upward search)
context.configure_tracing(integration=TraceIntegration.SDK)
print({"application": context.tags.application})

# COMMAND ----------

# MAGIC %md
# MAGIC ## Assurance evidence, not chain-of-thought
# MAGIC
# MAGIC Review four separate layers: **outcome** (the final response),
# MAGIC **behavior** (decision and TOOL spans), **operations** (status, latency,
# MAGIC tokens, and cost), and optional provider-supported internal diagnostics.
# MAGIC Production assurance relies on the first three. A decision span is a
# MAGIC concise application claim; the following TOOL span is the authoritative
# MAGIC record of what executed, and an evaluator later attaches its independent
# MAGIC Assessment. A decision reason is intentionally absent when trace policy is
# MAGIC metadata-only. `trace_execution_success` summarizes root/TOOL status as
# MAGIC operations evidence; it does not grade the answer or trajectory.

# COMMAND ----------

agent = ToolAgent(context)


@traced(name="agent.explore", span_type="AGENT")
async def invoke_agent():
    return await agent.ainvoke(
        AgentRequest(messages=[{"role": "user", "content": "Where is order A-1001?"}])
    )


try:
    response = await invoke_agent()
    print(response.content)
    print(response.metadata)
finally:
    await agent.aclose()

# COMMAND ----------

mlflow.flush_trace_async_logging()
trace_id = mlflow.get_last_active_trace_id()
trace = mlflow.get_trace(trace_id, flush=True)
for span in trace.data.spans:
    span_type = getattr(span.span_type, "value", span.span_type)
    if span_type == "AGENT" and span.name.startswith("decision."):
        reason = span.get_attribute("agent.decision.reason")
        print(
            {
                "layer": "behavior: decision claim",
                "span": span.name,
                "selected_action": span.get_attribute("agent.decision.selected_action"),
                "reason": reason,
                "reason_visibility": (
                    "captured" if reason is not None else "omitted by trace policy"
                ),
            }
        )
    elif span_type == "TOOL":
        status = getattr(getattr(span, "status", None), "status_code", None)
        print(
            {
                "layer": "behavior: observed execution",
                "span": span.name,
                "status": getattr(status, "value", status),
            }
        )

execution_feedback = trace_execution_success_scorer()(trace=trace)
print(
    {
        "layer": "operations",
        "assessment": execution_feedback.name,
        "value": execution_feedback.value,
        "rationale": execution_feedback.rationale,
    }
)
