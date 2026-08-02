# Current practices and source guide

Reviewed on **2026-08-01** against primary Microsoft, MLflow, Databricks, and
A2A documentation. Recheck preview APIs before promoting a lab into production.

## Recommended baseline

| Subject | Current practice | Status in this curriculum |
|---|---|---|
| Foundry SDK | Use `azure-ai-projects` 2.x, the Responses API, Conversations, and immutable agent versions. Avoid legacy Threads/Runs and classic Connected Agents samples. | Current |
| Conversation state | Use a conversation or `previous_response_id` for message history. Treat a Hosted Agent session as a separate sandbox/filesystem boundary. | Current |
| Context engineering | Layer stable instructions, scoped state, provenance-bearing retrieval, bounded recent history, and typed memory. Use `ContextProvider` for just-in-time context and Responses compaction for long histories. | Context providers and compaction current; managed Memory preview |
| A2A | Target A2A v1.0 for new integrations. Discover a versioned Agent Card, authenticate with Microsoft Entra ID, and delegate the minimum context needed. | Foundry A2A public preview |
| Foundry evaluation | Start with response-ID smoke tests, then frozen datasets or immutable agent targets. Require every evaluator to complete and pass its declared threshold. | Cloud evaluation current; trace/conversation paths preview |
| MLflow evaluation | Use `mlflow.genai.evaluate()` and GenAI scorers, not classic `mlflow.models.evaluate()` metrics. Turn reviewed production traces into versioned regression datasets. | MLflow 3.14 baseline |
| Tracing | Build one OpenTelemetry provider and attach multiple exporters when the same trace must reach multiple backends. Keep content capture off unless policy explicitly permits it. | Current OTel pattern |
| Foundry trace UI | Connect Application Insights, export spans to Azure Monitor, include agent and conversation correlation, then allow for ingestion delay. | Required prerequisite |
| MLflow OTel ingestion | OSS MLflow accepts OTLP/HTTP at `/v1/traces`, requires a SQL backend, and routes with `x-mlflow-experiment-id`. | Available since MLflow 3.6 |
| Databricks production trace storage | Prefer Unity Catalog trace tables for governed storage. Use a collector or gateway for keyless forwarding instead of putting a static Databricks bearer token in a Hosted Agent version. | MLflow 3.14+; UC feature limitations apply |

## What the Foundry A2A preview supports today

- Foundry supports A2A 1.0 and 0.3; new integrations should select 1.0 through
  the versioned Agent Card.
- Foundry's A2A v1.0 endpoint is JSON-RPC only.
- The current preview is text-only and doesn't support server-sent-event
  streaming, files, or other nontext artifacts.
- Incoming A2A requires the Responses protocol and Microsoft Entra
  authentication. The least-privilege calling role is Foundry Agent Consumer.
- Agent Cards, protocol enablement, project connections, and role assignments
  are platform provisioning. The notebooks only discover and invoke an already
  enabled endpoint.

The curriculum lock certifies Agent Framework Core 1.12.1, whose context
provider hooks are `before_run(...)` and `after_run(...)`. Older preview samples
using `invoking()` / `invoked()` are not copied into the runnable notebook.

Foundry safety evaluators also depend on regional Responsible AI capability.
An evaluator with `status=error` or a `null` score is failed evidence even when
the overall evaluation run reports `completed`; choose a supported region
through the approved platform process rather than dropping the safety gate.

## Why an empty Foundry Traces page is possible

Sending a normal Responses request does not by itself populate the Foundry
Traces page. The project must have Application Insights connected, the client
must export OpenTelemetry spans to that resource, and an agent trace needs the
agent/conversation correlation fields Foundry uses. The viewer also needs the
relevant Log Analytics permissions. Client-side traces commonly take a few
minutes to appear.

The SDK method named `get_application_insights_connection_string()` can return
either a full connection string or, on a legacy-linked project, only the
instrumentation-key GUID. A GUID alone can leave the Azure Monitor exporter
without a valid regional ingestion endpoint. The dual-export notebook uses the
configured non-secret Application Insights resource ID to resolve the full
connection string from Azure Resource Manager at runtime, without logging or
persisting either live telemetry value.

MLflow is a separate trace store. A trace is visible in both systems only when
the originating OpenTelemetry pipeline exports the same spans to both Azure
Monitor and MLflow. Foundry does not ingest an MLflow trace, and MLflow does not
pull a Foundry trace automatically. There is no backend synchronization.

## Primary sources

- [Foundry Responses API and compaction](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/responses)
- [Foundry client-side tracing](https://learn.microsoft.com/en-us/azure/foundry/observability/how-to/trace-agent-client-side)
- [Enable incoming Foundry A2A](https://learn.microsoft.com/en-us/azure/foundry/agents/how-to/enable-agent-to-agent-endpoint)
- [Use a remote A2A endpoint from Foundry](https://learn.microsoft.com/en-us/azure/foundry/agents/how-to/tools/agent-to-agent)
- [A2A protocol specification](https://a2a-protocol.org/latest/specification/)
- [Agent Framework `ContextProvider`](https://learn.microsoft.com/en-us/python/api/agent-framework-core/agent_framework.contextprovider?view=agent-framework-python-latest)
- [Foundry managed Memory](https://learn.microsoft.com/en-us/azure/foundry/agents/how-to/memory-usage)
- [Foundry cloud evaluation](https://learn.microsoft.com/en-us/azure/foundry/how-to/develop/cloud-evaluation)
- [MLflow OpenTelemetry ingestion](https://mlflow.org/docs/latest/genai/tracing/opentelemetry/ingest/)
- [MLflow GenAI evaluation](https://mlflow.org/docs/latest/genai/eval-monitor/running-evaluation/)
- [MLflow custom scorers](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/custom/)
- [Databricks Unity Catalog trace storage](https://docs.databricks.com/aws/en/mlflow3/genai/tracing/trace-unity-catalog)
