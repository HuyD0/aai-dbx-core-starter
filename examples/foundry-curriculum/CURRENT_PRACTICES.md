---
last_verified: 2026-08-09
review_by: 2026-11-07
verification_scope:
  - primary Microsoft Foundry and Azure Databricks documentation
  - primary MLflow and A2A documentation
  - locked dependency and credential-free notebook contracts
certified_versions:
  agent-framework-core: 1.12.1
  azure-ai-projects: 2.4.0
  mlflow: 3.15.1
  opentelemetry-sdk: 1.43.0
live_validation: required before production use
---

# Current practices and source guide

Reviewed on **2026-08-09** against primary Microsoft, MLflow, Databricks, and
A2A documentation. Recheck preview APIs before promoting a lab into production.
The `review_by` date is a maximum review interval, not a promise of stability:
recheck any preview feature, authentication contract, regional limitation, or
provider capability immediately before a connected exercise or release.

## Recommended baseline

| Subject | Current practice | Status in this curriculum |
|---|---|---|
| Foundry SDK | Use `azure-ai-projects` 2.x, the Responses API, Conversations, and immutable agent versions. Avoid legacy Threads/Runs and classic Connected Agents samples. | Current |
| Conversation state | Use a conversation or `previous_response_id` for message history. Treat a Hosted Agent session as a separate sandbox/filesystem boundary. | Current |
| Context engineering | Layer stable instructions, scoped state, provenance-bearing retrieval, bounded recent history, and typed memory. Use `ContextProvider` for just-in-time context and Responses compaction for long histories. | Context providers and compaction current; managed Memory preview |
| A2A | Target A2A v1.0 for new integrations. Discover a versioned Agent Card, authenticate with Microsoft Entra ID, and delegate the minimum context needed. | Foundry A2A public preview |
| Foundry evaluation | Start with response-ID smoke tests, then frozen datasets or immutable agent targets. Require every evaluator to complete and pass its declared threshold. | Cloud evaluation current; trace/conversation paths preview |
| MLflow evaluation | Use `mlflow.genai.evaluate()` and GenAI scorers, not classic `mlflow.models.evaluate()` metrics. Turn reviewed production traces into versioned regression datasets. | MLflow 3.15 baseline |
| Tracing | Build one OpenTelemetry provider and attach multiple exporters when the same trace must reach multiple backends. Keep content capture off unless policy explicitly permits it, and evaluate every instrumentation owner separately. | Current OTel pattern |
| Foundry trace UI | Connect Application Insights, export spans to Azure Monitor, include agent and conversation correlation, then allow for ingestion delay. Application Insights is the operational store/view, not the assurance evaluator. | Required prerequisite |
| MLflow OTel ingestion | OSS MLflow accepts OTLP/HTTP at `/v1/traces`, requires a SQL backend, and routes with `x-mlflow-experiment-id`. | Available since MLflow 3.6 |
| Databricks production trace storage | Prefer Unity Catalog trace tables for governed storage. The managed workspace receiver is `/api/2.0/otel/v1/traces` with `X-Databricks-UC-Table-Name`; use direct export only with renewable short-lived auth, otherwise put token refresh in a collector/gateway. | MLflow 3.15+; UC feature limitations apply |
| Assurance evidence | Treat MLflow traces, reviewed EvaluationDatasets, evaluation runs, and Feedback / Assessments as the authoritative evidence plane. Promote production traces only after minimization and human review. | Required lifecycle practice |

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

## Certified native span shape and its gaps

For `agent-framework-core==1.12.1`, a normal tool loop emits this application
subtree:

```text
invoke_agent <agent_name>
  +-- chat <model_name>
  +-- execute_tool <tool_name>
  +-- chat <model_name>
```

The `chat` and `execute_tool` spans are direct siblings beneath
`invoke_agent`; the TOOL is not nested under the model span that requested it.
Parallel tool calls preserve that parent. A Foundry hosted protocol runtime can
add a server-side parent/root, and managed ingestion can translate span types,
so validate that outer hierarchy live rather than freezing it into an offline
test.

The native spans cover execution, not the whole assurance vocabulary:

| Native signal | MLflow assurance interpretation | Important gap |
|---|---|---|
| `invoke_agent` | Agent invocation / behavior scope | No concise decision reason, evidence references, or confidence. |
| `chat` | LLM/model execution | A tool call in model output is intent, not proof of execution. |
| `execute_tool` | Actual local tool execution | Approval requests have no tool-execution span until the tool really runs. |
| Span status plus exception event | Observed failure | No separate retry or recovery decision. |
| Workflow/executor/message spans | Routing and causal links | No portable human-approval semantic span. |

MLflow's OTel ingestion translates recognized GenAI semantic conventions, but
the precise managed display/type mapping is version- and backend-dependent.
Keep native names and attributes in tests, and validate the managed rendering
before making dashboards or gates depend on it.

## Ownership, privacy, and routing boundaries

SDK-owned client instrumentation and Foundry-native telemetry are independent
producers. The application controls the client OTel provider, exporters, and
`AIProjectInstrumentor` content/context/baggage switches. Foundry controls
server/hosted protocol instrumentation and the platform-injected Application
Insights route. A content-off setting for one producer is not a global
redaction policy: exception text, IDs, tool definitions, and custom attributes
still require review. Optional provider-supported reasoning is diagnostic data,
not a decision record, and hidden chain-of-thought is never requested or
reconstructed.

Receiver contracts must not be conflated:

- OSS MLflow uses OTLP/HTTP `/v1/traces`, `x-mlflow-experiment-id`, and a SQL
  backend.
- Managed Databricks MLflow uses the workspace
  `/api/2.0/otel/v1/traces` endpoint plus
  `X-Databricks-UC-Table-Name` for the target spans table.
- A direct managed exporter needs a short-lived token it can renew. An immutable
  Hosted Agent environment cannot safely hold a one-hour token forever.
- A production collector/gateway owns credential acquisition and refresh, adds
  the managed routing header, and lets the application emit OTLP without a
  frozen Databricks bearer token.

## Debugging and production learning loop

Debug in this order: local in-memory span tree and IDs; one provider and privacy
policy; exporter protocol, endpoint, headers, flush, and renewable auth;
Application Insights ingestion; Foundry agent/conversation correlation and
viewer RBAC; MLflow ingestion and semantic translation; independent scorers.
An answer alone proves none of those hops.

The governed production loop is:

```text
production trace
  -> select and minimize
  -> human review plus MLflow Feedback
  -> approved versioned MLflow EvaluationDataset case
  -> deterministic checks and calibrated judges
  -> regression gate and lifecycle decision
```

Raw traffic is not benchmark truth. Do not fabricate a trace from an offline
fixture, and do not claim retry, recovery, or human intervention unless the
observed execution records it.

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
- [MLflow OpenTelemetry attribute mapping](https://mlflow.org/docs/latest/genai/tracing/opentelemetry/attribute-mapping/)
- [Databricks Unity Catalog trace storage](https://learn.microsoft.com/en-us/azure/databricks/mlflow3/genai/tracing/trace-unity-catalog)
- [Databricks OTLP clients and renewable collector authentication](https://learn.microsoft.com/en-us/azure/databricks/ingestion/opentelemetry/configure)
- [Agent Framework 1.12.1 observability source](https://github.com/microsoft/agent-framework/blob/711d6f24aeaf1b842a21bad059ef70112e701901/python/packages/core/agent_framework/observability.py)
