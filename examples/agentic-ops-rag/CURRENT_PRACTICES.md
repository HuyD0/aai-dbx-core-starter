# Current practices review

Reviewed against official documentation on **2026-08-03**. This is a dated
teaching baseline, not a substitute for checking service availability in the
target tenant and region before a production release.

## Recommended stack baseline

| Concern | Stable workshop and production path | Optional or evolving path |
|---|---|---|
| Serving | Databricks Apps with MLflow Agent Server and the Responses API contract | Models-from-code serving compatibility; Foundry Hosted Agents |
| Retrieval | Azure AI Search classic hybrid BM25 + vector, optionally followed by semantic ranker | Azure AI Search agentic retrieval after platform/API-status review |
| Lifecycle evidence | MLflow 3 traces, fixed evaluation data, `mlflow.genai.evaluate()`, and deterministic release gates | Unity Catalog trace storage and production monitoring when enabled |
| Identity | Entra ID, Azure CLI for development, managed/workload identity in hosted environments, Databricks unified auth | No API keys, PATs, or client secrets |
| Code layout | Packaged Python under `src/`; notebooks teach and gather evidence | Provider-native experiments behind an explicit opt-in |

## What changed from the source course

| Source-course approach | This adaptation |
|---|---|
| Colab cells install packages at runtime | One exact repository lock and a named kernel |
| Optional raw vendor keys | Keyless Entra and Databricks unified authentication |
| FAISS and custom hash/vector demonstrations as the main architecture | Transparent offline fixture for learning; Azure AI Search and Databricks AI Search for connected retrieval |
| Custom tracing and faithfulness arithmetic | Governed MLflow spans plus deterministic rules and native RAG judges |
| One tuned dataset and aggregate score | Fixed row-level cases, authorization/adversarial rows, baseline regression, and explicit coverage |
| Generic “wrap it in FastAPI” graduation | `rag-app` or `agent-app`, MLflow Agent Server, and Databricks Apps |

## MLflow 3

MLflow 3 is the evidence system for development and production GenAI
applications: tracing, evaluation, feedback, prompt/application versioning, and
comparison. Use `mlflow.genai.evaluate()` for GenAI evaluation; do not mix its
`Scorer` objects with classic `mlflow.models.evaluate()` metrics.

RAG judges require real trace evidence. A `RETRIEVER` span emits a list of
documents with `page_content`; metadata carries `doc_uri` and `chunk_id`, with
an optional `id`. The `aai-core` search adapters already normalize Azure and
Databricks results to that shape.

Use deterministic checks for tenant isolation, secret refusal, citations,
schemas, exact tool names/arguments, idempotency, and human approval. Add
`RetrievalRelevance`, `RetrievalGroundedness`, `RetrievalSufficiency`, and
`Safety` as judge evidence. Experimental tool-call judges should not be the sole
hard gate.

References:

- [MLflow 3 for GenAI](https://learn.microsoft.com/en-us/azure/databricks/mlflow3/genai/)
- [MLflow evaluation harness](https://learn.microsoft.com/en-us/azure/databricks/mlflow3/genai/eval-monitor/concepts/eval-harness)
- [MLflow span concepts](https://learn.microsoft.com/en-us/azure/databricks/mlflow3/genai/tracing/span-concepts)
- [MLflow RAG judges](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/rag/)

Unity Catalog trace storage is promising for new production workloads but has
platform prerequisites. Production monitoring is currently Beta. The workshop
therefore uses experiment traces and offline evaluation as the runnable base;
UC storage and scheduled monitoring are platform-enabled TODOs, not prerequisites.

## Azure AI Search

Classic hybrid retrieval remains the stable default. Azure executes keyword
BM25 and vector queries in parallel and merges their rankings with Reciprocal
Rank Fusion. Semantic ranker then reranks the fused candidates and returns a
separate reranker score. Benchmark the modes on application data; do not compare
or threshold BM25, cosine, RRF, and reranker scores as if they share a scale.

When semantic ranking is enabled, Microsoft recommends a vector candidate set
of about 50 so the reranker has enough input. Keep that candidate count separate
from the smaller context sent to the model. The current `aai-core` adapter has
one `top_k`; lesson 03 deliberately retrieves 50 and slices the application
context rather than hiding that capability gap.

For tenant and security filters, use pre-filtering so unauthorized documents do
not enter ranking, tracing, or generation. Stable security trimming uses
filterable identity/scope fields. Native document ACL enforcement continues to
have preview surfaces and requires a separate platform decision.

This workshop's connected helper maps that policy explicitly. Azure AI Search
uses `allowed_groups/any(...search.in(...))` over a filterable string
collection. Databricks AI Search standard endpoints support `ARRAY<STRING>`
contains-any filters. Storage-optimized endpoints do not currently support
array filtering, so the helper fails closed until the platform publishes a
compatible scalar ACL field; it never falls back to post-filtering retrieved
content in application code.

Both mappings also filter on `active=true` before ranking and return tenant,
region, group, active-state, runbook-code, and effective-date evidence. The
deterministic gate verifies each scope dimension independently, including
same-tenant documents restricted to a different group. The application keeps
the newest active revision of each runbook code and never interprets an
arbitrary positive provider score as proof that a result supports the query.
Its raw provider-candidate `retriever.search` span is nested beneath a top-level
`retriever.final_context` `RETRIEVER` span that records exactly the individually
supported documents supplied to generation, cited in the response, and seen by
MLflow retrieval scorers.

The stable 2026 API also supports query-time integrated vectorization through a
configured vectorizer. This SDK currently uses an explicit client-side query
embedding. A future adapter may add a typed `VectorizableTextQuery` mode, but it
must never silently switch embedding ownership; index-time and query-time
vectorizers must use the same embedding space.

References:

- [Hybrid search overview](https://learn.microsoft.com/en-us/azure/search/hybrid-search-overview)
- [Hybrid RRF scoring](https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking)
- [Vector query guidance](https://learn.microsoft.com/en-us/azure/search/vector-search-how-to-query)
- [Vector filtering](https://learn.microsoft.com/en-us/azure/search/vector-search-filters)
- [Security trimming](https://learn.microsoft.com/en-us/azure/search/search-security-trimming-for-azure-search)
- [Keyless client authentication](https://learn.microsoft.com/en-us/azure/search/search-security-rbac-client-code)
- [Databricks AI Search filtering](https://docs.databricks.com/aws/en/ai-search/filtering-guide)

Agentic retrieval adds query planning and knowledge-base capabilities, but parts
of the end-to-end experience still depend on preview APIs. Treat it as an
advanced experiment with explicit tenant, region, SLA, API-version, evaluation,
and rollback evidence—not as the default workshop architecture.

## Databricks and serving

For custom agents, Databricks Apps is the primary path when the application
needs custom server behavior, Git-based delivery, or local IDE development.
Use MLflow Agent Server and the Responses API contract. An App has a dedicated
service principal; the platform process provisions it and grants least
privilege. Deploy code only after CI and evaluation, then load test before
production. Apps do not scale to zero, so the repository keeps them stopped by
default and uses explicit start/restart/stop operations.

References:

- [Author a custom agent on Databricks Apps](https://learn.microsoft.com/en-us/azure/databricks/agents/custom-agents/author-agent)
- [Productionize an agent](https://learn.microsoft.com/en-us/azure/databricks/agents/agent-framework/productionize-agent)
- [Databricks authentication](https://learn.microsoft.com/en-us/azure/databricks/dev-tools/auth/)
- [Azure CLI authentication for Databricks](https://learn.microsoft.com/en-us/azure/databricks/dev-tools/auth/azure-cli)

## Microsoft Foundry

Foundry is a configured model/provider plane in this workshop, not a parallel
release system. Hosted Agents provide managed identity, containerized code,
sessions, endpoints, and OpenTelemetry integration, but several surrounding
observability capabilities remain preview. Do not dual-export every prompt and
response to MLflow and Application Insights by default; that duplicates
sensitive content across retention, RBAC, and cost boundaries. Demonstrate dual
export only after a privacy and operations review.

References:

- [Foundry Hosted Agents](https://learn.microsoft.com/en-us/azure/foundry/agents/concepts/hosted-agents)
- [Foundry agent tracing setup](https://learn.microsoft.com/en-us/azure/foundry/observability/how-to/trace-agent-setup)

## Review checklist for a future refresh

- Recheck stable versus preview status for agentic retrieval, native document
  ACLs, UC trace storage, and production monitoring.
- Confirm the certified versions in `pyproject.toml`, `dependency-policy.toml`,
  `uv.lock`, and `compatibility.json` still agree.
- Re-run the text/vector/hybrid/reranked matrix on representative production
  cases; never carry fixture thresholds into a new provider or index.
- Verify trace retention, access, and redaction before enabling connected data.
- Confirm serving, app identity, search roles, and judge endpoints through the
  external platform process.
