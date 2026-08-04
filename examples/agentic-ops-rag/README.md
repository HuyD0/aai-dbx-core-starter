# Agentic operations and RAG workshop

This is an original, credential-free-first adaptation of the teaching sequence
in Greg Swanson's
[`agentic-ops-and-rag`](https://github.com/gregworks/agentic-ops-and-rag/tree/b5e2482816cd85dcfe5c5df0de7decda6d9caab3)
course. It teaches the same useful progression—pipeline architecture, chunking,
hybrid retrieval, reranking, evaluation, and a capstone—but implements it with
this repository's governed SDK and templates.

The scenario is a synthetic operations assistant. It retrieves fictional
runbooks, diagnoses fictional incidents, refuses credential requests, and can
propose but never execute a production action. The data is not copied from the
upstream project and is not operational guidance.

## What you will build

```text
trusted request scope
  -> deterministic route
  -> text / vector / hybrid retrieval
  -> optional managed reranker
  -> bounded context with citations
  -> answer or abstention
  -> action proposal with human approval
  -> MLflow trace and release gate
```

The default path uses transparent local fixtures and makes no network request.
Connected cells are disabled until you deliberately set `RUN_CONNECTED = True`.

## One-time setup

From the repository root:

```bash
make ops-rag-install
make ops-rag-doctor
make ops-rag-notebook
```

`ops-rag-install` synchronizes the repository's exact `uv.lock`; notebooks do
not install or upgrade packages. `ops-rag-doctor` checks the kernel, data,
configuration shape, and optional provider packages without contacting a cloud.

Run the complete credential-free workshop gate with:

```bash
make ops-rag-check
```

## Optional connected setup

Choose one retrieval provider. For Azure AI Search:

```bash
cp examples/agentic-ops-rag/config/aai-platform.azure-search.example.yml \
  examples/agentic-ops-rag/config/aai-platform.yml
```

For Databricks AI Search, copy the adjacent Databricks example instead. Both
files expose the same logical resources:

| Logical name | Application capability |
|---|---|
| `operations-chat` | grounded answer generation |
| `operations-embedding` | query and document embedding space |
| `operations-knowledge` | text, vector, or hybrid retrieval |
| `judge-model` | governed MLflow GenAI judges |

Replace every `replace-with-*` value with resources supplied by the approved
platform process. Do not create an index, endpoint, identity, role assignment,
catalog, or volume from this example.

Authenticate keylessly:

```bash
az login
export DATABRICKS_AUTH_TYPE=azure-cli
make ops-rag-doctor CONNECTED=1
```

For Azure AI Search queries, the selected Entra principal normally needs the
least-privilege `Search Index Data Reader` role on the approved scope. Index
writers are separate identities and workflows. The configuration contains
identifiers and secret references only—never add an API key, PAT, client
secret, or raw Key Vault value.

## Course map

| Lesson | Outcome |
|---:|---|
| `00_environment_and_stack_map` | Safe preflight, identity/config boundaries, and the stack responsibility map. |
| `01_routing_filters_and_action_boundaries` | Exact-code routing, trusted tenant filters, secret refusal, and approval before side effects. |
| `02_chunking_embeddings_and_index_release` | Structural chunks, MLflow document fields, embedding compatibility, and index release evidence. |
| `03_hybrid_retrieval_and_reranking` | Text/vector/hybrid/reranked ablation, RRF, candidate budgets, and Azure semantic ranking. |
| `04_mlflow_tracing_guardrails_and_evaluation` | Retriever spans, deterministic gates, governed runs, and optional MLflow RAG judges. |
| `05_capstone_release_decision` | Baseline/change/result/decision, immutable release evidence, and template graduation. |

Every lesson has a focused `# YOUR TURN` exercise, a non-destructive check,
and a reference solution. The notebooks are generated from reviewable Python
under `scripts/`; edit that source and rerender instead of hand-editing JSON.

## How this maps to production

- `aai-core` owns configuration, resource context, provider resolution,
  normalized search results, trace policy, and release-gate contracts.
- Native MLflow owns experiments, traces, prompt/application lineage,
  `mlflow.genai.evaluate()`, built-in judges, and feedback.
- Azure AI Search or Databricks AI Search owns managed indexing and retrieval.
  The application asks for `operations-knowledge`, not a physical index name.
- Microsoft Foundry or a Databricks endpoint supplies models through configured
  logical resources. Enterprises can place a governed gateway in front without
  changing application architecture.
- Production code graduates into [`rag-app`](../../templates/rag-app/) or
  [`agent-app`](../../templates/agent-app/). The primary custom-agent HTTP path
  is MLflow Agent Server on Databricks Apps.

See [CURRENT_PRACTICES.md](CURRENT_PRACTICES.md) for the dated standards review
and [UPSTREAM_ADAPTATION.md](UPSTREAM_ADAPTATION.md) for the clean-room mapping.

## Safety and evidence boundaries

- All documents, incidents, tenants, timings, and costs are synthetic.
- `simulated_offline_fixture` latency teaches evidence shape; it is not an SLA.
- Cost remains unknown when provider pricing evidence is absent.
- The app service principal's search scope is platform state, not proof of an
  individual developer's access.
- Retrieved text is untrusted data. It cannot override system instructions,
  access scope, tool allowlists, or approval policy.
- A notebook never moves a prompt alias, deploys an app, or executes an
  operational action.
