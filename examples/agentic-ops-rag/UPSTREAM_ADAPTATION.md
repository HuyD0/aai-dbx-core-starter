# Upstream adaptation record

## Source reviewed

- Repository: [`gregworks/agentic-ops-and-rag`](https://github.com/gregworks/agentic-ops-and-rag)
- Revision: `b5e2482816cd85dcfe5c5df0de7decda6d9caab3`
- Revision date: 2026-08-03

No `LICENSE`, `COPYING`, or `NOTICE` file was present at that revision. To avoid
assuming redistribution permission, this directory is a clean-room adaptation:
it does not copy upstream notebook text, code, images, saved outputs, or Acme
datasets. It credits the high-level teaching sequence and exercise pattern.

## Teaching pattern retained

The most useful upstream pattern is:

```text
observable behavior
  -> focused TODO
  -> executable check
  -> reference explanation
  -> controlled comparison
```

This adaptation uses tagged `# YOUR TURN`, `# CHECK YOUR WORK`, and
`# Reference solution` cells in one generated notebook instead of maintaining a
second set of solution notebooks.

## Module mapping

| Upstream material | Concept retained | AAI adaptation |
|---|---|---|
| Environment verification and prerequisites | Make setup failures visible before a lab | `00_environment_and_stack_map`: locked environment, safe config summary, keyless identity, separate readiness stages |
| Architecting pipelines | Routing, pre-filtering, staged latency, failure boundaries | `01_routing_filters_and_action_boundaries`: strict routes, trusted tenant/region scope, secret refusal, human action approval |
| Chunking, embeddings, and indexing | Structure-aware chunks and embedding/index compatibility | `02_chunking_embeddings_and_index_release`: stable chunks, MLflow fields, `ChunkingProfile`, `EmbeddingProfile`, release evidence |
| Retrieval and reranking | Exact versus semantic retrieval, RRF, reranking trade-offs | `03_hybrid_retrieval_and_reranking`: text/vector/hybrid/semantic matrix through managed-search concepts |
| Guardrails, observability, and evaluation | Abstention, traces, row-level diagnosis, regression gates | `04_mlflow_tracing_guardrails_and_evaluation`: MLflow 3 spans, deterministic critical checks, native RAG judges |
| Four-configuration capstone | Hold inputs constant and choose from evidence | `05_capstone_release_decision`: baseline/change/result/decision, absolute and regression gates, `ApplicationRelease` |

## Deliberate replacements

- Runtime package installation is replaced by the repository's certified lock.
- Raw vendor-key checks are replaced by Azure CLI, managed/workload identity,
  and Databricks unified authentication.
- FAISS/HNSW/IVF implementation tuning is not the production architecture. A
  transparent local fixture explains ranking; Azure AI Search and Databricks AI
  Search are the connected providers.
- Custom lexical “LLM judge” functions are replaced by deterministic policy
  checks plus MLflow GenAI scorers over real retriever traces.
- Custom tracer objects are replaced by one governed MLflow tracing owner.
- Embedded duplicate datasets are replaced by versionable JSONL sources.
- Aggregate-only quality is replaced by fixed cases, critical authorization and
  refusal rows, coverage, row-level inspection, and baseline regression.
- “Wrap it in FastAPI” is replaced by the repository's `rag-app` and
  `agent-app` templates, MLflow Agent Server, and Databricks Apps.

## Known differences worth teaching

- Offline embeddings, latency, and generation remain transparent fixtures. They
  teach contracts and experiment shape, not cloud-provider performance.
- Managed search owns index algorithms. Application engineers evaluate chunking,
  embedding compatibility, filters, retrieval mode, reranking, and outcomes;
  they do not tune FAISS `nprobe` inside production notebooks.
- Abstention is reported alongside answerable coverage. It is not awarded a
  perfect faithfulness score.
- A guardrail evaluates the same retrieved evidence that reaches generation;
  the pipeline does not perform a second, inconsistent retrieval.
- Small fixture p95 values are explicitly labelled and never treated as an SLA.
- Every action is a proposal until an external human checkpoint approves it;
  no notebook executes a production side effect.
