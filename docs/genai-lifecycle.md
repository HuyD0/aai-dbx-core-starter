# GenAI and RAG lifecycle

## Lifecycle

1. Define the use case, risks, success metrics, and initial evaluation cases.
2. Instrument the application with MLflow traces.
3. Register and version prompts.
4. Capture model, tool, retrieval, and guardrail spans.
5. Build evaluation datasets from reviewed examples and real traces.
6. Evaluate the candidate against absolute thresholds and the deployed baseline.
7. Record an immutable application release.
8. Deploy through a protected bundle workflow.
9. Monitor sampled traces, quality, latency, errors, and cost.
10. Attach user/expert feedback and promote failures into regression cases.

This is an evaluation-driven loop, not a one-way release pipeline. Reviewed
production failures become evaluation records, and the same scorer definitions
are reused for candidate regression tests and production monitoring.

## RAG trace shape

```text
request
  query rewrite
  embedding
  retrieval
  reranking
  context assembly
  generation
  citation validation
```

Retriever outputs use MLflow's document schema:

```json
{
  "id": "document-or-chunk-id",
  "page_content": "retrieved text",
  "metadata": {
    "doc_uri": "governed source",
    "chunk_id": "stable chunk identifier"
  }
}
```

Do not trace complete sensitive documents unless the application's data policy
explicitly permits it. Prefer stable identifiers and controlled excerpts.

## Evaluation layers

- Unit and schema tests.
- Deterministic policy and tool tests.
- Retrieval and access-control evaluation.
- Response, groundedness, citation, and safety scoring.
- Human/domain review.
- Production sampled evaluation.

The same scorers should be reused before and after deployment so quality does
not mean something different in production.

## Offline and automatic evaluation

Use `mlflow.genai.evaluate()` before deployment for candidate comparison,
regression tests, and release gates. Configure MLflow automatic evaluation on
sampled development and production traces for ongoing quality monitoring.
Development can use a high sampling rate; production sampling should balance
coverage, judge cost, latency, and data-handling requirements. Filter automatic
evaluation to the intended environment and trace status.

The application team owns evaluation cases, scorer intent, and acceptance
thresholds. The platform team owns approved judge deployments, scorer
configuration controls, sampling guardrails, dashboards, and alert routing.

## Prompt promotion

Prompt versions are immutable. The `development`, `candidate`, and `production`
aliases are controlled, mutable pointers used for promotion. Evaluation and
release evidence must bind the exact prompt version even when runtime
configuration loads an alias.

## Current references

- [MLflow GenAI evaluation and monitoring](https://mlflow.org/docs/latest/genai/eval-monitor/)
- [MLflow automatic evaluation](https://mlflow.org/docs/latest/genai/eval-monitor/automatic-evaluations/)
- [MLflow Prompt Registry](https://mlflow.org/docs/latest/genai/prompt-registry/)
- [MLflow Tracing](https://mlflow.org/docs/latest/genai/tracing/)
