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
evaluation to the intended environment and trace status. Route every LLM
scorer through the explicit approved `judge-model`; never rely on a provider's
ambient default. Any row-level error from a gated scorer fails the release
because MLflow aggregates otherwise omit failed rows.

The application team owns evaluation cases, scorer intent, and acceptance
thresholds. The platform team owns approved judge deployments, scorer
configuration controls, sampling guardrails, dashboards, and alert routing.

## Agent trajectories and conversations

Agent evaluation must score both the final answer and the path used to produce
it. Trace each tool execution as an MLflow `TOOL` span and represent expected
calls with names, arguments, and multiplicity. Use deterministic exact
matching for release gates unless multiple paths are intentionally equivalent;
fuzzy tool matching invokes an LLM judge and has different cost and
repeatability. Compatibility checks must cover empty expected trajectories and
duplicate-call multiplicity, not only the set of unique calls.

For multi-turn applications, accept complete conversation history on each
request or use an application-owned durable state store. Attach the same
opaque session id to every turn's trace; do not keep production conversations
in an unbounded process-global dictionary. Evaluate complete sessions for
unresolved requests, policy drift, safety, knowledge retention, and user
frustration. Convert categorical scorer output into explicit numeric gate
metrics, and let critical scenarios fail individually rather than disappear
inside an average. Do not forward or trace a raw user identifier without an
approved pseudonymization and data-handling design.

## Judge alignment and optimization

LLM judges are measurements, not ground truth. Record human assessments with
explicit provenance and rationale, calibrate on one reviewed split, and verify
agreement on another before allowing the judge into a release gate. Keep judge
model, rubric, examples, and alignment state versioned.

Prompt optimization is an optional experiment, not a deployment action. Bound
its request and cost budget, train on a dedicated split, and run the resulting
exact prompt version against held-out cases through the normal gate. Only then
may a controlled alias move. Optimizers and alignment algorithms remain native
MLflow APIs rather than stable `aai-core` wrappers.

## Cost-quality decisions

Run candidate models against the same cases and record quality, latency,
input/output token usage, trace cost, and coverage for missing usage/cost
metadata. Prefer MLflow-recorded trace cost and governed gateway/billing data;
never embed changeable vendor price tables in `aai-core`. Treat cost per
quality point as a comparison aid, while safety and minimum quality remain
independent release requirements.

Framework autologgers are opt-in because they can capture raw call arguments.
The stable provider adapters emit bounded spans and canonical token usage
without additive provider options, raw `extra_body` payloads, or per-call
credential headers. Do not combine those adapter spans with OpenAI
autologging, which duplicates traces and token counts.

## Prompt promotion

Prompt versions are immutable. The `development`, `candidate`, and `production`
aliases are controlled, mutable pointers used for promotion. Evaluation and
release evidence must bind the exact prompt version even when runtime
configuration loads an alias.

## Current references

- [Cookbook relevance assessment](mlflow-cookbook-assessment.md)
- [MLflow GenAI evaluation and monitoring](https://mlflow.org/docs/latest/genai/eval-monitor/)
- [MLflow automatic evaluation](https://mlflow.org/docs/latest/genai/eval-monitor/automatic-evaluations/)
- [MLflow Prompt Registry](https://mlflow.org/docs/latest/genai/prompt-registry/)
- [MLflow Tracing](https://mlflow.org/docs/latest/genai/tracing/)
