# GenAI and RAG lifecycle

## Operating contract

Every AI application change must produce one connected evidence chain:

```text
hypothesis
  → baseline
  → one deliberate change
  → result
  → decision
  → immutable release
```

The words are intentional. A trace is an observation, an experiment compares
controlled runs, an evaluation measures behavior against cases, and a release
is the exact application state approved by a decision. None is a synonym for
another.

## Why the evidence chain matters

An LLM can produce a convincing answer while omitting a fact, inventing a
source, violating policy, or consuming far more time and tokens than expected.
Reading one response cannot reveal whether the behavior is repeatable or
whether a prompt change caused the result. The lifecycle turns those unknowns
into explicit, reviewable evidence.

For a Python developer new to MLflow:

| MLflow object | Question it answers | Risk it reduces |
|---|---|---|
| Prompt version | Which exact reusable instructions ran? | An edited alias or copied string makes the result impossible to reproduce. |
| Trace | What happened during this one request? | Failures, latency, tokens, and cost remain hidden inside one function call. |
| Span | Which prompt, model, retrieval, or tool operation caused it? | A team sees a bad output but cannot locate the responsible step. |
| Run | What inputs, settings, metrics, and artifacts did this test produce? | A result is discussed without enough context to repeat it. |
| Experiment | Which baseline and change runs belong to this decision? | Unrelated attempts are compared or evidence is lost. |
| Evaluation | Did behavior meet defined checks on representative cases? | A fluent demo response is mistaken for release readiness. |

The Prompt Registry is the version history for reusable prompt templates. It
is important for the same reason source control is important for code: a team
must know exactly which instructions produced a behavior, review the isolated
change, and load that immutable version again.

## Running example: fictional earnings summaries

The progressive examples make this contract concrete with a fictional Aster
Ridge Systems earnings-summary assistant. All issuer names, figures, excerpts,
and source identifiers are synthetic and must not be treated as market data or
investment advice. The assistant is explicitly prohibited from recommending
whether to buy, sell, or hold an investment.

The baseline `earnings_summary` prompt summarizes only facts in a supplied
`earnings_excerpt`. The changed prompt adds exactly one requirement: include
the supplied `source_id` once. Both prompts are tested against the same three
cases:

1. Quarterly revenue and operating margin.
2. Forward revenue and margin guidance.
3. Free cash flow, inventory growth, and supplier risk.

This controlled design is important. When only the citation requirement
changes, differences in citation rate, quality, latency, tokens, and cost can
be attributed to that prompt change. The full deterministic evaluation may
adopt `earnings-summary-prompt-v2`; earlier examples remain blocked because
observing or registering behavior is not enough to authorize a release.

## Lifecycle

1. Define the use case, risks, success metrics, and initial evaluation cases.
2. State a falsifiable hypothesis and identify the deployed or accepted baseline.
3. Instrument the application with governed MLflow traces.
4. Register and load exact immutable prompt versions.
5. Capture model, tool, retrieval, prompt, and guardrail spans.
6. Build versioned evaluation datasets from reviewed examples and real traces.
7. Evaluate one change against absolute thresholds and the same-case baseline.
8. Record the result, decision, and immutable application release.
9. Deploy through a protected bundle workflow.
10. Monitor sampled traces, quality, latency, errors, cost, and cost coverage.
11. Attach user/expert feedback and turn reviewed failures into regression cases.

This is an evaluation-driven loop, not a one-way release pipeline. Reviewed
production failures become evaluation records. Reuse the same scorer intent—and
the same implementation only when the runtime inputs match—across change
regression tests and production monitoring.

## Experiment identity and run lineage

An experiment is a stable container for one application decision area, not a
single attempt. Name it for the application and the decision it supports, for
example:

```text
/Shared/example-ai-earnings-summary-quality-cost
```

Run names identify the comparison role and version:

```text
baseline-earnings-summary-prompt-v1
change-cited-earnings-summary-prompt-v2
```

Do not use `first-comparison`, `test-2`, a model name alone, or a bare
timestamp. The searchable run contract records:

- `run_purpose`: baseline, change, result, decision, monitoring, or exploration;
- a stable `change_id` and concise `change_summary`;
- the hypothesis;
- `baseline_run_id` on the change/result/decision evidence;
- the application model ID when MLflow LoggedModel lineage is available.

The experiment name stays stable while exact run IDs, prompt URIs, dataset
digests, source revisions, and release digests make each result reproducible.

## Tracking and tracing boundary

Tracking records the experiment, run purpose, parameters, datasets, metrics,
artifacts, result, and decision. Tracing records one application execution and
its nested prompt, model, retriever, tool, and guardrail operations. A good
example teaches both and does not pretend that one replaces the other.

Select exactly one process-startup `TraceIntegration`. Use SDK-managed provider
spans for stable `model.generate()`, MLflow OpenAI autologging for direct native
OpenAI sync/async/stream calls, MLflow LangChain autologging for
LangChain/LangGraph, or Agent Server as the root owner with one selected child
path. Never enable or disable an autologger per request and never instrument
the same logical operation twice.

Prompt registration and loading are not model/framework inference calls.
There is therefore nothing for OpenAI or LangChain autologging to capture.
Track them with a governed run, exact prompt-version links and digests, and,
when useful, a manual `PROMPT` span around safe registration/loading/rendering
metadata. Do not enable an unrelated autologger merely to make an example look
instrumented.

Every trace policy states whether content capture is off, metadata-only,
redacted, bounded, or explicitly approved in full. Prompts, evidence, tool
arguments, and outputs must follow the application's data classification.
For new production workloads on Databricks, prefer Unity Catalog trace storage
when the workspace supports it: Databricks recommends it for governed access,
SQL-queryable OpenTelemetry tables, and avoiding the experiment trace-storage
cap. Treat the trace location as platform configuration, not application code.

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

The `agentkit` CLI implements this chain for agent evaluation — it decides the
run ontology, resolves scorers from the shared registry, and generates the
lineage described below. See `docs/agent-evaluation.md`.

- Unit and schema tests.
- Deterministic policy and tool tests.
- Retrieval and access-control evaluation.
- Response, groundedness, citation, and safety scoring.
- Human/domain review.
- Production sampled evaluation.

Use the same scorer semantics before and after deployment so quality does not
mean something different in production, but respect their input contracts. A
registered production scorer receives a trace without benchmark expectations.
Trace-only decision/action consistency can be reproduced as self-contained
notebook-defined `@scorer` code; registration serializes notebook code, not an
imported scorer factory and its helpers. Expectation-dependent decision/tool
appropriateness cannot be registered unchanged and remains a reviewed,
dataset-backed development/release assessment.

## Offline and automatic evaluation

Use `mlflow.genai.evaluate()` before deployment for baseline/change comparison,
regression tests, and release gates. Where the workspace preview is approved,
configure Databricks production monitoring on sampled development and
production traces for ongoing quality monitoring; this managed feature is
currently Beta and must not become an undeclared release dependency.
Development can use a high sampling rate; production sampling should balance
coverage, judge cost, latency, and data-handling requirements. Filter automatic
evaluation to the intended environment and trace status. Route every LLM
scorer through the explicit approved `judge-model`; never rely on a provider's
ambient default. Any row-level error from a gated scorer fails the release
because MLflow aggregates otherwise omit failed rows.

There is no public `promote_trace` API in the certified MLflow 3.15.1 surface.
After minimization and human review, use the MLflow UI Add/Export flow or call
native `dataset.merge_records(reviewed_traces)`. Trace
conversion carries root inputs/outputs, expectation Assessments, and source
trace/session lineage; it does not copy the full span tree into the dataset row.
Normalize reviewed behavior contracts such as `expected_tool_calls` before
using them as gate expectations. `aai-core` deliberately adds no stable wrapper
for this native boundary, so live-validate managed UI and conversion behavior
in the target workspace before automating it.

The application team owns evaluation cases, scorer intent, and acceptance
thresholds. The platform team owns approved judge deployments, scorer
configuration controls, sampling guardrails, dashboards, and alert routing.

Deterministic checks come first: schema, required facts, exact citation
occurrence, tool names and arguments, policy constraints, and critical-row
assertions. Add an LLM judge only where semantic judgment is actually required.
Stochastic applications need repeated evaluations; record the repetition count
and report dispersion rather than treating one run as stable evidence.

Every comparison binds the exact same ordered dataset version or canonical
digest. Dataset records carry provenance and split membership. Reviewed
production failures may enter a regression split, but must not silently mutate
the held-out set used to authorize release.

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

Calibration is measured, not asserted: `agentkit judge calibrate` computes
chance-adjusted agreement (Cohen's κ, target ≥ 0.60) between the judge and
the reviewer consensus, plus the pairwise inter-annotator ceiling — a judge
cannot be more consistent than the humans defining the target, and a low
ceiling means the rubric is under-specified, so fix the rubric before
touching the model. The committed per-judge record binds scorer version,
served judge identity, and prompt URI; evidence always reports it, and
`integrity.require_calibration` makes scoring and the gate demand it. The
auditable claim is never "the agent scores 0.87" — it is "0.87 under a judge
that agrees with our reviewers at κ 0.71 against a ceiling of 0.78".

The judge also stays measured *inside* every judged run: a re-judged sample
of the run's own outputs bounds how much of any delta is judge noise, and
frozen judge anchors separate "the judge moved" from "the agent regressed".
A judge release is therefore its own lifecycle, out-of-band from agent
changes: new prompt version and scorer version from the platform team,
re-calibration on held-out labels, then a re-established baseline and
anchors — in separate commits from any agent change, because a
champion-versus-change delta is a statement about the agent only while the
dataset and the judge were held constant.

Prompt optimization is an optional experiment, not a deployment action. Bound
its request and cost budget, train on a dedicated split, and run the resulting
exact prompt version against held-out cases through the normal gate. Only then
may a controlled alias move. Optimizers and alignment algorithms remain native
MLflow APIs rather than stable `aai-core` wrappers.

## Cost-quality decisions

Run the baseline and changed model/application against the same cases and
record quality, latency, input/output token usage, trace cost, and coverage for
missing usage/cost metadata. Prefer MLflow-recorded trace cost and governed
gateway/billing data; never embed changeable vendor price tables in
`aai-core`. Treat cost per quality point as a comparison aid, while safety and
minimum quality remain independent release requirements.

Unknown cost is not zero cost. Record observation count, known count, known
subtotal, and coverage. A total is valid only at complete coverage. A release
gate either requires a declared minimum coverage or explicitly records that
cost is report-only; it must not silently pass because cost telemetry is
missing.

Framework autologgers are opt-in because they can capture raw call arguments.
The stable provider adapters emit bounded spans and canonical token usage
without additive provider options, raw `extra_body` payloads, or per-call
credential headers. Do not combine those adapter spans with OpenAI
autologging, which duplicates traces and token counts.

## Model customization

Prompting, retrieval, and fine-tuning form one escalation ladder, not three
competing options. A prompt change alters instructions, a retrieval change
alters the context the model reads, and fine-tuning alters the weights
themselves. Escalate only when the cheaper lever is exhausted: fine-tuning is
justified when repeated prompt versions and retrieval improvements still fail
a behavioral requirement — output format, tone, citation discipline,
consistency — on the governed evaluation set.

Parameter-efficient fine-tuning (LoRA and its variants) changes behavior, not
knowledge. It teaches a model how to respond in the required shape; it is not
a mechanism for injecting new facts, which remains retrieval's job. A team
reaching for fine-tuning to fix factual gaps should fix its retrieval and
evaluation design instead.

A fine-tuned adapter is an application release like every other change in the
reproducibility manifest. It carries the same falsifiable hypothesis, the same
baseline/change/result/decision contract, the same fixed ordered evaluation
dataset, and the same release gate as a prompt change. Training data are
versioned evidence: record the dataset digest, split membership, filtering
rules, and pre-training validation results exactly as for evaluation datasets,
and keep training, validation, and held-out evaluation splits disjoint. The
adapter artifact, base model identity and revision, hyperparameters (rank,
alpha, learning rate, epochs), and environment digest all enter the manifest.
An adapter version is immutable; retraining creates a new version and a new
comparison, never an overwrite.

Serving cost changes shape at this stage. A pay-per-token endpoint bills
nothing when idle; a dedicated fine-tuned endpoint bills continuously whether
or not it serves traffic. A cost-quality comparison between a fine-tuned small
model and a prompted larger model must therefore use expected request volume,
not per-request price alone. `docs/cost-estimation.md` prices both shapes:
GPU model serving, and foundation models as provisioned throughput or
pay-per-token.

Fine-tuning compute and serving endpoints are provisioned through the approved
platform process like every other resource change, and training jobs stay
keyless under the same workspace authentication chain as deployment. The
standalone course in `examples/fine-tuning/` teaches these mechanics from
first principles, credential-free — beginning with the memory economics
above, with later lessons building through quantization, LoRA, QLoRA, and
the PEFT library — before any connected training job.

## Prompt promotion

Prompt versions are immutable. Mutable aliases such as `development` and
`production` are controlled deployment pointers, never release evidence.
Some upstream tools and older project versions call a pre-release alias
`candidate`; that prompt alias is deprecated here in favor of `validation`.
It is distinct from the schema-v2 application maturity tag
`ResourceContext.lifecycle="validation"`. Historical schema-v1 evidence with
the `candidate` lifecycle remains readable with a deprecation warning.
Evaluation and release evidence always bind the exact prompt URI, version, and
content digest even when runtime configuration loads an alias. Promotion also
cites the finished MLflow decision
run: the SDK reloads `decision/decision.json` and verifies its digest, lifecycle
tags, gate metrics, run purpose, identity, and status before moving the alias.

## Reproducibility manifest

A result is reproducible only when it records, as applicable:

- source commit and whether the source tree was clean or dirty;
- application release or MLflow LoggedModel ID;
- logical model name and resolved physical provider/model/deployment;
- model parameters and seed when the provider supports one;
- exact prompt URI/version and content digest;
- ordered dataset ID/version or canonical digest and split;
- scorer names, rubrics, judge model, and judge alignment version;
- tool-schema, retriever/index, embedding, and chunking versions;
- dependency/environment digest;
- repetition count for stochastic behavior;
- quality, latency, token, cost, and cost-coverage evidence.

Changing code, model, prompt, tool, index, embedding, or chunking creates a new
application change and must go through the same baseline/result/decision loop.

## Progressive learning examples

The repository examples implement this contract in order:

1. `00_offline_hello_world.py` proves provider-neutral contracts and represents
   unknown cost explicitly.
2. `01_first_trace.py` records one bounded fictional earnings-summary execution
   so a developer can inspect what happened inside the request.
3. `02_first_experiment.py` compares the named baseline and changed
   earnings-summary runs on the same ordered dataset digest.
4. `03_first_prompt.py` registers `earnings_summary` idempotently and binds exact
   immutable prompt versions without an irrelevant framework autologger.
5. `04_first_evaluation.py` uses native deterministic MLflow scorers and
   row-critical release checks before adopting `earnings-summary-prompt-v2`.
6. `05_connected_setup.ipynb` separates kernel/config readiness from keyless
   Azure identity, workspace membership, and endpoint readiness without making
   a model request.
7. `06_connected_first_call.py` makes one real call through the stable
   synchronous `model.generate()` path with bounded SDK tracing and explicit
   unknown cost.
8. `07_first_llm_call.ipynb` demonstrates an approved OpenAI autolog path through
   a native async client and stream, compares both exact prompts on three
   synthetic cases, reads trace-level token/cost evidence, and keeps release
   blocked until the full evaluation passes.
9. `08_tool_trajectory_evaluation.ipynb` rejects a wrong exact tool trajectory
   even when the final answer contains every expected fact.
10. `09_multi_turn_session_evaluation.ipynb` scopes traces by opaque session,
    release, environment, and evaluation batch before applying session-level
    metrics and critical-case gates.
11. `10_layered_judges.ipynb` keeps exact rules deterministic and measures a
    nuanced judge against balanced human feedback on a held-out split.
12. `11_cost_quality_tradeoff.ipynb` filters logical model candidates through
    quality and policy gates before comparing known, covered cost.
13. `12_agent_alignment_optimization.ipynb` keeps judge calibration, optimizer
    training, and final held-out release evidence separate; it is disabled by
    default and cannot move a production alias.
14. `13_decision_and_promotion_lifecycle.ipynb` ends the comparison in a
    recorded adopt/reject decision bound to its gate evidence and shows the
    production alias refusing to move without adopt-grade evidence.
15. `14_platform_llm_operations.ipynb` walks the platform team's operating
    loop: judge governance, gateway request tags, cost by tag, fleet
    provenance, monitoring adoption, and rollback levers.
16. `15_compare_and_select_llms.ipynb` is a credential-free interactive
    workshop that compares baseline/change logical models through a golden
    dataset, blinded pairwise judging, session-level TCO, and a fail-closed
    governance evidence preflight. Its separate learner workshop contains the
    intentional stubs; the canonical notebook is complete, and its simulated
    measurements cannot authorize a release.

The executable lifecycle scripts emit hypothesis, baseline, change, result,
decision, and release. The advanced notebooks preserve that vocabulary while
remaining exploration and teaching artifacts. See
[the executable curriculum](../examples/README.md).

## Current references

- [LLMOps playbook](llmops-playbook.md)
- [Cookbook relevance assessment](mlflow-cookbook-assessment.md)
- [MLflow GenAI evaluation and monitoring](https://mlflow.org/docs/latest/genai/eval-monitor/)
- [MLflow automatic evaluation](https://mlflow.org/docs/latest/genai/eval-monitor/automatic-evaluations/)
- [MLflow Prompt Registry](https://mlflow.org/docs/latest/genai/prompt-registry/)
- [MLflow Tracing](https://mlflow.org/docs/latest/genai/tracing/)
- [Databricks MLflow 3 tracing and Unity Catalog trace storage](https://docs.databricks.com/aws/en/mlflow3/genai/tracing)
- [Databricks development evaluation harness](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/concepts/eval-harness)
- [Databricks prompt-version comparison best practices](https://docs.databricks.com/aws/en/mlflow3/genai/prompt-version-mgmt/prompt-registry/evaluate-prompts)
- [Databricks production monitoring](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/production-monitoring)
- [LangSmith evaluation concepts](https://docs.langchain.com/langsmith/evaluation-concepts)
- [LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
- [LangGraph interrupts and idempotency](https://docs.langchain.com/oss/python/langgraph/interrupts)
- [Production LangGraph agents in this repository](langgraph-production.md)
- [Multi-agent systems in production in this repository](multi-agent-systems.md)
- [W3C Trace Context](https://www.w3.org/TR/trace-context/)
- [OpenTelemetry GenAI conventions — currently Development](https://github.com/open-telemetry/semantic-conventions-genai)
