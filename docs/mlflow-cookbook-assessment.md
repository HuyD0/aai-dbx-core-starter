# MLflow cookbook relevance assessment

Reviewed on 2026-07-26 against the repository's minimum MLflow 3.14 API and
current MLflow 3.14 lock.

The seven cookbooks describe a useful evaluation-driven lifecycle, but they
are examples rather than platform policy. Adopt the durable trace, evaluation,
and promotion shapes below; keep framework-specific and experimental MLflow
APIs native and opt-in.

## Coverage and decisions

| Cookbook | Repository coverage | Decision |
|---|---|---|
| [Agent optimization pipeline](https://mlflow.org/cookbook/agent-alignment-optimization/) | Prompt versions, native MLflow evaluation datasets and feedback, deterministic regression gates, and controlled aliases exist. The judge-alignment notebook still does not execute agreement measurement and there is no optimizer workflow. | Keep optimizer and assessment objects native. Complete judge calibration and teach optional optimization with separate calibration, training, and held-out data. Never let an optimizer promote `production`; the normal release gate remains authoritative. |
| [LangGraph agent](https://mlflow.org/cookbook/langgraph-agent/) | The application-owned agent loop emits standard `TOOL` spans, and the agent template includes a native async LangGraph 1.2 recipe with injected async checkpointer/store, interrupts, idempotency, and a behavioral canary. | Keep MLflow LangChain autologging opt-in and keep LangGraph out of core. Maintain the recipe against the certified range without wrapping graph or state APIs. |
| [Multi-turn agent](https://mlflow.org/cookbook/multi-turn-agent/) | `AgentRequest` and the agent template now preserve governed conversation history and bind an opaque session id to each trace. | Retain the core session seam and add a focused conversational-evaluation recipe. State belongs in the application or a durable framework store, never a process-global core dictionary. |
| [Custom LLM judges](https://mlflow.org/cookbook/custom-llm-judges/) | The evaluation template now combines deterministic scorers, routed Correctness/Safety judges, and an executable native `Guidelines` domain judge. It scaffolds a calibration/validation split but does not yet compute or record agreement. | Keep the custom judge report-only until held-out human calibration supports a threshold. Keep `Guidelines`, `make_judge`, `Feedback`, and registered scorers as native MLflow objects. |
| [Evaluation-driven development](https://mlflow.org/cookbook/eval-driven-development/) | Strong coverage: reviewed cases, offline and credentialed tiers, per-release baselines, absolute/regression gates, tracked runs, production-failure curation, critical-case rules in the learning path, and bounded per-row failure triage in the evaluation template. Scorer errors on gated rows fail rather than disappearing from aggregates. | Retain the current design and extend critical-case rules consistently across templates; aggregate improvements alone are not enough evidence. |
| [Prompt engineering lifecycle](https://mlflow.org/cookbook/prompt-engineering/) | Strong coverage: immutable prompt versions, exact-version evaluation, controlled aliases, and gated promotion. | Retain the stricter repository behavior. A prompt change is an application release; do not rely on a long-running service hot-reloading a moved alias without an evaluated rollout. Add few-shot and side-by-side comparison guidance. |
| [Cost-quality trade-off](https://mlflow.org/cookbook/cost-quality-tradeoff/) | The progressive example compares baseline/change quality, latency, tokens, cost, and cost coverage on the same cases. SDK gates treat unknown cost as unknown and support lower-is-better metrics. | Keep vendor prices out of core. Extend the same comparison to selectable logical models using MLflow-recorded cost and the platform gateway/billing source of truth. |

## Core boundary

The stable SDK now provides:

- one process-startup OpenAI, LangChain, Agent Server, or SDK tracing owner;
- bounded task-local resource/session context for MLflow traces;
- synchronous non-streaming `model.generate()` plus actual native sync and
  caller-owned native async clients;
- serializable Pydantic agent request/response contracts, without an SDK tool
  registry or execution loop;
- `MetricRule`, `GatePolicy`, `GateResult`, and `apply_gate()` over native
  `mlflow.genai.evaluate()` results;
- governed experiment/run context and reproducibility evidence while native
  MLflow retains dataset, metric, artifact, feedback, and model operations.

The stable SDK should not provide:

- LangGraph graphs, checkpointers, or an in-memory conversation store;
- tool registries, execution loops, or deployment administration;
- wrappers around `MemAlignOptimizer`, `GepaPromptOptimizer`,
  `optimize_prompts`, or conversational scorers;
- vendor API keys or direct provider credentials;
- a model-pricing table;
- automatic optimizer-to-production promotion.

Those capabilities move faster than the SDK contract. Templates may pin and
exercise them while still exposing the underlying native MLflow objects.

OpenAI and LangChain autologging are opt-in because they can capture raw
framework arguments. Stable `model.generate()` emits one bounded SDK LLM span.
Direct native sync/async/stream calls use MLflow OpenAI autologging with no SDK
provider span. LangGraph uses MLflow LangChain autologging without manual SDK
decorators. Agent Server owns its root trace and uses one selected child path.

## Required lifecycle controls

### Judge alignment and prompt optimization

1. Define a domain rubric and version the judge.
2. Collect reviewer assessments with explicit human provenance and rationale.
3. Separate judge calibration from judge validation.
4. Optimize only against a bounded training set and explicit request/cost
   budget.
5. Evaluate the resulting exact prompt version on held-out cases.
6. Run the existing release gate, then move a controlled alias.

Do not copy the optimization cookbook literally into an automated release
path. Its example moves the production alias before held-out evaluation, uses
provider API-key setup, and includes implementation details that are not
stable platform contracts. Prompt optimizers also need to consume the changed
prompt inside `predict_fn`; otherwise the comparison does not measure that
change.

### Agent trajectories

Evaluation records use MLflow's standard shape:

```json
{
  "inputs": {"question": "Compare orders A-1001 and A-1002."},
  "expectations": {
    "expected_tool_calls": [
      {
        "name": "lookup_order_status",
        "arguments": {"order_id": "A-1001"}
      },
      {
        "name": "lookup_order_status",
        "arguments": {"order_id": "A-1002"}
      }
    ]
  }
}
```

Prefer exact matching for deterministic release gates. Fuzzy matching is an
LLM-judge decision and is appropriate only when several trajectories are
semantically interchangeable. See MLflow's
[ToolCallCorrectness requirements](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/tool-call/correctness/).
The template adds a narrow MLflow 3.14 compatibility layer because the native
unordered exact scorer treats an empty expected list as missing and does not
fully preserve duplicate-call multiplicity. The layer compares a multiset of
normalized name/argument signatures, then delegates matching non-empty cases
to the native scorer.

### Conversations

Each turn should have its own trace and share a pseudonymous session id. Accept
complete history on the request or use an application-owned durable state
store. Scope trace queries by application release and environment before
running session-level scorers. Convert categorical results into explicit gate
metrics such as completion rate, guideline pass rate, and unresolved
frustration rate; critical scenarios may also require per-case pass rules.

MLflow documents dedicated session and user trace fields in
[Track users and sessions](https://mlflow.org/docs/latest/genai/tracing/track-users-sessions/).
Do not attach a raw personal identifier merely because the tracing API permits
one.

### Cost and quality

Compare the baseline and each proposed change on the same cases and record:

- exact logical model/deployment and application release;
- quality metrics and per-row rationales;
- latency and input/output/total tokens;
- trace cost when available;
- usage/cost coverage, since not every provider reports it;
- quality per unit cost only as a decision aid, never as the sole gate.

MLflow 3.14 exposes aggregated `trace.info.token_usage` and
`trace.info.cost`; cost availability depends on the tracing integration,
tracking-server extras, and model pricing support. See
[MLflow token and cost tracking](https://mlflow.org/docs/latest/genai/tracing/token-usage-cost/).
Azure/Databricks billing and the governed AI gateway remain authoritative for
chargeback.

## Follow-up backlog

1. Complete the judge-calibration notebook by executing evaluation, retrieving
   human and judge assessments, calculating held-out agreement, and recording
   versioned calibration evidence.
2. Add native critical-row pass rules and locked calibration/held-out split
   manifests to the prompt and evaluation templates.
3. Resolve an alias to an exact prompt version at process startup, then load
   that exact version inside each active trace for native prompt lineage;
   document that promotion affects future rollout, not existing instances.
4. Add side-by-side change comparison and bounded failure/rationale triage
   to the prompt template; the evaluation template now has bounded triage.
5. Add an optional multi-turn evaluation recipe using native conversational
   scorers and explicit numeric gates.
6. Add a changed logical-model cost/quality experiment that consumes
   trace-recorded usage and cost.
7. Evaluate prompt optimization separately for text prompts and chat-message
   prompts before adding an opt-in template dependency such as DSPy/GEPA.
