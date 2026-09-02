# MLflow cookbook relevance assessment

Reviewed on 2026-08-10 against the repository's certified MLflow 3.15.1 API and
current MLflow 3.15.1 lock.

The seven cookbooks describe a useful evaluation-driven lifecycle, but they
are examples rather than platform policy. Adopt the durable trace, evaluation,
and promotion shapes below; keep framework-specific and experimental MLflow
APIs native and opt-in.

## Coverage and decisions

| Cookbook | Repository coverage | Decision |
|---|---|---|
| [Agent optimization pipeline](https://mlflow.org/cookbook/agent-alignment-optimization/) | Prompt versions, native evaluation data and feedback, deterministic gates, and controlled aliases exist. `examples/12_agent_alignment_optimization.ipynb` adds a disabled-by-default three-way split, dependency check, request budget, prompt-consuming prediction function, and held-out release handoff. | Keep optimizer and assessment objects native. Experimental DSPy/GEPA dependencies remain outside certified locks. Never let an optimizer promote `production`; the normal release gate remains authoritative. |
| [LangGraph agent](https://mlflow.org/cookbook/langgraph-agent/) | The application-owned agent loop emits standard `TOOL` spans, the agent template includes a durable native LangGraph recipe, and `examples/08_tool_trajectory_evaluation.ipynb` demonstrates exact multiset scoring including a correct-answer/wrong-tool failure. | Keep MLflow LangChain autologging opt-in and LangGraph out of core. Exact trajectory checks gate releases; fuzzy judge matching remains report-only until calibrated. |
| [Multi-turn agent](https://mlflow.org/cookbook/multi-turn-agent/) | `AgentRequest` preserves governed history and binds an opaque session ID to each trace. `examples/09_multi_turn_session_evaluation.ipynb` adds scoped session fixtures, explicit numeric metrics, critical-session checks, and a guarded native conversational-judge handoff. | Retain the core session seam. State belongs in the application or a durable framework store, never a process-global core dictionary. Add the connected recipe to the generated agent template when its experimental judges are approved. |
| [Custom LLM judges](https://mlflow.org/cookbook/custom-llm-judges/) | The evaluation template combines deterministic and routed judges. `examples/10_layered_judges.ipynb` executes exact rule checks, balanced calibration/validation agreement, sample-size authority, and a guarded keyless `make_judge` registration. | Keep a custom judge report-only until held-out human calibration supports both agreement and sample-size thresholds. Keep `Guidelines`, `make_judge`, `Feedback`, and registered scorers as native MLflow objects. |
| [Evaluation-driven development](https://mlflow.org/cookbook/eval-driven-development/) | Strong coverage: reviewed cases, offline and credentialed tiers, per-release baselines, absolute/regression gates, tracked runs, production-failure curation, critical-case rules in the learning path, and bounded per-row failure triage in the evaluation template. Scorer errors on gated rows fail rather than disappearing from aggregates. | Retain the current design and extend critical-case rules consistently across templates; aggregate improvements alone are not enough evidence. |
| [Prompt engineering lifecycle](https://mlflow.org/cookbook/prompt-engineering/) | Strong coverage: immutable prompt versions, exact-version evaluation, controlled aliases, and gated promotion. | Retain the stricter repository behavior. A prompt change is an application release; do not rely on a long-running service hot-reloading a moved alias without an evaluated rollout. Add few-shot and side-by-side comparison guidance. |
| [Cost-quality trade-off](https://mlflow.org/cookbook/cost-quality-tradeoff/) | The progressive example compares baseline/change quality, latency, tokens, cost, and coverage. `examples/11_cost_quality_tradeoff.ipynb` fixes the prompt/data/scorer contract, separates target and judge cost, filters on quality first, and excludes incomplete cost from ranking. | Keep vendor prices out of core. The next connected step compares configured logical models using trace-recorded cost and platform gateway/billing evidence. |

## Blog post review (2026-09-02)

Two later MLflow posts were reviewed against the same certified line. They
describe the loop this repository already teaches — trace first, build a
dataset, layer deterministic and judged scorers, inspect traces, iterate,
then monitor and optimize — so the decisions below are narrow.

| Post | Repository coverage | Decision |
|---|---|---|
| [Evaluating and improving agent skills](https://mlflow.org/blog/evaluating-improving-agent-skills/) | The three scorer classes it names — output-based, rule-based, trace-based — are the curriculum's Outcome/Behavior/Operations layers. Its headline example, a precondition tool that must run *before* the tool it guards, had no equivalent: the trajectory scorers are unordered by design. The shared registry now carries `tool_order_policy`, reading `expectations.expected_tool_order` against TOOL spans in start order, and `examples/08_tool_trajectory_evaluation.ipynb` adds the case the multiset cannot see. The Deep Agents accelerator, which already synthesizes and promotes `SKILL.md`, now records the skill digest on every evaluation run and compares a promoted skill with the last run under the previous digest. | Adopt ordering as a deterministic registry scorer, not a judge (`docs/decisions/2026-09-02-tool-order-policy-is-a-code-scorer.md`). A skill change is an application release: prove it on the same signals, never assume it helped. |
| [Structured AI evaluation](https://mlflow.org/blog/structured-ai-eval/) | Its three phases map onto the lessons directly: prototype with tracing (01, 07), evaluate with judges and datasets (04, 08–10), monitor and optimize (12–14). `Guidelines`, `make_judge`, datasets, prompt versions, and human feedback are all native objects here already, with the judge-calibration gate the post omits. The console's Evaluate track, previously a checklist, now carries the `agentkit` command loop the post describes. | Keep autologging opt-in (it captures raw framework arguments) and prompt optimization outside the certified locks. Categorical `make_judge` output (`Literal[...]`) stays a native option; the registry converts categorical verdicts into explicit numeric gate metrics rather than adding a scale. |

One correction: the first post's code calls a scorer named `ToolCallRelevance`.
No such class exists in MLflow 3.15.1 or on MLflow's main branch; the tool-call
scorers are `ToolCallCorrectness` and `ToolCallEfficiency`, both already in
the registry. Do not copy that snippet into a lesson.

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
Durable LangGraph persistence follows the same rule: it ships as the
certified `recipes/langgraph-lakebase/` template recipe against Lakebase
(see [Production LangGraph agents](langgraph-production.md)), never as SDK
checkpoint or store wrappers.

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
The template adds a narrow MLflow 3.15.1 compatibility layer because the native
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

MLflow 3.15.1 exposes aggregated `trace.info.token_usage` and
`trace.info.cost`; cost availability depends on the tracing integration,
tracking-server extras, and model pricing support. See
[MLflow token and cost tracking](https://mlflow.org/docs/latest/genai/tracing/token-usage-cost/).
Azure/Databricks billing and the governed AI gateway remain authoritative for
chargeback.

## Follow-up backlog

1. Complete the generated evaluation-project judge-calibration notebook by
   executing evaluation, retrieving assessments, calculating held-out
   agreement, and recording versioned evidence. The root layered-judge lab now
   demonstrates the agreement and sample-size decision shape offline.
2. Add native critical-row pass rules and locked calibration/held-out split
   manifests to the prompt and evaluation templates.
3. Resolve an alias to an exact prompt version at process startup, then load
   that exact version inside each active trace for native prompt lineage;
   document that promotion affects future rollout, not existing instances.
4. Add side-by-side change comparison and bounded failure/rationale triage
   to the prompt template; the evaluation template now has bounded triage.
5. Promote the root multi-turn lab into an optional connected agent-template
   recipe after approving the experimental conversational judges.
6. Extend the root cost/quality fixture with a connected two-logical-model
   experiment that consumes trace-recorded usage and authoritative cost.
7. Evaluate prompt optimization separately for text and chat-message prompts
   before adding opt-in DSPy/GEPA dependencies through the full policy, lock,
   template-lock, and compatibility workflow.
