# Progressive AI application lifecycle

These examples are one curriculum, not unrelated API snippets. They carry the
same fictional Aster Ridge Systems earnings-summary assistant through:

```text
offline contract
  → governed trace
  → baseline/change experiment
  → exact prompt lineage
  → deterministic evaluation gate
  → connected setup
  → connected stable first call
  → native async streaming observation
  → exact tool-trajectory evaluation
  → multi-turn session evaluation
  → layered and calibrated judges
  → logical-model cost/quality analysis
  → optional aligned-judge prompt optimization
  → recorded decisions and evidence-gated promotion
  → platform-team LLM operations
```

The first four MLflow examples run locally and deterministically without a
model, cloud access, or credentials. The connected setup, small stable-adapter
call, and native async/streaming notebook come next. Advanced labs 08–11 and
13 are credential-free decision fixtures; lab 12 is a disabled-by-default
connected optimization skeleton whose experimental dependencies are not in
the certified locks; lab 14 is the platform operator's connected loop with a
credential-free default path.

Every latency, token, and cost value in the credential-free stages is labelled
`simulated_offline_fixture`; those values teach evidence shape and comparison
discipline, not provider performance.

All issuer names, earnings excerpts, figures, and source identifiers in this
curriculum are synthetic. They are teaching data, not market data, and the
assistant must summarize the supplied excerpt without making an investment
recommendation.

## Separate classical ML course

[`local-classification/`](local-classification/README.md) is a standalone,
zero-download sklearn and MLflow course. It is not another stage in the numbered
GenAI curriculum below and has its own exact environment, local SQLite tracking
store, tests, and ten-notebook path.

Use it when you want to learn a conventional binary-classification lifecycle:
problem and data contracts, time splits, leakage-safe Pipelines, imbalance-aware
metrics, validation-only model/threshold selection, a frozen release test,
registry aliases, inference, drift, and the move to Databricks. Start with:

```bash
make classification-install
make classification-check
make classification-notebook
```

## Separate governed batch inference pattern

[`governed-batch-inference/`](governed-batch-inference/README.md) is a
standalone reference implementation for running `ai_query` over large tables
without shipping unvalidated model output into finance data products. It is
not a stage in the numbered curriculum: one Databricks notebook plus a pure,
unit-tested Python module walk `declare → estimate → sample → evaluate →
gate → execute → land → monitor` end to end on synthetic tax documents, with
Wilson-lower-bound gates, worst-stratum rules for high-criticality fields, an
abstention path, and three-layer provenance. Its statistics are pinned by
`tests/test_governed_batch_inference.py`.

## Separate offline Apple-silicon fine-tuning study

[`local-finetuning/`](local-finetuning/README.md) is an Apple-silicon,
offline-first Bitext and MLX-LM study project. It prepares all third-party
assets before travel, proves local execution with sockets blocked, compares
deterministic and prompting baselines with a LoRA change, logs local MLflow
evidence, and includes a deterministic application-readiness capstone. Start
from the repository root:

```bash
make study-prepare-flight
make study-offline-check
```

## Why this curriculum exists

Calling an LLM is easy; knowing whether a changed AI application is safer or
better is the hard part. A plausible response from one call does not establish
that the response is reproducible, grounded in the supplied earnings excerpt,
properly cited, affordable, or consistently free of investment advice.

The examples therefore build an evidence chain around one controlled change:

- **Prompt v1** summarizes only facts from an earnings excerpt.
- **Prompt v2** adds one requirement: include the exact source identifier once.
- Both versions receive the same three fictional cases and use the same model.
- The result compares fact coverage, citation behavior, policy compliance,
  latency, tokens, cost, and cost coverage.

Holding everything except the prompt requirement constant matters because it
makes the result explainable. If the model, cases, and instructions all changed
at once, the team could not tell what caused a different answer.

## MLflow concepts in plain language

| Concept | Plain-language meaning | Why it matters |
|---|---|---|
| Prompt template | Reusable instructions with variables such as an earnings excerpt and source ID. | Keeps application instructions separate from one test case. |
| Prompt Registry or catalog | Version history for prompt templates. | Preserves exactly what changed and prevents a mutable name from masquerading as release evidence. |
| Trace | The record of what happened during one request. | Makes a slow, costly, or incorrect response diagnosable. |
| Span | One operation inside a trace, such as prompt rendering or the model call. | Shows where time, tokens, errors, and behavior came from. |
| Run | The parameters, inputs, metrics, and artifacts from one test. | Turns an observation into evidence that can be reproduced and compared. |
| Experiment | A stable collection of related runs. | Keeps baseline and change evidence together for one decision. |

These objects are not interchangeable. A trace can explain one request but
cannot establish release quality. A run can record an evaluation but does not
preserve prompt content unless it links an exact registered version. An
experiment organizes comparisons but does not itself approve a release.

## The lifecycle record

Every executable script in the core lifecycle emits a final
`LIFECYCLE_RESULT`. The advanced notebooks preserve the same vocabulary:

| Field | Meaning |
|---|---|
| `hypothesis` | The falsifiable claim being tested. |
| `baseline` | The current behavior and its immutable evidence. |
| `change` | One deliberate difference from the baseline. |
| `result` | Quality, latency, token, cost, coverage, or lineage evidence. |
| `decision` | What the evidence permits next. |
| `release` | The exact eligible release, or why release remains blocked. |

Earlier stages correctly keep release blocked. Creating a trace, registering a
prompt, or observing one model response is not a release decision.

## Run the curriculum locally

```bash
make quickstart
make local-start
make local-example EXAMPLE=first_experiment
make local-example EXAMPLE=first_prompt
make local-example EXAMPLE=first_evaluation
make local-ui
```

`quickstart` creates or synchronizes the locked development environment and
runs `00_offline_hello_world.py`. The MLflow commands use the isolated, ignored
`.aai/local/mlflow.db` tracking and prompt-registry store. `local-ui` serves
only that store at `http://127.0.0.1:5000`.

| Order | Example | What it teaches locally |
|---:|---|---|
| 00 | [`00_offline_hello_world.py`](00_offline_hello_world.py) | Provider-neutral SDK contracts, secret redaction, and explicitly unknown cost. |
| 01 | [`01_first_trace.py`](01_first_trace.py) | Why one earnings-summary request needs a bounded trace, nested model span, token usage, and an explicit no-autolog choice. |
| 02 | [`02_first_experiment.py`](02_first_experiment.py) | Why baseline/change runs must use the same ordered earnings cases before quality, latency, tokens, and cost can be compared. |
| 03 | [`03_first_prompt.py`](03_first_prompt.py) | Why `earnings_summary` prompt versions are registered idempotently, loaded by exact URI, digested, and linked to runs. |
| 04 | [`04_first_evaluation.py`](04_first_evaluation.py) | Why deterministic scorers and row-critical gates—not one good-looking answer—are required before adopting `earnings-summary-prompt-v2`. |
| 05 | [`05_connected_setup.ipynb`](05_connected_setup.ipynb) | Why kernel/config readiness and cloud authorization are separate checkpoints. It makes no LLM request. |
| 06 | [`06_connected_first_call.py`](06_connected_first_call.py) | How to call a real configured LLM through stable synchronous `model.generate()` while recording bounded trace and run evidence. |
| 07 | [`07_first_llm_call.ipynb`](07_first_llm_call.ipynb) | Native async streaming, readable chat traces, exact prompt lineage, and an optional UC EvaluationDataset linked to described A/B runs. |
| 08 | [`08_tool_trajectory_evaluation.ipynb`](08_tool_trajectory_evaluation.ipynb) | Why a correct answer can fail an exact tool trajectory, with optional governed dataset/run evidence but no fabricated trace. |
| 09 | [`09_multi_turn_session_evaluation.ipynb`](09_multi_turn_session_evaluation.ipynb) | How to scope real traces, retain their IDs, register the session contract, and gate complete conversations. |
| 10 | [`10_layered_judges.ipynb`](10_layered_judges.ipynb) | How deterministic checks and human calibration become separate UC datasets linked to a report-only judge run. |
| 11 | [`11_cost_quality_tradeoff.ipynb`](11_cost_quality_tradeoff.ipynb) | Why quality comes before cost, with actual synthetic cases registered separately from simulated measurement artifacts. |
| 12 | [`12_agent_alignment_optimization.ipynb`](12_agent_alignment_optimization.ipynb) | How disjoint UC datasets, immutable prompt versions, readable real-call traces, and held-out runs prevent optimizer-to-production shortcuts. |
| 13 | [`13_decision_and_promotion_lifecycle.ipynb`](13_decision_and_promotion_lifecycle.ipynb) | Why every comparison ends in a recorded `adopt`/`reject`/`inconclusive` decision, and why the `production` prompt alias moves only on adopt-grade evidence. |
| 14 | [`14_platform_llm_operations.ipynb`](14_platform_llm_operations.ipynb) | The platform team's operating loop: judge governance, gateway request tags, cost-by-tag queries, fleet provenance, monitoring adoption, and rollback levers. |

Open any advanced lab through the stable runner name, for example:

```bash
make local-example EXAMPLE=tool_trajectory_evaluation
make local-example EXAMPLE=multi_turn_session_evaluation
make local-example EXAMPLE=layered_judges
make local-example EXAMPLE=cost_quality_tradeoff
make local-example EXAMPLE=agent_alignment_optimization
make local-example EXAMPLE=decision_promotion_lifecycle
```

The command prints the exact numbered path and selected kernel. The default
path for all six labs makes no model request and writes no remote evidence.
Each lab exposes an explicit Databricks switch for its governed evidence path.
The platform-operations lab runs through the workspace runner
(`make workspace-example EXAMPLE=platform_llm_operations`) because its
connected checks address the operator, not the application developer; its
deterministic cells still run anywhere.

### Cookbook adaptations

| MLflow cookbook | Curriculum coverage | Platform strengthening |
|---|---|---|
| [Evaluation-driven development](https://mlflow.org/cookbook/eval-driven-development/) | 02, 04, 07 | Fixed ordered data, exact digests, scorer-error failure, critical-row gates, and explicit release decisions. |
| [Prompt engineering lifecycle](https://mlflow.org/cookbook/prompt-engineering/) | 03, 04, 07, 12 | Idempotent immutable versions, exact-version lineage, one controlled change, and alias movement only after the normal gate. |
| [Cost-quality trade-off](https://mlflow.org/cookbook/cost-quality-tradeoff/) | 07, 11 | Logical model names, no embedded vendor prices, separate target/judge cost, and explicit cost coverage. |
| [LangGraph agent](https://mlflow.org/cookbook/langgraph-agent/) | 08 plus the optional agent-template recipe | Exact tool-call scoring for gates, one tracing owner, durable checkpoints, interrupts, and idempotency. |
| [Multi-turn agent](https://mlflow.org/cookbook/multi-turn-agent/) | 09 | One trace per turn, opaque session IDs, exact trace scoping, numeric gates, and durable application-owned state. |
| [Custom LLM judges](https://mlflow.org/cookbook/custom-llm-judges/) | 10 | Deterministic rules first, keyless governed judge models, balanced human rationales, and held-out agreement. |
| [Agent alignment and optimization](https://mlflow.org/cookbook/agent-alignment-optimization/) | 10, 12 | Three disjoint evidence splits, bounded calls, prompt loading inside `predict_fn`, and no optimizer-to-production shortcut. |

The scripts take no command-line arguments. For isolated direct execution,
they honor:

- `MLFLOW_TRACKING_URI`
- `MLFLOW_REGISTRY_URI`
- `AAI_PLATFORM_CONFIG`
- `AAI_EXAMPLE_LOCAL_DIR`
- `AAI_EXAMPLE_ARTIFACT_ROOT`

When none are set, the MLflow scripts use `aai-platform.example.yml` and the
ignored `.aai/local` store.

## Naming is part of reproducibility

The curriculum creates a stable experiment named for the decision it supports:

```text
/Shared/example-ai-earnings-summary-quality-cost
```

Run names describe their role and change:

```text
baseline-earnings-summary-prompt-v1
change-cited-earnings-summary-prompt-v2
```

Do not use names such as `first-comparison`, `test-2`, or a bare timestamp.
Searchable run metadata additionally records the purpose, change ID, change
summary, hypothesis, and baseline run ID.

The shared identifiers keep every stage connected:

| Record | Stable name |
|---|---|
| Prompt | `earnings_summary` |
| Trace prefix | `earnings_summary` |
| Dataset | `fictional-earnings-summary-regression-v1` |
| Eligible release after the full gate | `earnings-summary-prompt-v2` |

The dataset's three stable cases cover quarterly revenue and operating margin,
forward revenue and margin guidance, and free cash flow, inventory growth, and
supplier risk. Keeping that ordered dataset fixed prevents accidental case
selection from making one prompt look better than another.

## What the SDK owns and what remains native

The examples use `aai-core` for governed, reusable contracts:

- platform configuration and `ResourceContext`;
- stable experiment context and searchable run metadata;
- trace configuration and bounded span helpers;
- governed prompt names, tags, registration, and loading.

They use native MLflow APIs for capabilities that should remain visible:

- `mlflow.data.from_pandas()` and `mlflow.log_input()`;
- metrics, tags, and artifacts;
- exact prompt-version-to-run links;
- `@mlflow.genai.scorers.scorer`;
- `mlflow.genai.evaluate()`.

This boundary teaches why governance evidence is needed without hiding the
native MLflow operations a Python developer will use outside the SDK.

## Autologging and trace data policy

The deterministic earnings-summary application uses SDK-managed spans, so
OpenAI and LangChain autologging are disabled. Enabling an autologger for the
same call would duplicate spans and token counts.

`03_first_prompt.py` does not call a model or framework, so there is nothing
for a framework autologger to instrument. It records prompt registration,
exact loading, and safe synthetic rendering through a governed run, a manual
`PROMPT` span, native version links, and prompt digests. That is intentional
tracking, not missing autologging.

`06_connected_first_call.py` is the paved-road first call. It uses
`model.generate()` once, so `TraceIntegration.SDK` owns one bounded provider
span. It records latency and normalized usage, keeps unavailable cost
explicitly unknown, and blocks release pending evaluation.

`07_first_llm_call.ipynb` is the advanced positive autologging example. It opts
into approved capture for synthetic earnings cases, invokes a
worker/event-loop-owned client from `model.create_native_async_client()`,
consumes native provider streams, and lets MLflow own the provider span. One
manual application parent attaches governed context; the same invocation never
passes through `model.generate()`, whose SDK provider span would duplicate
usage evidence. The parent uses MLflow's OpenAI message format, stores only the
assistant content as its output, and sets plain-text request/response previews;
latency, model, usage, and cost remain telemetry instead of rendering as the
answer. The notebook compares both exact prompt versions across the
same three cases, reads tokens and cost back from each completed trace, records
cost coverage when pricing is unavailable, and links each local trace to the
exact local prompt version and comparison run. It intentionally concludes
`inconclusive / run full evaluation`: six exploratory calls are useful for
learning and debugging, not sufficient release evidence.

Use [`05_connected_setup.ipynb`](05_connected_setup.ipynb) after the local
gate when you want to diagnose the kernel, configuration, Azure identity,
workspace membership, and endpoint separately. Both connected notebooks call
`notebook_setup.py`; the tutorial does not use `%run` or depend on state left
by the setup notebook.

The default notebook path intentionally splits execution from evidence:
Databricks serves the LLM, while local SQLite stores experiment, run, trace,
and prompt metadata and `.aai/local/mlruns` stores artifacts. Setting the
top-level `SEND_EVIDENCE_TO_DATABRICKS` switch to `True` instead routes
tracking to `databricks` and prompt registration to `databricks-uc`, so the
experiment, runs, traces, exact prompts, metrics, and lineage links share the
compatible Databricks backend. In that mode the three cases are also merged
idempotently into a fully qualified Unity Catalog EvaluationDataset and linked
as a native input to both described runs. The later prompt-only publishing guard remains
available for copying prompts after a deliberately local comparison, but those
local traces are never linked across stores.

Labs 08–12 follow the same evidence rule: a disabled-by-default connected path
gets or creates fully qualified Unity Catalog EvaluationDatasets, merges only
synthetic records, and links each dataset to a natively described run with
`mlflow.log_input`. Databricks-managed datasets do not accept MLflow dataset
tags, so governed context stays on runs and platform-managed UC securables.
Offline fixtures never manufacture traces or prompt lineage. Prompt links are
added only by labs 07 and 12 when a registered prompt actually produced a real
model trace.

Before enabling any provider or framework autologger, document:

1. which inputs and outputs it captures;
2. why the data classification permits that capture;
3. which manual/provider instrumentation is disabled to prevent duplicates;
4. how root and child spans receive the same resource and conversation context.

## Completion rubric

An example or generated project meets this teaching standard only when:

- the experiment name identifies a stable application decision;
- run names and metadata distinguish baseline, change, result, and decision;
- one falsifiable hypothesis and one change are recorded;
- baseline and change use the exact same ordered dataset digest;
- connected evaluation data is a native UC EvaluationDataset linked to its run;
- connected runs carry a native description that distinguishes observed and
  simulated evidence;
- traces declare capture policy and autolog mode without duplicate spans;
- model traces render assistant content as text while retaining telemetry;
- prompt evidence uses an exact URI/version and content digest, never only an
  alias;
- quality, latency, input/output tokens, cost, and cost coverage are recorded;
- unknown cost remains unknown rather than becoming zero;
- critical rows can fail individually instead of disappearing in an average;
- scorer errors fail gated metrics;
- source revision, environment digest, and deterministic seed or repetition
  policy are recorded;
- the final decision names the eligible release or explicitly blocks it.

The connected notebook is exploratory and therefore always ends with an
inconclusive decision and `release=blocked_until_evaluated`, regardless of how
good its small live sample looks. The deterministic `04_first_evaluation.py`
stage is the release-grade gate and adopts `earnings-summary-prompt-v2` only
when every required threshold passes. Only after that gate succeeds does it
move the SDK-governed `production` alias to the exact adopted version. The
alias is a deployment pointer; the evaluation, run, and trace evidence continue
to use the immutable version URI.

## Move the same evidence to Databricks

After the local lifecycle is understood:

```bash
make workspace-connect
# Complete only the reported keyless authentication/configuration actions.
make workspace-example EXAMPLE=first_trace
make workspace-example EXAMPLE=first_experiment
make workspace-example EXAMPLE=first_prompt
make workspace-example EXAMPLE=first_evaluation
make workspace-example EXAMPLE=connected_first_call
```

The workspace runner supplies the non-secret Databricks host and MLflow
routing. It never asks for a PAT, client secret, or API key.

If running a connected file directly:

```bash
make examples-install
az login
export DATABRICKS_HOST=<workspace host from platform-identifiers.json>
export DATABRICKS_AUTH_TYPE=azure-cli
export MLFLOW_TRACKING_URI=databricks
export MLFLOW_REGISTRY_URI=databricks-uc
export AAI_PLATFORM_CONFIG="$PWD/aai-platform.yml"
.venv/bin/python examples/01_first_trace.py
```

Connected prompt examples require the externally provisioned Unity Catalog
prompt registry and least-privilege access. The evaluation example remains
LLM-free; adding an approved judge is a later, calibrated change.

Use `make examples-list` to see the runner's accepted names and modes.

## Where each example leads

| Example | Graduates into |
|---|---|
| `02_first_experiment.py` | `templates/experiment-starter` |
| `03_first_prompt.py` | `templates/prompt-app` |
| `04_first_evaluation.py` | `templates/evaluation-project` |
| `06_connected_first_call.py`, `07_first_llm_call.ipynb`, `01_first_trace.py` | `templates/rag-app` / `templates/agent-app` |
| `08_tool_trajectory_evaluation.ipynb`, `09_multi_turn_session_evaluation.ipynb` | `templates/agent-app` and its optional LangGraph recipe |
| `10_layered_judges.ipynb` | `templates/evaluation-project` |
| `11_cost_quality_tradeoff.ipynb`, `12_agent_alignment_optimization.ipynb` | a connected prompt or agent project after dependency and judge approval |
| `13_decision_and_promotion_lifecycle.ipynb` | `templates/prompt-app` promotion scripts and every template's release gate |
| `14_platform_llm_operations.ipynb` | the platform team's operating runbook ([platform operations](../docs/platform-operations.md)) |
| `00_offline_hello_world.py` | every template's hermetic test pattern |

## Notebook conventions

- Jupyter (`.ipynb`) is for local exploration and explicitly guarded connected
  labs, like `07_first_llm_call.ipynb` through
  `14_platform_llm_operations.ipynb`.
- Generated projects use packaged Python under `src/`; Databricks-format
  notebooks remain thin teaching or operational entry points.
- Configuration is never hardcoded. `bootstrap()` discovers
  `aai-platform.yml` by walking upward, or uses `AAI_PLATFORM_CONFIG`.
