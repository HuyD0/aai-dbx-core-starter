# Agent evaluation: the comparison-first paved road

This guide is for a software engineer who has shipped plenty of code and has
never evaluated an LLM. It explains *why* the workflow looks the way it does
before it explains which commands to run, because the why is the part that
does not transfer from normal software.

## Why your usual test instinct does not work here

You already know how to test software: assert the output equals the expected
value. That works because the function is deterministic and there is exactly
one right answer.

An agent has neither property. Ask it the same question twice and you get two
different sentences, both arguably correct. There is no single expected string
to assert against, so `assertEqual` has nothing to hold on to.

The failure mode this creates is subtle and very common. You change a prompt,
run the agent, read three answers, and they look good — so you ship it. What
you actually learned is that three answers looked good *to you*, *today*,
*on the examples you happened to try*. You did not learn whether the change
made the system better or worse than what was already in production. Reading
outputs feels like testing and is not.

## The unit of evidence is a comparison

> **An experiment is a comparison, not a log.**

The question worth answering is never "is this good?" — it is **"is this
better than what we had?"** That question has a real answer, and getting it
requires exactly three things held still:

1. **The same dataset.** A fixed set of questions with expectations, versioned
   so you can prove both runs saw identical inputs.
2. **The same scorers.** Automated graders that turn an answer into a number
   the same way every time.
3. **A recorded baseline.** The previous version's scores on that dataset with
   those scorers.

Change one thing, score it against the baseline, and the delta is your
evidence. That is why the primary verb is `compare` and why running an
evaluation with nothing to compare against is treated as an incomplete
submission rather than a result.

This is also why the tool refuses to be helpful in one specific way: if no
baseline exists it will not quietly score your agent and print a number. A
number with nothing beside it is the thing that misleads people. It tells you
so, and offers to record the current version as the baseline.

## What a scorer actually is

Two kinds, and the difference matters when you read a result:

- **Code scorers** are ordinary Python functions — does the answer contain the
  expected terms, did it refuse when it should have, is it a sane length.
  Deterministic, instant, free, no credentials. They catch blunt failures.
- **LLM judges** are a language model grading another model's answer against
  written criteria. They catch things code cannot: is this actually correct,
  is it grounded in the retrieved documents, is it safe. They cost money, take
  time, and are themselves fallible — which is why a judge must be calibrated
  against human labels before it gates a release, and why judges you have not
  calibrated run in report-only mode.

## Why scorers come from a shared registry

This is the part that justifies the whole toolkit.

If every project writes its own `correctness` scorer, then two teams both
reporting 0.8 means nothing: different judge model, different prompt,
different scale, different definition of correct. You cannot compare across
teams, you cannot set a platform-wide bar, and you cannot audit a promotion
because "0.8" is not a fact about the world.

So scorer name, judge model binding, judge prompt version, input contract and
scale are **versioned platform assets**, shipped in `aai-core` and pinned by
every project through its `aai_core_version`. Judge instructions live in the
Unity Catalog Prompt Registry, so a change to what a judge asks is a governed,
versioned event.

A project **selects** scorers and **sets thresholds**. It never redefines what
one means. Browse the registry:

```bash
agentkit scorers ls
```

Which scorers run is inferred from what the dataset rows contain, so nobody
has to memorise the contracts:

| Row contains | Scorers added |
|---|---|
| `expectations.expected_response` | correctness, keyword coverage, refusal compliance |
| `expectations.guidelines` | per-row guideline adherence |
| retrieval spans in the trace | groundedness, retrieval relevance, sufficiency |
| tool-call spans in the trace | tool-call correctness and efficiency |
| always | response length; safety on judged runs |

The inferred plan is printed before every run, along with what the judge calls
will cost, and anything that *cannot* run is listed with the reason. A
retrieval scorer needs retriever spans in a trace; recorded answers do not
have those, so it is excluded and says so rather than silently scoring zero.

## The commands

```
agentkit init       scaffold a working project from the governed template
agentkit compare    THE primary verb - score this version against the last
agentkit smoke      fast gate: a sample, seconds, no cluster, no judges
agentkit eval       the full suite, locally or as a Databricks job
agentkit gate       promotion check against thresholds and the baseline
agentkit evidence   the release record a reviewer can read
agentkit scorers ls browse the shared registry
```

The whole configuration is three lines:

```yaml
version: 1
agent: src/app/example_agent.py:respond
dataset: evals/data/golden_cases.json
```

Everything else — the experiment name, the run tags, which scorers apply, what
this is compared against, where evidence lands — is inferred or generated. The
optional keys (thresholds, regression budgets, scorer selection, judge budget,
HTTP field mapping) are escape hatches, not the normal path.

`agent:` resolves by shape, so the same project can evaluate a local Python
function today and a deployed endpoint tomorrow with a one-line change:

| Value | Resolves to |
|---|---|
| `src/app/agent.py:respond` or `pkg.module:respond` | a local Python callable |
| `endpoints:/my-agent` or a bare endpoint name | a Databricks serving endpoint |
| `models:/catalog.schema.model` | a Unity Catalog registered model |
| `https://host/score` | any HTTP/JSON endpoint, including a hosted agent elsewhere |
| a logical name from `aai-platform.yml` | whatever that name is configured to be |

Execution can sit anywhere. The record stays in one place, which is the point.

## Two speeds, deliberately

| | Where it runs | When | Why |
|---|---|---|---|
| `agentkit smoke` | your laptop, pull-request CI | every commit | Seconds. No cluster, no credentials, no judge spend — spinning up compute would add latency to exactly the loop that needs to be fast. |
| `agentkit eval` | a Databricks job | pre-merge, pre-promotion | The datasets and production traces already live in Unity Catalog. Compute goes to the data, and results land in the record with no upload step. |

`agentkit eval --submit` runs the bundle's `release_gate` job.

**Smoke does not create an MLflow run.** A code-scorer-only pass over
recorded answers needs nothing from MLflow, so it does not open one. That is
deliberate on two counts: it keeps smoke runnable on every commit with no
credentials and no tracking backend, and it keeps an afternoon of throwaway
runs out of your experiment. `compare` and `eval` are what record a
comparison — that is the ontology, and you do not have to decide it.

A note on the machinery: LLM evaluation is **I/O-bound**. You are waiting on
judge calls, not computing anything. The toolkit uses concurrent requests and
sets MLflow's judge concurrency from your config; the real ceiling is the judge
endpoint's rate limit. Spark is the wrong tool for the scoring loop itself,
though it remains the right tool for the work around it — scanning production
traces to build a dataset, aggregating across many runs.

## Cost is visible before the run, never after

Every judged run prints the estimate first — how many judge calls, roughly how
many tokens, and the dollar figure if you have configured your negotiated
rate — and asks before spending. `budget.max_judge_calls` aborts before the
first call rather than after the last one. `agentkit smoke` is free by
construction: it runs only code scorers.

## What the gate refuses

`agentkit gate` is the promotion check, and it says no in three situations:

1. **No evidence at all.** Nothing has been scored yet.
2. **Evidence that is not a comparison.** A run that never named a baseline
   does not answer "what did you compare against", so it does not pass.
3. **A thresholded metric that never appeared.** If you gate on correctness
   and the judge failed, the run did not produce the evidence — that fails
   closed rather than passing by omission.

Exit codes are a stable CI contract:

| Code | Meaning |
|---|---|
| `0` | every threshold passed |
| `2` | ran successfully, one or more thresholds failed — CI should treat this as a hard failure |
| `1` | runtime or configuration error |

These are CI-agnostic. The repository ships GitHub Actions wiring; any system
that can read an exit code works the same way.

## What lands in the record

Every run writes a governed MLflow run carrying the platform resource tags
plus the lineage the developer would otherwise have to type: dataset reference
and version digest, row count, agent target, scorer versions, judge model,
resolved judge prompt versions, the baseline it was compared against, the gate
verdict, and the decision.

Decisions use the platform vocabulary — **adopt**, **reject**, or
**inconclusive** — and default to `inconclusive`, because a comparison that
nobody has interpreted has not concluded anything.

`agentkit evidence` renders that into `evidence.md` and `evidence.json`:
what ran, on which data version, scored how, against what, with which verdict
and whose approval. Attach it to the promotion request.

## Promotion and the approval gate

For projects promoting into a Unity Catalog registered model, the template
ships an optional deployment-job gate: registering a new model version
triggers a job that evaluates it, waits for a human approval, then hands off
to deployment.

Two things about it are worth knowing before you enable it:

- **The first run always fails at the approval task.** That is by design.
  Approval is recorded as a Unity Catalog tag on the model version, and no tag
  exists yet. Approving in the UI writes the tag and the run resumes.
- **The bundle schema cannot express the link** between a registered model and
  its deployment job, so `scripts/link_deployment_job.py` makes it once through
  the MLflow client after deployment.

The approver needs `APPLY TAG` on the model and `CAN MANAGE RUN` on the job.
Use governed tag policies when several groups must sign off, so nobody can
approve their own change.

## Getting started

```bash
source scripts/platform-env.sh      # exports the platform identifiers
agentkit init --name my-agent-eval
cd my-agent-eval
python3.12 scripts/setup_dev.py
make install-ci

agentkit smoke                       # works immediately, no credentials
agentkit compare --establish-baseline
# ...change something...
agentkit compare
agentkit gate
agentkit evidence
```

The generated project contains a real, runnable agent and a real dataset, and
its gate passes on the first run. Edit `src/app/example_agent.py`, run
`agentkit compare` again, and watch the numbers move — that loop is the thing
worth learning.

## Escape hatches

This toolkit wraps ceremony, not capability. Underneath it is ordinary
MLflow 3 GenAI evaluation, and nothing hides it: `mlflow.genai.evaluate`,
the native scorers, the runs and traces are all reachable directly, and
`ExperimentManager.native_client` and `PromptManager.native_client` hand you
the native module when you need something the toolkit does not wrap. When you
outgrow a piece of it, drop to the native API for that piece and keep the rest.

## Related documents

- `docs/genai-lifecycle.md` — the full evidence chain and lifecycle vocabulary
- `docs/tagging-standard.md` — the governed tag fields every run carries
- `docs/developer-guide.md` — the end-to-end developer path
- `docs/cost-estimation.md` — how platform cost attribution works

## Notes for the other templates

The five other templates still ship the pre-agentkit `evals/` pattern: a
hand-written `offline_checks.py` and `evaluate.py` per project, with
thresholds in `gate_config.json` and judges defined in `src/app/judges.py`.
They keep working. To adopt the toolkit, add an `agentkit.yaml`, replace those
two scripts with the shims the `evaluation-project` template uses, and delete
the local judge definitions in favour of the shared registry.

Azure DevOps pipeline templates are not shipped yet. The exit-code contract is
CI-agnostic, so wiring `agentkit smoke` to pull requests and `agentkit eval`
pre-merge works the same way on any runner that can authenticate without a
stored secret.
