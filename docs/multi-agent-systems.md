# Multi-agent systems in production

This guide maps the failure modes reported by frontier multi-agent research —
Anthropic's ["Patterns and problems in emerging multiagent
systems"](https://www.anthropic.com/research/multiagent-systems) is the
reference study — onto this platform's trace conventions, scorer registry,
and release gates. It exists so a team adding a second agent inherits a
measurement discipline on day one instead of discovering the failure modes in
production.

The boundary stays fixed: `aai-core` ships no orchestration runtime. The
coordination framework (`deepagents`, LangGraph, or the application's own
loop) owns supervisors, subagents, handoffs, and scheduling; the SDK owns
what it has always owned — identity, tracing policy, the evaluation
ontology, and release evidence. Everything below is measurement and
governance, not a coordination API.

## When a second agent pays its way

The research is unambiguous about where coordination helps. Swarms of
coordinated agents beat the same number of isolated parallel agents on work
that decomposes into independent sub-problems — in the reference study's
vulnerability-detection experiment, at roughly four times the token budget
for roughly twelve times the findings. On deeply interdependent work, the
same swarms produced consistently poor results no matter how the roles were
prompted, opening hundreds of pull requests and merging few.

The platform framing is cost per accepted outcome, which is a comparison —
exactly what `agentkit` already measures. Before reaching for a supervisor
and subagents, record the single-agent baseline, then let the comparison
carry the decision the way every other change does: `adopt`, `reject`, or
`inconclusive`, with the token cost beside the quality delta. The judge-cost
budget (`budget.max_judge_calls`) bounds evaluation spend; the same
cost-per-outcome question should be asked of the agent tokens themselves.

Two properties of identical-model agents deserve respect before scaling a
fleet: low variance means agents given the same context converge on the
same strategy, so an individually reasonable decision becomes a stampede
(the study's job-market experiment produced millions of requests for a
hundred acceptances); and shared channels invite convergence the way public
price boards invite price-matching. Diversify roles through distinct
prompts, tools, and skills — not by hoping identical agents will disagree.

How to structure the second agent once it is justified — the ten-question
workflow-shape checklist, the recurring shapes with their guardrails, and
the single-agent-first rule — is `docs/langgraph-production.md`; this
document owns why, and how the collective is measured.

## The delegation trace convention

A multi-agent application declares itself in its traces. The convention,
established by the Deep Agents solution accelerator
(`examples/deepagents-solution-accelerator/` — its README documents the
workflow shape), is:

- One root `AGENT` span for the supervisor.
- A child `AGENT` span per delegation (`delegation.<role>`), and an `AGENT`
  span per subagent turn, each carrying the subagent's role in an
  `agent.role` attribute.
- Operational `TOOL` spans nested under the subagent that executed them —
  never directly under the supervisor, which delegates and does not act.

The marker the toolkit reads is precise: a **non-root `AGENT` span carrying
a non-empty `agent.role` attribute**. Role-less `AGENT` spans are what
`record_agent_decision` writes inside single-agent applications, and a role
on only the root labels a single agent; neither switches the multi-agent
scorers on. Single-agent gates therefore never pay for delegation checks
they cannot satisfy, and a supervisor application cannot accidentally opt
out — its delegation spans are the opt-in.

## Measuring coordination

The reference study's own method is the discipline to copy: coordination is
measured, not assumed. The shared scorer registry carries two versioned
entries for it, selected automatically when dataset rows carry delegation
spans (`agentkit scorers ls` shows both):

- **`delegation_structure_ok`** (code, default `>=1.0`) verifies the
  convention deterministically: exactly one root span and it is an `AGENT`
  span, a parent graph that resolves, and every `TOOL` span executing under
  a role-carrying subagent. A row whose trace carries no delegation spans
  is skipped and reported, mirroring how retrieval scorers treat rows that
  retrieved nothing; a structural violation scores zero and fails the gate.
- **`subagent_routing_accuracy`** (prompt judge, default `>=0.7`) reads the
  trace and grades the delegations against the request: the right subagent
  for each question, no unnecessary delegation, no supervisor bypassing a
  needed one. The rubric is graded (1.0 / 0.5 / 0.0), so the metric
  aggregates as a fraction rather than a pass rate.

Scorer name, rubric, scale, and judge binding are platform assets: a
project selects these scorers and sets thresholds, never redefines them —
the rule section 5 of `AGENTS.md` states for every registry entry. Routing
accuracy measured two ways is not comparable, and comparability is the
point.

## Failure modes and the platform's answer

- **Conformity cascades and resource exhaustion.** Low-variance agents make
  the same retry, the same request, the same branch name — simultaneously.
  The single-agent disciplines are collective requirements: bounded loops
  and fan-out caps (the agent template's `controls.py`), idempotency keys
  in front of side effects (`execute_once` in the LangGraph recipe), and
  budgets that stop a run rather than describe it. Watch retry and
  delegation-depth counts against their caps in monitoring, not only in
  release evaluation.
- **Collusion over shared channels.** Agents that share a writable channel
  converge on it, for good and ill. The accelerator's governed skill
  promotion is the platform pattern: a shared instruction file that only a
  reviewed, allow-listed, deterministic pipeline may change, with immutable
  guardrails prepended and concurrent edits detected. An ungoverned shared
  scratchpad that every agent can write is an incident with a delay on it.
- **Epistemic brittleness.** Groups of agents converge on shared knowledge
  and suppress decisive private information, and a single agent extends
  trust to whatever enters its context. Treat tool output and retrieved
  text as data, never as instructions; require claims to trace to recorded
  evidence. The analytics template's opt-in reviewer is the shape: a second
  role that sees only tool-recorded evidence and can strike unsupported
  numbers but never introduce new ones.
- **Incompatible directives.** Agents given conflicting goals without
  mutual awareness read each other as adversaries and escalate. Give every
  agent prescriptive, measurable success criteria — a gate it can pass, not
  a mandate it can defend — and make escalation to a human the designed
  success path for conflict, which is the interrupt loop
  `docs/langgraph-production.md` builds: the reviewer's decision becomes
  trace evidence and a regression case, not a lost argument.
- **Autonomy against corrigibility.** Capability at execution does not
  imply judgment about when to stop. The platform's answer is structural,
  not aspirational: interrupts before irreversible work, strict typed
  approval decisions with reason codes, and deferral recorded as evidence.
  An agent asking for help is the system working.
- **No reputation, no court, no colleague who remembers.** Human
  institutions correct miscalibrated trust; agents arrive without them.
  What stands in here is provenance: one `ResourceContext` projected onto
  every run, trace, and log; delegation recorded as spans; decisions
  carrying reason codes and request ids; and gates that refuse to mint a
  release from evidence they cannot attribute.

## Follow-up backlog

Deliberately not in this change, recorded so it is owned rather than
implied:

- Point the Deep Agents accelerator's continuous-evaluation notebook at the
  registry's `subagent_routing_accuracy` entry instead of its notebook-local
  judge definition, now that the registry ships one. The accelerator's
  README, examples-index entry, and contract tests already exist, and its
  `deepagents`/`databricks-langchain` pins are a documented,
  test-cross-checked accelerator-only channel — deliberately outside
  `aai-core`'s dependency closure.
- Promote the accelerator's signal taxonomy (subagent exceptions, routing
  score floors, HITL rejections, retry pressure) into a platform monitoring
  vocabulary once more than one application consumes it.
- Add a `delegation` member to `AgentDecisionType` when an application
  records delegation decisions through `record_agent_decision`; the scorers
  read spans today, so the enum member without a consumer would be surface
  without evidence.
- A registered-monitoring variant of the structure check (self-contained
  scorer body), so production traces get the same verification the release
  gate applies.
- Exhaustion and collusion monitors over production traces — identical
  failure categories repeating across agents, delegation depth trending
  into caps, duplicate-request pressure.

## Related documents

- `docs/agent-evaluation.md` — the comparison-first evaluation paved road
- `docs/langgraph-production.md` — durable execution, interrupts, and
  review decisions as evidence
- `docs/genai-lifecycle.md` — the full evidence chain and lifecycle
  vocabulary
- `docs/llmops-playbook.md` — the practice map these controls belong to
