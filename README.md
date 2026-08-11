# AAI Core

AAI Core is the AI/ML platform SDK and project-template hub for teams building
ML, RAG, and agentic applications on Azure Databricks.

It provides a paved road for:

- Keyless Azure and Databricks authentication.
- Azure Key Vault and Databricks secret references.
- Ownership, governance, lifecycle, and cost-attribution tags.
- Structured logging and MLflow tracing.
- Governed MLflow experiment/prompt context and deterministic gates over
  native MLflow evaluation results.
- Databricks and Microsoft Foundry model endpoints.
- Azure AI Search and Databricks AI Search retrieval.
- Reproducible application releases.
- Databricks Declarative Automation Bundles project templates.

The SDK deliberately keeps native clients and results available. The stable
model adapter is synchronous and non-streaming; advanced applications create a
caller-owned native async client for async, streaming, Responses API, or
framework-specific behavior.

## Repository layout

```text
src/aai_core/               installable platform SDK
src/platform_app/           guided onboarding console (a Databricks App)
templates/                  five AI lifecycle Databricks project templates
templates/_shared/          canonical scaffold synced into every template
examples/                   lifecycle examples and standalone sample projects
resources/                  this repository's bundle smoke job
docs/                       developer and platform operating guides
.github/workflows/          credential-free CI and keyless deployment/release
```

## Start locally, then move to the workspace

### Workstation prerequisites

Install these tools before running the repository commands:

| Tool | Needed for | Repository-certified version |
|---|---|---|
| Git | Clone and contribute to the repository | Current supported release |
| GNU Make | Run the documented `make` workflows | Current supported release |
| Python | SDK, examples, tests, and generated projects | 3.11 or 3.12; 3.12 is the default |
| `uv` | Create the locked Python environment | 0.8.23 |
| Azure CLI (`az`) | Keyless Azure login for workspace operations | 2.88.0 |
| Databricks CLI | Bundle generation, validation, and workspace operations | 1.9.0 |

The local, credential-free examples require Git, Make, Python, and `uv`.
Workspace examples and project generation additionally require Azure CLI and
Databricks CLI. Check what is available on your `PATH` with:

```bash
git --version
make --version
python3.12 --version
uv --version
az version
databricks version
```

Install missing tools using your organization's approved workstation process.
The repository versions are recorded in [`toolchain.json`](toolchain.json);
the [developer onboarding checklist](docs/developer-onboarding.md) explains the
required workspace access. A PAT, client secret, or API key is not a
prerequisite.

```bash
git clone https://github.com/HuyD0/aai-dbx-core-starter
cd aai-dbx-core-starter
make quickstart
make local-lifecycle
```

`make quickstart` uses the locked `uv` environment and runs the offline example.
It requires Python 3.11 or 3.12 and `uv` 0.8.23, but no cloud configuration or
credentials. The remaining commands carry one deterministic application
through a governed trace, named baseline/change experiment, exact prompt
lineage, and native MLflow evaluation gate. They use the isolated, Git-ignored
`.aai/local/mlflow.db` tracking and prompt-registry store. View it locally in
another terminal:

```bash
make local-ui
# Open http://127.0.0.1:5000; Ctrl-C stops the server.
```

The curriculum follows a fictional Aster Ridge Systems earnings-summary
assistant. Every company name, financial figure, and source identifier is
synthetic, and the assistant is prohibited from making investment
recommendations. The stable learning experiment is named for its decision
scope: `/Shared/example-ai-earnings-summary-quality-cost`. Runs carry
searchable purpose, change ID/summary, hypothesis, and baseline linkage rather
than names such as `first-comparison`.

The point is not merely to produce a fluent answer. The examples show why a
team must be able to answer:

- Which exact prompt produced this summary?
- What happened inside this one model request?
- Did the changed prompt improve the same cases as the baseline?
- Did quality improve without unacceptable latency, token, or cost growth?
- Is the evidence strong enough to adopt a release?

MLflow supplies different records for these questions: prompt versions preserve
instructions, traces explain individual requests, runs preserve test evidence,
and experiments collect comparable runs. The examples introduce those ideas in
that order and explain their purpose before using their APIs.

When the local lifecycle is understood, send the same evidence to the
configured Databricks experiment:

```bash
make workspace-connect
# Follow the reported configuration/authentication actions, then rerun it.
make workspace-example EXAMPLE=first_trace
```

The workspace setup creates a local, ignored `aai-platform.yml` when needed,
checks keyless Azure CLI and Databricks authentication, detects configuration
placeholders, and sets the MLflow tracking and registry destinations for the
example process. It never creates or requests a PAT, client secret, or API key.
The connected notebook uses the non-secret workspace host from
`platform-identifiers.json` and explicitly selects `azure-cli` authentication,
so it does not require a Databricks CLI profile. If you create a profile for
other CLI work, it is not automatically inherited by an already-running
notebook kernel; restart the editor/kernel after changing its environment.
Workspace traces and runs are viewed in the configured Databricks experiment.
The local UI and Databricks workspace are deliberately separate destinations;
the commands always print which one they used.

List every example and its execution mode with `make examples-list`.
See the [progressive executable curriculum](examples/README.md) for its
baseline/change/result/decision/release rubric and the boundary between
`aai-core` contracts and native MLflow APIs.

### Prepare an offline fine-tuning study pack

The standalone [local fine-tuning sample](examples/local-finetuning/README.md)
uses a pinned public Kaggle dataset and a small MLX model on Apple silicon. It
has its own environment and lock so Apple-only training dependencies never
enter the cross-platform SDK runtime. Prepare it while connected, then prove it
works with downloads disabled and Python sockets blocked:

```bash
make study-prepare-flight
make study-offline-check
```

Only leave for offline study after the check prints `READY FOR OFFLINE STUDY`.
The project records source/model hashes, leakage-safe balanced splits, local
MLflow evidence, deterministic baselines, LoRA evaluation, and a policy-derived
application-readiness capstone. Kaggle files, model weights, adapters, and local
experiment stores stay ignored by Git.

Study is notebook-led: `make notebook` registers the exact nested Python kernel
and opens a numbered 12-notebook course covering provenance, data quality,
leakage-safe splits, baselines, prompting, LoRA, frozen evaluation, MLflow
decisions, and the capstone. CLI targets remain optional automation for
preflight and long runs.

### Learn classical classification locally

The standalone [local classification course](examples/local-classification/README.md)
trains a small sklearn subscription-churn model on deterministic synthetic data.
It needs no download, cloud credential, GPU, or Databricks workspace and keeps
its dependencies in a separate exact lock.

```bash
make classification-install
make classification-check
make classification-notebook
```

The ten notebooks cover problem framing, data contracts, time-based leakage-safe
splits, a no-skill baseline, sklearn Pipelines, explicit MLflow tracking, model
and threshold selection, one frozen-test gate, conditional registry promotion,
model reload, monitoring, and a current Unity Catalog/Databricks handoff. Run a
completed lifecycle with `make classification-train`, then inspect it in another
terminal with `make classification-ui`.

The course uses the same lifecycle vocabulary as the platform—`baseline ->
change -> result -> decision`, with `adopt`, `reject`, or `inconclusive`—while
keeping sklearn out of the SDK runtime.

### Learn governed agentic operations and RAG

The standalone [agentic operations RAG workshop](examples/agentic-ops-rag/README.md)
adapts a progressive public RAG course outline to this platform's MLflow 3,
Databricks, Microsoft Foundry, Azure AI Search, identity, and release contracts.
Its six generated notebooks use original synthetic runbooks and execute without
credentials by default; real provider calls are explicit opt-ins.

```bash
make ops-rag-install
make ops-rag-check
make ops-rag-notebook
```

The workshop compares text, vector, hybrid, and reranked retrieval, records
normalized MLflow document evidence, enforces tenant and action boundaries, and
finishes with a baseline/change/result/decision release gate. Production work
graduates to the existing `rag-app` or `agent-app` template.

### Use the Email Support Agent reference design

The standalone [Email Support Agent solution accelerator](examples/email-support-agent/README.md)
turns a reviewed LangGraph demo into a production-minded reference design. Its
offline package and synthetic release set demonstrate redacted checkpoint
state, deterministic risk routing, identity/group/release-scoped retrieval,
grounded drafting, identity-authorized human approval, namespaced idempotency
keys, an atomic transactional outbox and delivery worker, MLflow
span/evaluation/monitoring contracts, governed feedback, pre-spend
request/judge budgets, and a fail-closed operational-readiness scorecard. Its
[maturity scorecard](examples/email-support-agent/MATURITY_SCORECARD.md)
separates 10/10 reference coverage from live production certification.

```bash
make email-support-check
make email-support-demo
make email-support-workshop
make email-support-mlops-plan
```

The accelerator intentionally graduates into `agent-app` rather than
duplicating bundle, identity, serving, locking, and cost-tag scaffolding.

## Install for SDK development

```bash
make install
make hooks-install
make check
```

`make install` creates or synchronizes `.venv` from `uv.lock`. Run `make help`
to see focused targets for formatting, tests, builds, template synchronization,
and authenticated Databricks bundle validation.
`make hooks-install` installs a fast credential-free commit hook and the full
CI-equivalent verifier as a pre-push hook. Both use only repository-local hook
definitions—no third-party hook repository is downloaded or executed. Run them
manually with `make pre-commit`, `make pre-push`, or `make hooks-run`.

Optional provider dependencies are separated:

```bash
uv sync --extra databricks --extra genai --locked
uv sync --extra foundry --extra azure-search --extra keyvault --extra genai --locked
```

## Configure an application

Copy [`aai-platform.example.yml`](aai-platform.example.yml) to
`aai-platform.yml` in a consuming project. The file contains identifiers and
secret references, never secret values.

```python
from aai_core import bootstrap

ctx = bootstrap()
model = ctx.providers.model("general-chat")
retriever = ctx.providers.retriever("product-knowledge")
```

Run safe local diagnostics:

```bash
aai-core doctor
```

After authenticating with Azure CLI and Databricks unified authentication:

```bash
aai-core doctor --cloud
```

## Generate a project

Six templates cover the AI application lifecycle; pick by what the team is
building:

| Template | Use when you want |
|---|---|
| `experiment-starter` | Reproducible MLflow experiments (LLM-free): dataset lineage, tags, metrics, artifacts, deterministic gate |
| `prompt-app` | A governed prompt lifecycle: versioned registration, pinned-version LLM-judge evaluation, gated alias promotion |
| `evaluation-project` | A standalone eval harness for an existing app/endpoint, driven by the `agentkit` CLI: UC datasets, registry scorers, pinned baselines, CI regression gate, published evidence |
| `rag-app` | Governed RAG: chunking pipeline, declared vector index (or Azure AI Search), traced grounded generation, groundedness gate |
| `agent-app` | Tool-using agents: application-owned async loop, Pydantic outputs/tools, trajectory-aware evals, native MLflow Agent Server invoke/stream, and optional LangGraph recipe |
| `analytics-app` | Self-service analytics: a runbook agent over a neutral git-versioned semantic layer, knowledge-doc router, provenance footers, snapshot-pinned golden evals, and a warehouse-portable executor protocol |

```bash
az login
source scripts/platform-env.sh
databricks bundle init "$AAI_TEMPLATE_REPO" \
  --template-dir templates/<template-name> --output-dir my-project
cd my-project
python3.12 scripts/setup_dev.py
```

See the [developer onboarding checklist](docs/developer-onboarding.md) for the
group permissions and workstation tools required before generation. The
generated setup command validates access, creates `.venv`, installs the pinned
checksum-verified SDK and project dependencies, and runs the offline checks.

Every generated project shares the same spine: pinned checksum-verified
`aai-core`, strict boundary schemas, the 9 mandatory cost tags on bundle
presets and job clusters,
credential-free PR CI with a deterministic gate tier, a keyless OIDC deploy
workflow, hermetic tests on `aai_core.testing` fakes, and an
`.aai-template.json` provenance stamp. (`agentic-rag` retired into
`rag-app` + `agent-app`; it is last renderable at tag
`v0.2.0-agentic-rag-final` and each successor records it in `supersedes`.)

The SDK stays close to MLflow, Databricks, and provider APIs. It supplies
governed defaults and typed evidence contracts, returns native result objects,
and exposes native clients for features outside the paved road. Terms are
kept deliberately small: baseline, change, result, and an
adopt/reject/inconclusive decision.

Compatibility is maintained as code: `compatibility.json` declares the SDK,
template, Python, runtime, and feature-support matrix;
`dependency-policy.toml` declares supported and certified library versions;
`uv.lock` records the exact certified SDK development stack; generated
projects carry exact universal transitive runtime locks. PRs test those locks,
and the scheduled credential-free canary tests minimum and latest supported
dependency resolutions on every supported Python version.

## Publish the private SDK

Wheels are released immutably to:

```text
<sdk_artifact_volume>/aai_core/<version>/
```

where `sdk_artifact_volume` comes from `platform-identifiers.json`.

From the `main` branch, run the `publish-sdk` workflow with the exact version
from `pyproject.toml`. The release identity uses GitHub OIDC and receives only
the Databricks `WRITE VOLUME` permission needed for this volume. No GitHub
secret or package-registry token is used.

Generated projects pin an exact version and verify its SHA-256 checksum before
local installation.

## Security boundaries

- No PAT, client secret, storage key, or API key belongs in Git.
- Pull-request CI has no OIDC permission and performs no cloud login.
- Credentialed workflows have no GitHub `environment:` because the configured
  federated identity subject is the `main` branch-ref form.
- CI has no Azure ARM role and is constrained by Databricks object privileges.
- Tags contain no secrets or personal data.
- Production applications must choose managed or workload identity explicitly.

Read [`AGENTS.md`](AGENTS.md) before making repository changes. Connection and
recovery instructions remain in [`docs/cloud-setup.md`](docs/cloud-setup.md);
cloud and identity resources are provisioned outside this repository.

## Evaluating an agent

`agentkit` is the paved road for agent evaluation, built around one idea: an
experiment is a comparison, not a log. It scores this version of an agent
against the recorded baseline on the same dataset with the same versioned
scorers, and generates the MLflow run, lineage, and evidence as byproducts.

```bash
agentkit smoke                        # seconds, free, no cluster
agentkit compare --establish-baseline # this run IS the baseline
agentkit compare                      # is this better than what we had?
agentkit gate                         # exit 0 pass / 2 threshold failed
agentkit evidence                     # the release record
```

Read [`docs/agent-evaluation.md`](docs/agent-evaluation.md) for why comparison
is the unit of evidence, how the shared scorer registry keeps one team's 0.8
comparable with another's, and what the gate refuses.

## Learning paths

- `make quickstart` — clone-to-running, with zero credentials
- `make local-lifecycle` — compare two exact versions of a fictional
  earnings-summary prompt through the complete local trace, experiment, and
  evaluation lifecycle
- `make local-ui` — inspect the isolated local tracking and prompt store
- `make workspace-connect` — guided keyless setup for workspace examples
- `make app-run` — the guided platform console, served locally ([docs](docs/platform-console.md))
- [Progressive lifecycle examples](examples/README.md)
- [00 — Offline hello world](examples/00_offline_hello_world.py) — zero credentials
- [01 — First governed trace](examples/01_first_trace.py)
- [02 — First baseline/change experiment](examples/02_first_experiment.py)
- [03 — First exact prompt lineage](examples/03_first_prompt.py)
- [04 — First deterministic evaluation gate](examples/04_first_evaluation.py)
- [05 — Connected notebook setup checks](examples/05_connected_setup.ipynb)
- [06 — Connected stable first call](examples/06_connected_first_call.py)
- [07 — Native async/streaming comparison](examples/07_first_llm_call.ipynb)
- [08 — Tool-trajectory evaluation](examples/08_tool_trajectory_evaluation.ipynb)
- [09 — Multi-turn session evaluation](examples/09_multi_turn_session_evaluation.ipynb)
- [10 — Layered and calibrated judges](examples/10_layered_judges.ipynb)
- [11 — Cost-quality trade-off](examples/11_cost_quality_tradeoff.ipynb)
- [12 — Optional agent alignment and optimization](examples/12_agent_alignment_optimization.ipynb)
- [13 — Decision records and gated promotion](examples/13_decision_and_promotion_lifecycle.ipynb)
- [14 — Platform LLM operations](examples/14_platform_llm_operations.ipynb)
- [Local classical-classification course](examples/local-classification/README.md)
- [Offline Apple-silicon fine-tuning sample](examples/local-finetuning/README.md)
- [Governed batch inference pattern](examples/governed-batch-inference/README.md)
- [Developer guide](docs/developer-guide.md)
- [LLMOps playbook](docs/llmops-playbook.md) — industry LLMOps practice map
  onto this platform, for application teams and the platform team
- [Platform architecture](docs/platform-architecture.md)
- [Secrets and identity](docs/secrets-and-identity.md)
- [SDK versioning policy](docs/versioning.md)
- [Enterprise clone runbook](docs/enterprise-clone-runbook.md) — including
  how a downstream clone tracks this repository without re-resolving the
  same identifier conflicts on every sync
- [Platform audit](docs/platform-audit.md) — findings and prioritised backlog
- [Tagging standard](docs/tagging-standard.md)
- [Cost estimation](docs/cost-estimation.md) — the console's list-price estimator
  and its pricing snapshot
- [GenAI and RAG lifecycle](docs/genai-lifecycle.md)
- [Self-service analytics lifecycle](docs/analytics-lifecycle.md)
- [Platform operations](docs/platform-operations.md)
