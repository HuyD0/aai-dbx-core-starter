# AAI Core

AAI Core is the AI/ML platform SDK and project-template hub for teams building
ML, RAG, and agentic applications on Azure Databricks.

It provides a paved road for:

- Keyless Azure and Databricks authentication.
- Azure Key Vault and Databricks secret references.
- Ownership, governance, lifecycle, and cost-attribution tags.
- Structured logging and MLflow tracing.
- MLflow experiments, prompts, evaluation datasets, scorers, and feedback.
- Databricks and Microsoft Foundry model endpoints.
- Azure AI Search and Databricks AI Search retrieval.
- Reproducible application releases.
- Databricks Declarative Automation Bundles project templates.

The SDK deliberately keeps native clients available. It standardizes platform
contracts without hiding useful differences between providers.

## Repository layout

```text
src/aai_core/               installable platform SDK
templates/                  five lifecycle-ladder Databricks project templates
templates/_shared/          canonical scaffold synced into every template
examples/                   focused learning examples
resources/                  this repository's bundle smoke job
infra/                      human-run keyless CI identity bootstrap
docs/                       developer and platform operating guides
.github/workflows/          credential-free CI and keyless deployment/release
```

## Start from a fresh clone

```bash
git clone https://github.com/HuyD0/aai-dbx-core-starter
cd aai-dbx-core-starter
make quickstart
```

`make quickstart` uses the locked `uv` environment and runs the offline example.
It requires Python 3.11 or 3.12 and `uv` 0.8.23, but no cloud configuration or
credentials. If `uv` is unavailable, the target prints the pinned installation
command instead of continuing with a partial environment. To continue into
examples that write to Azure Databricks:

```bash
make examples-connect
# Follow the reported configuration/authentication actions, then rerun it.
make example EXAMPLE=first_trace
```

The connected setup creates a local, ignored `aai-platform.yml` when needed,
checks keyless Azure CLI and Databricks authentication, detects configuration
placeholders, and sets the MLflow tracking and registry destinations for the
example process. It never creates or requests a PAT, client secret, or API key.
Connected traces and runs are viewed in the configured Databricks experiment;
they are not served by a local `mlflow ui`.

List every example and its execution mode with `make examples-list`.

## Install for SDK development

```bash
make install
make check
```

`make install` creates or synchronizes `.venv` from `uv.lock`. Run `make help`
to see focused targets for formatting, tests, builds, Terraform validation,
template synchronization, and authenticated Databricks bundle validation.
`make verify` runs the complete credential-free verification path used by CI.

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

Five templates cover the lifecycle ladder; pick by what the team is
building:

| Template | Use when you want |
|---|---|
| `experiment-starter` | Reproducible MLflow experiments (LLM-free): dataset lineage, tags, metrics, artifacts, deterministic gate |
| `prompt-app` | A governed prompt lifecycle: versioned registration, pinned-version LLM-judge evaluation, gated alias promotion |
| `evaluation-project` | A standalone eval harness for an existing app/endpoint: UC datasets, reusable scorers, baselines, CI regression gate, published results |
| `rag-app` | Governed RAG: chunking pipeline, declared vector index (or Azure AI Search), traced grounded generation, groundedness gate |
| `agent-app` | Tool-using agents: SDK tool loop, structured outputs, trajectory-aware evals, feedback, gated Model Serving deploy, monitoring |

```bash
databricks bundle init https://github.com/HuyD0/aai-dbx-core-starter \
  --template-dir templates/<template-name> --output-dir my-project
```

Every generated project shares the same spine: pinned checksum-verified
`aai-core`, the 9 mandatory cost tags on bundle presets and job clusters,
credential-free PR CI with a deterministic gate tier, a keyless OIDC deploy
workflow, hermetic tests on `aai_core.testing` fakes, and an
`.aai-template.json` provenance stamp. (`agentic-rag` retired into
`rag-app` + `agent-app` — see `templates/agentic-rag/README.md`.)

## Publish the private SDK

Wheels are released immutably to:

```text
/Volumes/platform/artifacts/python_packages/aai_core/<version>/
```

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

Read [`AGENTS.md`](AGENTS.md) before making repository changes. Provisioning and
recovery instructions remain in [`docs/cloud-setup.md`](docs/cloud-setup.md).

## Learning paths

- `make quickstart` — clone-to-running, with zero credentials
- `make examples-connect` — guided keyless setup for connected examples
- [Offline hello world](examples/offline_hello_world.py) — zero credentials
- [First LLM call notebook](examples/first_llm_call.ipynb)
- [Developer guide](docs/developer-guide.md)
- [Platform architecture](docs/platform-architecture.md)
- [Secrets and identity](docs/secrets-and-identity.md)
- [SDK versioning policy](docs/versioning.md)
- [Enterprise clone runbook](docs/enterprise-clone-runbook.md)
- [Tagging standard](docs/tagging-standard.md)
- [GenAI and RAG lifecycle](docs/genai-lifecycle.md)
- [Platform operations](docs/platform-operations.md)
