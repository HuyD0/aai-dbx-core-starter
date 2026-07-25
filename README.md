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
templates/agentic-rag/      custom Databricks project template
examples/                   focused learning examples
resources/                  this repository's bundle smoke job
infra/                      human-run keyless CI identity bootstrap
docs/                       developer and platform operating guides
.github/workflows/          credential-free CI and keyless deployment/release
```

## Install for SDK development

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest -q
```

## Codex Cloud development

The repository includes a reproducible, credential-free Codex Cloud toolchain.
Its setup and cached-container maintenance scripts install the pinned Azure,
Databricks, Terraform, Python, and `uv` versions automatically. Cloud tasks run:

```bash
./scripts/cloud-verify.sh
```

This performs the locked install check, linting, formatting, unit tests, wheel
build, Terraform validation, static Databricks bundle schema check, and YAML
parsing. The Codex agent container contains no Azure or Databricks credential.
After a reviewed change reaches protected `main`, GitHub Actions performs the
authenticated bundle validation and deployment through keyless OIDC.

Optional provider dependencies are separated:

```bash
python -m pip install -e '.[databricks,genai]'
python -m pip install -e '.[foundry,azure-search,keyvault,genai]'
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

## Generate an Agentic RAG project

```bash
databricks bundle init ./templates/agentic-rag \
  --output-dir ../my-agent
```

The generated project contains:

- An exploration notebook.
- A packaged, framework-neutral RAG agent.
- Logical model, embedding, and retrieval configuration.
- MLflow tracing and normalized retriever documents.
- Prompt registration.
- An offline evaluation gate.
- A wheel-based Databricks job.
- Keyless local and CI setup instructions.

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

- [Developer guide](docs/developer-guide.md)
- [Platform architecture](docs/platform-architecture.md)
- [Secrets and identity](docs/secrets-and-identity.md)
- [Tagging standard](docs/tagging-standard.md)
- [GenAI and RAG lifecycle](docs/genai-lifecycle.md)
- [Platform operations](docs/platform-operations.md)
