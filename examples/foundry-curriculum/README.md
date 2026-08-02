# Microsoft Foundry curriculum project

This standalone learning project complements the repository's packaged SDK and
Databricks templates. Notebooks are for exploration and evidence gathering;
reusable application logic graduates into `src/` in a generated project.

## Structure

```text
foundry-curriculum/
├── README.md
├── config/
│   ├── .gitignore
│   ├── aai-platform.dev.example.yml   portable, tracked example
│   └── aai-platform.dev.yml           local endpoint config, ignored by Git
├── data/
│   └── evaluation_cases.jsonl         20 held-out starter cases
├── notebook_setup.py                  validation and keyless Responses helper
└── notebooks/
    ├── 00_setup_and_architecture.ipynb
    ├── 01_models_and_prompting.ipynb
    ├── 02_responses_and_structured_outputs.ipynb
    ├── 03_rag_and_retrieval_security.ipynb
    ├── 04_agents_tools_and_mcp.ipynb
    ├── 05_evaluation_safety_and_red_team.ipynb
    ├── 06_observability_and_genaiops.ipynb
    ├── 07_capstone_release_gate.ipynb
    └── 08_agentops_release_gate.ipynb
```

## How the notebooks teach

Every executable cell is surrounded by prose in the same three-part shape:
the concept and the failure it prevents, the cell, then a **What you just saw**
read-back that names the one line carrying the lesson and a **Change this and
re-run** mutation with its expected outcome. A test enforces that no code cell
is left un-narrated. Run the mutations — several of them are designed to make a
gate look healthier while measuring less, which is the point.

## Configure it

A clean checkout intentionally contains only the portable example. Before
opening any notebook, copy it to the ignored local development path:

```bash
cp examples/foundry-curriculum/config/aai-platform.dev.example.yml examples/foundry-curriculum/config/aai-platform.dev.yml
```

Edit the new file so `platform.repository` names this clone as
`<owner>/<repository>`. Disconnected exercises may keep the Foundry endpoint
and deployment placeholders. Replace both before making a connected request:

```yaml
platform:
  repository: <owner>/<repository>
providers:
  models:
    foundry-chat:
      provider: foundry
      endpoint: <project endpoint>
      deployment: <model deployment name>
```

To use a different environment file, set the SDK's standard configuration
override before starting Jupyter:

```bash
export AAI_PLATFORM_CONFIG=/absolute/path/to/aai-platform.yml
```

Endpoints are non-secret identifiers, but API keys and bearer tokens must never
be stored in these files. The notebooks use `azure_identity: azure_cli`; run
`az login` and obtain the least-privilege Foundry project role through the
approved platform process.

## Run the notebooks

From the repository root, install the locked interactive and Foundry extras:

```bash
uv sync --extra dev --extra foundry --extra genai --extra examples --locked
az login
```

Open the repository in VS Code or JupyterLab, choose `.venv/bin/python`, and run
the notebooks in order. Every notebook is safe to run without a cloud request
by default. The first connected call requires both:

1. a real `deployment` value in the selected configuration; and
2. changing that notebook's explicit `RUN_CONNECTED` switch to `True`.

The switch is a learning guard, not a production control. Production access is
enforced by Microsoft Entra identity, Foundry RBAC, tool authorization, network
policy, evaluation gates, and deployment policy.

## Notebook 08 and the AgentOps Accelerator

Notebook 08 teaches release gating with the
[AgentOps Accelerator](https://azure.github.io/agentops/). It runs entirely
offline — nothing in this curriculum installs or invokes the CLI. Two notes if
you go on to use it for real:

- **Install `agentops-accelerator`, not `agentops`.** An unrelated third-party
  product owns the shorter name on PyPI. The correct spec is
  `agentops-accelerator[foundry,agent]`.
- **Do not commit its generated workflows into this repository.** `agentops
  workflow generate` emits a PR gate carrying `id-token: write` and
  `azure/login` on a `pull_request` trigger, with an `environment:` on the
  credentialed job — three violations of `AGENTS.md` §4. The notebook teaches
  the reconciliation; a downstream project not under these rules can adopt the
  generated form directly.

## Graduation rule

Preview features are optional stretch work. The capstone's required path uses
GA capabilities and must include its impact assessment, threat model, versioned
evaluation data, trace/privacy policy, release manifest, and tested rollback.
