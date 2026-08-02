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
│   └── aai-platform.dev.yml           local clone config, ignored by Git
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
    └── 07_capstone_release_gate.ipynb
```

## Configure it

In a fresh clone, create the ignored local configuration from the portable
example before editing settings or opening a notebook. From the repository
root, run:

```bash
cp examples/foundry-curriculum/config/aai-platform.dev.example.yml \
  examples/foundry-curriculum/config/aai-platform.dev.yml
```

Edit `examples/foundry-curriculum/config/aai-platform.dev.yml`, not the tracked
example. Set the clone's repository and approved catalog, then complete the
Foundry **project** endpoint and model deployment before making a connected
request:

```yaml
platform:
  repository: <owner>/<repository>
  catalog: <approved catalog>

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

## Graduation rule

Preview features are optional stretch work. The capstone's required path uses
GA capabilities and must include its impact assessment, threat model, versioned
evaluation data, trace/privacy policy, release manifest, and tested rollback.
