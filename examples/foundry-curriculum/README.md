# Microsoft Foundry curriculum project

This standalone learning project complements the repository's packaged SDK and
Databricks templates. Notebooks are for exploration and evidence gathering;
reusable application logic graduates into `src/` in a generated project.

## Structure

```text
foundry-curriculum/
├── README.md
├── CURRENT_PRACTICES.md              dated source and status guide
├── config/
│   ├── .gitignore
│   ├── aai-platform.dev.example.yml   portable, tracked example
│   └── aai-platform.dev.yml           local endpoint config, ignored by Git
├── data/
│   ├── evaluation_cases.jsonl         20 held-out starter cases
│   ├── context_cases.jsonl            context/security regression cases
│   └── a2a_cases.jsonl                routing and handoff regression cases
├── notebook_setup.py                  strict config and keyless call helpers
└── notebooks/
    ├── 00_setup_and_architecture.ipynb
    ├── 01_models_and_prompting.ipynb
    ├── 02_responses_and_structured_outputs.ipynb
    ├── 03_rag_and_retrieval_security.ipynb
    ├── 04_agents_tools_and_mcp.ipynb
    ├── 05_evaluation_safety_and_red_team.ipynb
    ├── 06_observability_and_genaiops.ipynb
    ├── 07_capstone_release_gate.ipynb
    ├── 08_context_engineering_and_memory.ipynb
    ├── 09_foundry_a2a_and_handoffs.ipynb
    ├── 10_foundry_native_evaluation.ipynb
    ├── 11_mlflow_tracing_and_genai_evaluation.ipynb
    └── 12_dual_otel_export_foundry_and_mlflow.ipynb
```

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
foundry:
  agent:
    name: <immutable agent name>
    version: <immutable agent version>
    id: <agent id used for trace correlation>
  evaluation:
    evaluator_model: <evaluator model deployment>
  memory:
    store_name: <pre-provisioned memory store>
  a2a:
    remote_agent_name: <pre-enabled remote agent>
    connection_name: <pre-provisioned A2A connection>
    protocol_version: "1.0"
  observability:
    application_insights_resource_id: <linked Application Insights resource ID>
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
uv sync --extra dev --extra foundry --extra foundry-labs --extra genai --extra examples --locked
az login
```

Open the repository in VS Code or JupyterLab, choose `.venv/bin/python`, and run
the notebooks in order. Every notebook is safe to run without a cloud request
by default. The first connected call requires both:

1. a real `deployment` value in the selected configuration; and
2. changing that notebook's explicit `RUN_CONNECTED` switch to `True`.

Advanced notebooks have narrower switches such as `RUN_A2A_CONNECTED`,
`RUN_FOUNDRY_EVAL`, and `RUN_DUAL_EXPORT`. They default to `False`, and they
skip with an explanation when an optional pre-provisioned resource is absent.
No notebook enables A2A, creates a project connection, changes RBAC, or creates
a memory store.

The switch is a learning guard, not a production control. Production access is
enforced by Microsoft Entra identity, Foundry RBAC, tool authorization, network
policy, evaluation gates, and deployment policy.

## Why the Foundry trace view can still be empty

The project must be connected to Application Insights before Foundry's Traces
view has a telemetry source. A normal model response is not enough: the client
must export OpenTelemetry spans to the connected resource and agent calls must
carry the agent/conversation correlation fields. The viewer also needs the
required Log Analytics permissions, and client traces can take a few minutes to
arrive.

Notebook 12 demonstrates the correct dual-export shape: one OpenTelemetry
provider sends the same spans to Azure Monitor and MLflow. It is not a sync
between the two backends. For a managed Databricks destination, use a
platform-owned collector or gateway with keyless authentication; never freeze a
Databricks bearer token into a Foundry Hosted Agent version.

See [CURRENT_PRACTICES.md](CURRENT_PRACTICES.md) for the dated status matrix and
official primary sources used by the advanced labs.

## Graduation rule

Preview features remain optional stretch work: Foundry A2A, managed Memory, and
trace/conversation evaluation must not become a production dependency without
an approved preview assessment and fallback. The capstone's required path uses
GA capabilities and must include its impact assessment, threat model, versioned
evaluation data, trace/privacy policy, release manifest, and tested rollback.
