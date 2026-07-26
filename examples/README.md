# Learning examples

From a fresh clone, prove the SDK works and then create a local MLflow trace:

```bash
make quickstart
make local-start
make local-ui
```

`quickstart` creates or synchronizes the locked development environment and
runs the offline example. `local-start` writes to the isolated
`.aai/local/mlflow.db`; `local-ui` serves only that store at
`http://127.0.0.1:5000`. These steps need no cloud configuration or credentials.

Next, prepare keyless Databricks access and send the same trace to the workspace:

```bash
make workspace-connect
# Complete any reported authentication or configuration actions.
make workspace-example EXAMPLE=first_trace
```

The command creates a local, Git-ignored `aai-platform.yml` if needed and
reports incomplete configuration, Azure CLI authentication, or Databricks
connectivity without asking for a stored credential. After addressing its
reported actions, rerun `make workspace-connect` before the workspace example.

Use `make examples-list` to see all accepted names.

| Example | Requires |
|---|---|
| `offline_hello_world.py` | Nothing. No cloud, no config, no credentials. |
| `first_trace.py` | Local: `make local-start`. Workspace: keyless auth + `aai-platform.yml`. |
| `first_experiment.py` | Local: `make local-example EXAMPLE=first_experiment`. Workspace: keyless auth + `aai-platform.yml`. |
| `first_llm_call.ipynb` | `az login`, `DATABRICKS_HOST`, `aai-platform.yml`, and a serving endpoint with `CAN_QUERY`. |
| `first_prompt.py` | Keyless auth + `aai-platform.yml` + Unity Catalog prompt registry access. |
| `first_evaluation.py` | Keyless auth + `aai-platform.yml` + model access for LLM judges. |

Suggested order: offline hello world → local trace → local experiment →
workspace trace → first prompt → first evaluation → first LLM call.

The workspace runner supplies the non-secret Databricks host and MLflow routing
to the child process. If running a file directly instead, configure the shell:

```bash
make examples-install
az login
export DATABRICKS_HOST=<workspace host from platform-identifiers.json>
export DATABRICKS_AUTH_TYPE=azure-cli
export MLFLOW_TRACKING_URI=databricks
export MLFLOW_REGISTRY_URI=databricks-uc
cp aai-platform.example.yml aai-platform.yml  # then replace the placeholders
.venv/bin/python examples/first_trace.py
```

No example ever needs a PAT, client secret, or API key.

Workspace example results are stored in the configured Databricks experiment
and viewed in the Databricks workspace. `make local-ui` is deliberately pinned
to `.aai/local/mlflow.db`; a bare `mlflow ui` may select another Python and an
unrelated root database.

Do not run the connected files directly immediately after cloning. The Make
runner installs their optional dependencies and checks configuration and
identity before allowing an expensive or state-changing platform request.

## Where each example leads

| Example | Graduates into |
|---|---|
| `first_experiment.py` | `templates/experiment-starter` |
| `first_prompt.py` | `templates/prompt-app` |
| `first_evaluation.py` | `templates/evaluation-project` |
| `first_llm_call.ipynb`, `first_trace.py` | `templates/rag-app` / `templates/agent-app` |
| `offline_hello_world.py` | every template's hermetic test pattern |

## Notebook conventions

- **Jupyter (`.ipynb`) for local exploration** — like `first_llm_call.ipynb`.
- **Databricks-format `.py` notebooks for anything riding CD** — generated
  projects ship them and bundles sync them to the workspace.
- **No hardcoded configuration in either**: `bootstrap()` discovers
  `aai-platform.yml` by walking up from the working directory (override with
  `AAI_PLATFORM_CONFIG`), and experiments default to the platform naming
  convention `/Shared/<team>-<application>-<environment>` unless configured.
