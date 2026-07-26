# Learning examples

Start with the offline example — it needs nothing but this checkout. Every
other example talks to real platform services and states its prerequisites
explicitly.

| Example | Requires |
|---|---|
| `offline_hello_world.py` | Nothing. No cloud, no config, no credentials. |
| `first_llm_call.ipynb` | `az login`, `DATABRICKS_HOST`, `aai-platform.yml`, and a serving endpoint with `CAN_QUERY`. |
| `first_experiment.py` | Keyless auth + `aai-platform.yml` + a Databricks MLflow experiment path. |
| `first_trace.py` | Keyless auth + `aai-platform.yml` (writes traces to the workspace). |
| `first_prompt.py` | Keyless auth + `aai-platform.yml` + Unity Catalog prompt registry access. |
| `first_evaluation.py` | Keyless auth + `aai-platform.yml` + model access for LLM judges. |

Suggested order: offline hello world → first LLM call → first experiment →
first trace → first prompt → first evaluation.

Keyless auth for the cloud examples:

```bash
az login
export DATABRICKS_HOST=<workspace host from platform-identifiers.json>
export DATABRICKS_AUTH_TYPE=azure-cli
cp aai-platform.example.yml aai-platform.yml  # then replace the placeholders
```

No example ever needs a PAT, client secret, or API key.
