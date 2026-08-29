# Developer onboarding

The platform team completes access onboarding before an application developer
generates a project. The generated setup command verifies that onboarding; it
never grants permissions or creates cloud resources.

## Platform-team checklist

Confirm the developer's approved group has:

- Access to the `dbx-dev` Databricks workspace.
- `READ VOLUME` on the SDK artifact volume (`sdk_artifact_volume` in
  `platform-identifiers.json`).
- `CAN_USE` on the constrained job compute policy (`job_compute_policy_id`
  in the same file).
- The application-specific catalog, schema, model endpoint, and search
  permissions required by the chosen template.
- Access to create or push the consuming GitHub repository.

Grant permissions to groups rather than individual users. Application
developers never need `WRITE VOLUME`, workspace admin, a PAT, or a client
secret.

## Wizard values the platform team hands over

Several wizard prompts default to `replace-with-*` placeholders because their
real values name externally provisioned, application-specific resources. A
developer cannot discover them alone, so hand them over together with the
access grants above. Collect the rows for the chosen template:

| Wizard prompt | Hand over | Asked by |
|---|---|---|
| `model_deployment` | Name of the approved Databricks model serving endpoint for the application | every template |
| `judge_deployment` | Name of the approved gateway-fronted serving endpoint the LLM judges call | agent-app, analytics-app, prompt-app, rag-app |
| `experiment_id` | ID of the pre-created MLflow experiment (the App identity needs `CAN_EDIT`) | agent-app |
| `app_usage_policy_id` | Externally provisioned Databricks Apps usage policy ID | agent-app |
| `warehouse_id` | Serverless SQL warehouse id (the deploy identity needs `CAN USE`) | analytics-app |
| `search_endpoint` | Azure AI Search service URL or Databricks AI Search endpoint | rag-app |
| `search_index` | Name of the provisioned search index | rag-app |
| `embedding_deployment` | Embedding endpoint whose dimension matches the index's vector field | rag-app |
| `knowledge_version` | Immutable identifier of the current knowledge/chunk/index snapshot | rag-app |

The developer supplies `repository_url` themselves: the HTTPS URL of the
consuming repository from the last checklist item. Every other wizard default
is already stamped with the approved platform value.

## Developer workstation checklist

Install:

- Git.
- Python 3.12 (Python 3.11 is also supported).
- Azure CLI.
- Databricks CLI.

Authenticate before invoking the template wizard. Run these from a checkout of
this repository: `platform-env.sh` reads `platform-identifiers.json`, so the
commands below carry your platform's workspace and template repository without
either being written down here.

```bash
az login
source scripts/platform-env.sh
```

Generate the selected project:

```bash
databricks bundle init "$AAI_TEMPLATE_REPO" \
  --template-dir templates/<template-name> --output-dir my-project
cd my-project
```

The wizard asks for application ownership, cost, catalog/schema, and provider
choices. Platform-controlled fields already default to the approved workspace,
compute policy, SDK version, and SDK artifact volume.

Run the generated setup:

```bash
python3.12 scripts/setup_dev.py
```

For a non-mutating access check without creating `.venv` or installing
packages:

```bash
python3.12 scripts/setup_dev.py --check-only
```

The preflight verifies the local toolchain, Azure and Databricks
authentication, both pinned SDK release artifacts, and compute-policy
visibility. A failure identifies the missing prerequisite and tells the
developer what to request from the platform team.
