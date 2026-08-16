"""Render combinations and expectations per template.

Each combo renders the template with schema defaults + platform-identifier
overrides + the combo's `overrides`, then asserts `expect_present` /
`expect_absent` paths (relative to the rendered project root). Every file
toggled by a template's __preamble must be asserted present in one combo and
absent in a sibling — that is the dead-skip-glob guard.

Every combo runs the deep generated-project tier (validation, Ruff, Black,
mypy, branch coverage, offline checks, and a wheel build), so each branch must
render a fully working project.
"""

COMBOS = {
    "analytics-app": [
        {
            "name": "databricks-review-off",
            "overrides": {
                "project_name": "test-analytics",
                "model_provider": "databricks",
                "model_deployment": "chat",
            },
            "expect_present": [
                "semantics/semantic_model.yml",
                "src/app/semantics/models.py",
                "src/app/semantics/compiler.py",
                "src/app/semantics/executor.py",
                "src/app/knowledge.py",
                "src/app/provenance.py",
                "src/app/tools.py",
                "src/app/agent.py",
                "src/app/reviewer.py",
                "src/app/scorers.py",
                "knowledge/orders.md",
                "knowledge/customers.md",
                "knowledge/metrics_definitions.md",
                "prompts/system_prompt.json",
                "prompts/reviewer_prompt.json",
                "jobs/seed_lakehouse.py",
                "resources/analytics_job.yml",
                "evals/evaluate.py",
                "evals/offline_checks.py",
                "evals/data/seed_data.json",
                "evals/data/golden_cases.json",
                "evals/data/answer_sheet.json",
                "notebooks/01_explore_semantics.py",
                "notebooks/02_context_engineering.py",
                "notebooks/03_run_the_agent.py",
                "notebooks/04_evaluate_and_gate.py",
            ],
            # No serving surface by design: projects graduate to agent-app.
            "expect_absent": ["app.yaml", "start_server.py", "resources/agent_app.yml"],
        },
        {
            "name": "databricks-review-on",
            "overrides": {
                "project_name": "test-analytics",
                "model_provider": "databricks",
                "model_deployment": "chat",
                "adversarial_review": "yes",
            },
            "expect_present": ["src/app/agent.py", "src/app/reviewer.py"],
            "expect_absent": [],
        },
    ],
    "experiment-starter": [
        {
            "name": "with-notebook",
            "overrides": {"project_name": "test-experiment"},
            "expect_present": [
                "src/app/experiment.py",
                "jobs/run_experiment.py",
                "data/sample.csv",
                "notebooks/01_explore.py",
                "evals/evaluate.py",
            ],
            "expect_absent": [],
        },
        {
            "name": "no-notebook",
            "overrides": {
                "project_name": "test-experiment",
                "include_notebook": "no",
            },
            "expect_present": ["src/app/experiment.py"],
            "expect_absent": ["notebooks/01_explore.py", "notebooks"],
        },
    ],
    "prompt-app": [
        {
            "name": "databricks",
            "overrides": {"project_name": "test-prompt-app"},
            "expect_present": [
                "src/app/assistant.py",
                "src/app/config.py",
                "prompts/system_prompt.json",
                "scripts/register_prompt.py",
                "scripts/promote_prompt.py",
                "evals/evaluate.py",
            ],
            "expect_absent": [],
        },
    ],
    "evaluation-project": [
        {
            "name": "databricks-judge",
            "overrides": {
                "project_name": "test-evaluation",
                "model_deployment": "judge-endpoint",
            },
            "expect_present": [
                "agentkit.yaml",
                "src/app/example_agent.py",
                "src/app/scorers.py",
                "src/app/judges.py",
                "src/app/targets.py",
                "scripts/sync_dataset.py",
                "scripts/link_deployment_job.py",
                "evals/data/answer_sheet.json",
                "notebooks/01_align_judge.py",
                "resources/optional/deployment_job.yml",
                "resources/optional/registered_model.yml",
            ],
            # Thresholds live in agentkit.yaml now; scorer categorization is
            # structural (a scorer's kind comes from the shared registry).
            "expect_absent": ["evals/gate_config.json"],
        },
    ],
    "rag-app": [
        {
            "name": "dbx-dbx-search",
            "overrides": {
                "project_name": "test-rag",
                "model_provider": "databricks",
                "model_deployment": "chat",
                "retrieval_provider": "databricks_ai_search",
                "search_endpoint": "https://unused",
                "search_index": "knowledge",
                "embedding_deployment": "embedding",
            },
            "expect_present": [
                "src/app/rag.py",
                "src/app/release_evidence.py",
                "jobs/build_chunks.py",
                "tests/test_chunks.py",
                "tests/test_release_evidence.py",
                "scripts/promote_prompt.py",
            ],
            "expect_absent": [],
        },
        {
            "name": "dbx-azure-search",
            "overrides": {
                "project_name": "test-rag",
                "model_provider": "databricks",
                "model_deployment": "chat",
                "retrieval_provider": "azure_ai_search",
                "search_endpoint": "https://search.search.windows.net",
                "search_index": "knowledge",
                "embedding_deployment": "embedding",
            },
            "expect_present": ["src/app/rag.py"],
            "expect_absent": [
                "jobs/build_chunks.py",
                "jobs",
                "tests/test_chunks.py",
            ],
        },
    ],
    "agent-app": [
        {
            "name": "dbx-serving",
            "overrides": {
                "project_name": "test-agent-app",
                "model_provider": "databricks",
                "model_deployment": "chat",
            },
            "expect_present": [
                "src/app/agent.py",
                "src/app/messages.py",
                "src/app/tool_scoring.py",
                "src/app/tools.py",
                "src/app/endpoint.py",
                "start_server.py",
                "app.yaml",
                "requirements.txt",
                "resources/agent_app.yml",
                "tests/_endpoint_trace_probe.py",
                "tests/test_app_endpoint.py",
                "notebooks/02_enable_monitoring.py",
                "scripts/create_release.py",
                "tests/test_evaluation_config.py",
                "tests/test_feedback.py",
                "tests/test_release_evidence.py",
                "tests/test_sync_dataset.py",
            ],
            "expect_absent": ["src/app/scoring.py"],
        },
        {
            "name": "dbx-no-serving",
            "overrides": {
                "project_name": "test-agent-app",
                "model_provider": "databricks",
                "model_deployment": "chat",
                "include_serving": "no",
            },
            "expect_present": ["src/app/agent.py", "notebooks/01_explore.py"],
            "expect_absent": [
                "notebooks/02_enable_monitoring.py",
                "src/app/endpoint.py",
                "start_server.py",
                "app.yaml",
                "requirements.txt",
                "resources/agent_app.yml",
                "tests/_endpoint_trace_probe.py",
                "tests/test_app_endpoint.py",
            ],
        },
    ],
}
