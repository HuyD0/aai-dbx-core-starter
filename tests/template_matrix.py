"""Render combinations and expectations per template.

Each combo renders the template with schema defaults + platform-identifier
overrides + the combo's `overrides`, then asserts `expect_present` /
`expect_absent` paths (relative to the rendered project root). Every file
toggled by a template's __preamble must be asserted present in one combo and
absent in a sibling — that is the dead-skip-glob guard.

The first combo of each template is also the deep-tier combo (ruff, black,
generated pytest, offline checks), so it must render a fully working project.
"""

COMBOS = {
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
        {
            "name": "foundry",
            "overrides": {
                "project_name": "test-prompt-app",
                "model_provider": "foundry",
                "foundry_endpoint": "https://unused.services.ai.azure.com",
            },
            "expect_present": ["src/app/assistant.py"],
            "expect_absent": [],
        },
    ],
    "evaluation-project": [
        {
            "name": "databricks-judge",
            "overrides": {"project_name": "test-evaluation"},
            "expect_present": [
                "src/app/scorers.py",
                "src/app/judges.py",
                "src/app/targets.py",
                "scripts/sync_dataset.py",
                "evals/data/answer_sheet.json",
                "notebooks/01_align_judge.py",
            ],
            "expect_absent": [],
        },
        {
            "name": "foundry-judge",
            "overrides": {
                "project_name": "test-evaluation",
                "model_provider": "foundry",
                "foundry_endpoint": "https://unused.services.ai.azure.com",
            },
            "expect_present": ["src/app/judges.py"],
            "expect_absent": [],
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
                "jobs/build_chunks.py",
                "tests/test_chunks.py",
                "resources/index.yml",
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
                "resources/index.yml",
                "jobs/build_chunks.py",
                "jobs",
                "tests/test_chunks.py",
            ],
        },
        {
            "name": "foundry-azure-search",
            "overrides": {
                "project_name": "test-rag",
                "model_provider": "foundry",
                "foundry_endpoint": "https://unused.services.ai.azure.com",
                "model_deployment": "chat",
                "retrieval_provider": "azure_ai_search",
                "search_endpoint": "https://search.search.windows.net",
                "search_index": "knowledge",
                "embedding_deployment": "embedding",
            },
            "expect_present": ["src/app/rag.py"],
            "expect_absent": ["resources/index.yml"],
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
                "src/app/tools.py",
                "src/app/scoring.py",
                "serving/model.py",
                "scripts/deploy_serving.py",
                "notebooks/02_enable_monitoring.py",
            ],
            "expect_absent": [],
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
                "serving/model.py",
                "serving",
                "scripts/deploy_serving.py",
                "notebooks/02_enable_monitoring.py",
            ],
        },
        {
            "name": "foundry-serving",
            "overrides": {
                "project_name": "test-agent-app",
                "model_provider": "foundry",
                "foundry_endpoint": "https://unused.services.ai.azure.com",
                "model_deployment": "chat",
            },
            "expect_present": ["src/app/agent.py", "serving/model.py"],
            "expect_absent": [],
        },
    ],
}
