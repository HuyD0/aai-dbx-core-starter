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
    "agentic-rag": [
        {
            "name": "dbx-azure-search",
            "overrides": {
                "project_name": "test-agent",
                "model_provider": "databricks",
                "model_deployment": "chat",
                "retrieval_provider": "azure_ai_search",
                "search_endpoint": "https://search.search.windows.net",
                "search_index": "knowledge",
                "embedding_deployment": "embedding",
            },
            "expect_present": [
                "src/app/agent.py",
                "evals/evaluate.py",
                "scripts/promote_prompt.py",
            ],
            "expect_absent": [],
        },
    ],
}
