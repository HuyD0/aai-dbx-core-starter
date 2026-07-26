"""Gated serving deployment (credentialed path, AFTER the release gate).

Logs the models-from-code agent with its resource declarations, registers it
in Unity Catalog, and deploys via the SDK's serving adapter. Run only after
`python evals/evaluate.py` passed for this exact code + prompt version.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from aai_core import bootstrap
from aai_core.serving import agent_resources, deploy_agent
from app.tools import UC_FUNCTION_TOOLS

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-name",
        default=None,
        help="UC model name; defaults to <catalog>.<schema>.<application>.",
    )
    args = parser.parse_args()

    import mlflow

    context = bootstrap(ROOT / "aai-platform.yml")
    settings = context.settings
    uc_model_name = args.model_name or (
        f"{settings.catalog}.{settings.schema}."
        f"{context.tags.application.replace('-', '_')}"
    )

    mlflow.set_experiment(settings.experiment_name)
    mlflow.set_registry_uri("databricks-uc")
    with mlflow.start_run(run_name="agent-deploy"):
        logged = mlflow.pyfunc.log_model(
            name="agent",
            python_model=str(ROOT / "serving" / "model.py"),
            resources=agent_resources(
                settings,
                models=["general-chat"],
                uc_functions=list(UC_FUNCTION_TOOLS),
            ),
            registered_model_name=uc_model_name,
        )
    deployment = deploy_agent(
        context,
        uc_model_name=uc_model_name,
        version=logged.registered_model_version,
    )
    print({"model": uc_model_name, "deployment": str(deployment)})


if __name__ == "__main__":
    main()
