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
from app.config import AAI_CORE_WHEEL
from app.tools import UC_FUNCTION_TOOLS

ROOT = Path(__file__).resolve().parents[1]


def serving_pip_requirements() -> list[str]:
    """The serving environment installs exactly what runtime jobs use: the
    pinned volume wheel for aai-core plus the locked runtime versions."""

    locked = [
        line.strip()
        for line in (ROOT / "requirements.lock").read_text("utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    return [AAI_CORE_WHEEL, *locked]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-name",
        default=None,
        help="UC model name; defaults to <catalog>.<schema>.<application>.",
    )
    parser.add_argument(
        "--register-only",
        action="store_true",
        help="Log and register the agent as code in Unity Catalog without "
        "deploying a serving endpoint (the governed record of the release).",
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
            # Everything the model needs away from this checkout: the app
            # package, the platform configuration (references only), and the
            # pinned dependencies incl. the aai-core volume wheel.
            code_paths=[str(ROOT / "src" / "app")],
            model_config=str(ROOT / "aai-platform.yml"),
            pip_requirements=serving_pip_requirements(),
            resources=agent_resources(
                settings,
                models=["general-chat"],
                uc_functions=list(UC_FUNCTION_TOOLS),
            ),
            registered_model_name=uc_model_name,
        )
    if args.register_only:
        print(
            {
                "model": uc_model_name,
                "version": logged.registered_model_version,
                "deployed": False,
            }
        )
        return
    deployment = deploy_agent(
        context,
        uc_model_name=uc_model_name,
        version=logged.registered_model_version,
    )
    print({"model": uc_model_name, "deployment": str(deployment)})


if __name__ == "__main__":
    main()
