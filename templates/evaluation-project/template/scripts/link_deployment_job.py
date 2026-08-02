"""Link a Unity Catalog registered model to its deployment job.

The Databricks Asset Bundle schema has no field for this: a registered
model resource cannot name its deployment job, so the link is made once
through the MLflow client after the bundle is deployed. Run this from a
machine authenticated to the workspace:

    python scripts/link_deployment_job.py --model <catalog>.<schema>.<name> \\
        --job-id <deployment job id>

It is idempotent — re-running it reports the existing link and changes
nothing. It never creates the model, the job, or any permission; those are
provisioned through the platform process.

The `--await-approval` and `--report-deployment` modes are used by the
deployment job's own tasks (see resources/optional/deployment_job.yml).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


def link(model_name: str, job_id: str) -> int:
    from mlflow import MlflowClient

    client = MlflowClient(registry_uri="databricks-uc")
    model = client.get_registered_model(model_name)
    current = getattr(model, "deployment_job_id", None)
    if current == job_id:
        print(f"{model_name} is already linked to deployment job {job_id}")
        return 0
    if current:
        print(f"{model_name} is linked to deployment job {current}; relinking")
    client.update_registered_model(name=model_name, deployment_job_id=job_id)
    print(f"linked {model_name} -> deployment job {job_id}")
    return 0


def await_approval() -> int:
    """The approval gate task.

    This task exists to be approved. Databricks records approval as a Unity
    Catalog tag on the model version, so the FIRST run of a deployment job
    always fails here: no tag exists yet. Approve the run in the Databricks
    UI — that writes the tag and the job resumes automatically.
    """

    print(
        "Approval required. Approve this run in the Databricks UI to record "
        "the approval tag on the model version. The first run of a new "
        "model version always stops here by design."
    )
    print(
        "Promotion evidence for this version: run `agentkit evidence` and "
        "attach the generated evidence.md to the approval."
    )
    return 1


def report_deployment() -> int:
    print(
        "Approved. Serving-endpoint updates are performed by the platform "
        "deployment process; this task records that the gate was passed."
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="<catalog>.<schema>.<model>")
    parser.add_argument("--job-id", help="the deployment job id")
    parser.add_argument("--await-approval", action="store_true")
    parser.add_argument("--report-deployment", action="store_true")
    arguments = parser.parse_args(argv)

    if arguments.await_approval:
        return await_approval()
    if arguments.report_deployment:
        return report_deployment()
    if not arguments.model or not arguments.job_id:
        parser.error("--model and --job-id are required to link a model")
    return link(arguments.model, arguments.job_id)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
