"""Notebook-source recipe for trace configuration and sampled scorers.

Production scorer registration must be run as notebook source because MLflow's
monitoring service serializes scorer code from that notebook.  Keep
``EXECUTION`` at its default for previews.  A reviewed notebook may explicitly
switch it to ``execute``, set ``notebook_confirmed=True``, and supply the
acknowledgement environment variable documented in this directory's README.
"""

from __future__ import annotations

from pathlib import Path

from email_support_agent.mlops import (
    ExecutionPolicy,
    OperationReceipt,
    TraceDestinationPlan,
    apply_trace_destination,
    monitoring_plan_from_agentkit,
    register_and_start_monitoring,
)

from aai_core.tags import ResourceContext

EXECUTION = ExecutionPolicy()


def configure(
    *,
    context: ResourceContext | None = None,
    execution: ExecutionPolicy = EXECUTION,
    project_root: str | Path | None = None,
) -> tuple[OperationReceipt, OperationReceipt]:
    root = (
        Path(project_root)
        if project_root is not None
        else Path(__file__).resolve().parents[2]
    )
    trace_receipt = apply_trace_destination(
        TraceDestinationPlan(),
        context=context,
        execution=execution,
    )
    scorer_receipt = register_and_start_monitoring(
        monitoring_plan_from_agentkit(root),
        execution=execution,
    )
    return trace_receipt, scorer_receipt


if __name__ == "__main__":
    for receipt in configure():
        print(receipt.model_dump_json(indent=2))
