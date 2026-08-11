"""Render the Email Support Agent's MLflow/AIMLOps plan without cloud calls.

Run from the accelerator root with ``PYTHONPATH=../../src:src``.  This file is
safe in CI: every connected executor remains in dry-run mode.
"""

from __future__ import annotations

import json
from pathlib import Path

from email_support_agent.mlops import (
    GepaOptimizationPlan,
    JudgeAlignmentPlan,
    RegisteredScorerRef,
    TraceCurationPlan,
    TraceDestinationPlan,
    estimate_agentkit_judge_budget,
    estimate_alignment_budget,
    estimate_gepa_budget,
    monitoring_plan_from_agentkit,
)


def build_plan(project_root: str | Path | None = None) -> dict[str, object]:
    root = (
        Path(project_root)
        if project_root is not None
        else Path(__file__).resolve().parents[2]
    )
    aligned_guidelines = RegisteredScorerRef(
        shared_name="guidelines",
        version_env="AAI_ALIGNED_GUIDELINES_JUDGE_VERSION",
    )
    alignment = JudgeAlignmentPlan(
        judge=aligned_guidelines,
        label_schema_name="guidelines",
    )
    optimization = GepaOptimizationPlan(scorers=(aligned_guidelines,))
    monitoring = monitoring_plan_from_agentkit(root)
    return {
        "trace_destination": TraceDestinationPlan().model_dump(mode="json"),
        "production_monitoring": monitoring.model_dump(mode="json"),
        "trace_curation": TraceCurationPlan().model_dump(mode="json"),
        "judge_alignment": alignment.model_dump(mode="json"),
        "judge_alignment_budget_at_cap": estimate_alignment_budget(
            alignment,
            labeled_trace_count=alignment.maximum_alignment_traces,
        ).model_dump(mode="json"),
        "gepa_proposal": optimization.model_dump(mode="json"),
        "gepa_budget": estimate_gepa_budget(optimization).model_dump(mode="json"),
        "release_evaluation_budget": estimate_agentkit_judge_budget(root).model_dump(
            mode="json"
        ),
        "mutation_default": "dry_run",
        "automatic_promotion": False,
    }


if __name__ == "__main__":
    print(json.dumps(build_plan(), indent=2, sort_keys=True))
