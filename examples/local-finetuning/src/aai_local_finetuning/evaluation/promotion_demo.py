"""Deliberately synthetic promotion-gate demo evidence for the course.

Notebook 08 demonstrates the promotion gates *catching* a failing change by
degrading one report in memory and re-running :func:`decide_lora_promotion`.
``decide_lora_promotion`` verifies every report against the live source and
runtime contract, the verified base model, and validated training lineage, so
committed fixtures cannot pass those gates as-is: they carry placeholder
identity values that must be rebound to the live machine's boundary at run
time.  These helpers perform that loading and rebinding.

The fixture reports are invented classroom numbers.  They are never written
back to disk, never logged as evidence, and every place they appear labels
them as fixtures.  Nothing here may be used to fabricate promotion evidence.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from ..training import BaseModelExecutionContract
from .models import EvaluationReport, LocalMLXInferenceConfig

PROMOTION_DEMO_METHODS = (
    "majority",
    "keyword-rule",
    "basic",
    "strong",
    "few_shot",
    "lora-change",
)


def load_promotion_demo_reports(
    fixture_dir: str | Path,
) -> dict[str, EvaluationReport]:
    """Load the six committed synthetic method reports through the strict schema."""

    directory = Path(fixture_dir)
    reports: dict[str, EvaluationReport] = {}
    for method in PROMOTION_DEMO_METHODS:
        path = directory / f"{method}-report.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing promotion-demo fixture report: {path}")
        reports[method] = EvaluationReport.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    return reports


def bind_promotion_demo_reports(
    reports: Mapping[str, EvaluationReport],
    *,
    evaluation_execution_contract_sha256: str,
    base_model: BaseModelExecutionContract,
    training_manifest_sha256: str,
    training_execution_contract_sha256: str,
) -> dict[str, EvaluationReport]:
    """Rebind fixture reports to the live machine's identity boundary.

    The caller supplies the live evaluation session's execution-contract hash
    and verified base-model contract plus a genuinely validated training
    snapshot's manifest hashes (the flight-preparation preflight adapter works
    on any prepared machine).  Only those identity fields are replaced; the
    synthetic metric values stay untouched, so the demo exercises the real
    gates without inventing lineage on disk.
    """

    missing = [method for method in PROMOTION_DEMO_METHODS if method not in reports]
    if missing:
        raise ValueError(
            "promotion-demo reports are incomplete; missing: " + ", ".join(missing)
        )
    bound: dict[str, EvaluationReport] = {}
    for method, report in reports.items():
        update: dict[str, object] = {
            "evaluation_execution_contract_sha256": (
                evaluation_execution_contract_sha256
            ),
        }
        inference = report.inference_config
        if isinstance(inference, LocalMLXInferenceConfig):
            inference_update: dict[str, object] = {"base_model": base_model}
            if method == "lora-change":
                inference_update["adapter_manifest_sha256"] = training_manifest_sha256
            update["inference_config"] = inference.model_copy(update=inference_update)
        if method == "lora-change":
            update["training_manifest_sha256"] = training_manifest_sha256
            update["training_execution_contract_sha256"] = (
                training_execution_contract_sha256
            )
        bound[method] = report.model_copy(update=update)
    return bound


def degrade_schema_validity(
    report: EvaluationReport,
    *,
    rate: float,
) -> EvaluationReport:
    """Return an in-memory copy whose schema-validity rate is replaced.

    The original report object is frozen and stays untouched; nothing is
    persisted.  This exists so a learner can watch the absolute schema gate
    produce a named rejection from otherwise-passing evidence.
    """

    if not 0.0 <= rate <= 1.0:
        raise ValueError("rate must be between 0.0 and 1.0")
    return report.model_copy(
        update={
            "output_quality": report.output_quality.model_copy(
                update={"json_schema_validity_rate": rate}
            ),
        }
    )
