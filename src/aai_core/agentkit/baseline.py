"""The committed baseline record — what every comparison compares against.

``evals/baseline.json`` is reviewed evidence: which run, on which dataset
version, at which scope, produced which metrics with which scorer and
prompt versions. ``compare`` refuses to score into a vacuum; when no
baseline exists it says so and offers ``--establish-baseline``, which
records that the current version IS the baseline.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, ValidationError, field_serializer, field_validator

from aai_core.agentkit.datasets import LoadedDataset
from aai_core.agentkit.errors import BaselineMissingError, ConfigError
from aai_core.contracts import ContractModel, freeze_value, thaw_value

_LEGACY_PLACEHOLDER = "unknown"


class BaselineScope(ContractModel):
    mode: Literal["full", "sample"]
    rows: int = Field(ge=0)
    seed: int | None = None


class BaselineDataset(ContractModel):
    ref: str = Field(min_length=1)
    digest: str = Field(min_length=1)
    rows: int = Field(ge=0)


class BaselineVersions(ContractModel):
    agent: str = Field(min_length=1)
    scorers: Mapping[str, int] = Field(default_factory=dict)
    judge_model: str | None = None
    judge_prompts: Mapping[str, str] = Field(default_factory=dict)
    aai_core: str = Field(min_length=1)

    @field_validator("scorers", "judge_prompts", mode="after")
    @classmethod
    def freeze_versions(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        return freeze_value(value)

    @field_serializer("scorers", "judge_prompts")
    def serialize_versions(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return thaw_value(value)


class BaselineRecord(ContractModel):
    schema_version: Literal[1]
    run_id: str | None
    experiment_id: str | None = None
    recorded_at: str = Field(min_length=1)
    dataset: BaselineDataset
    scope: BaselineScope
    metrics: Mapping[str, float]
    versions: BaselineVersions
    recorded_by: str = Field(min_length=1)
    change_id: str = Field(min_length=1)

    @field_validator("metrics", mode="before")
    @classmethod
    def coerce_metrics(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                key: float(item) if isinstance(item, int) else item
                for key, item in value.items()
                if not isinstance(item, bool)
            }
        return value

    @field_validator("metrics", mode="after")
    @classmethod
    def freeze_metrics(cls, value: Mapping[str, float]) -> Mapping[str, float]:
        return freeze_value(value)

    @field_serializer("metrics")
    def serialize_metrics(self, value: Mapping[str, float]) -> dict[str, float]:
        return thaw_value(value)


def load_baseline(path: Path) -> tuple[BaselineRecord | None, list[str]]:
    """Load the committed baseline; legacy files upgrade with a warning."""

    if not path.is_file():
        return None, []
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ConfigError(f"{path} is not valid JSON: {error}") from error
    if not isinstance(document, Mapping):
        raise ConfigError(f"{path} must contain a JSON object")
    if "schema_version" not in document:
        metrics = document.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ConfigError(f"{path} has neither schema_version nor metrics")
        record = BaselineRecord(
            schema_version=1,
            run_id=None,
            recorded_at=_LEGACY_PLACEHOLDER,
            dataset=BaselineDataset(
                ref=_LEGACY_PLACEHOLDER, digest=_LEGACY_PLACEHOLDER, rows=0
            ),
            scope=BaselineScope(mode="full", rows=0),
            metrics=dict(metrics),
            versions=BaselineVersions(
                agent=_LEGACY_PLACEHOLDER, aai_core=_LEGACY_PLACEHOLDER
            ),
            recorded_by="legacy-baseline-file",
            change_id=_LEGACY_PLACEHOLDER,
        )
        warning = (
            f"{path} is a legacy baseline without lineage; re-establish it "
            "with `agentkit compare --establish-baseline` to record dataset, "
            "scope, and scorer versions"
        )
        return record, [warning]
    try:
        return BaselineRecord(**document), []
    except ValidationError as error:
        raise ConfigError(f"{path} is not a valid baseline: {error}") from error


def write_baseline(path: Path, record: BaselineRecord) -> None:
    """Atomic, sorted, newline-terminated write (review-friendly diffs)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(record.model_dump(), indent=2, sort_keys=True, default=str) + "\n"
    scratch = path.with_suffix(path.suffix + ".tmp")
    scratch.write_text(text, encoding="utf-8")
    os.replace(scratch, path)


def select_baseline(
    *,
    baseline_path: Path,
    flag_run_id: str | None = None,
    config_run_id: str | None = None,
    mlflow_module: Any | None = None,
) -> tuple[BaselineRecord, list[str]]:
    """Deterministic, offline-first baseline selection.

    Precedence: an explicit ``--baseline-run`` flag, then
    ``baseline.run_id`` from the config (both fetched from MLflow), then
    the committed baseline file. Nothing found is an explicit refusal —
    never a silent score-in-a-vacuum.
    """

    for run_id, origin in (
        (flag_run_id, "--baseline-run"),
        (config_run_id, "baseline.run_id"),
    ):
        if run_id:
            return _baseline_from_run(run_id, origin, mlflow_module), []
    record, warnings = load_baseline(baseline_path)
    if record is not None:
        return record, warnings
    raise BaselineMissingError(
        f"No baseline exists for this project (looked for {baseline_path}).\n"
        "A comparison needs something to compare against. Run:\n"
        "    agentkit compare --establish-baseline\n"
        "to score the current version and record it as the baseline. That "
        "run IS the baseline - the next `agentkit compare` will score "
        "against it."
    )


def drift_warnings(
    record: BaselineRecord,
    *,
    dataset: LoadedDataset,
    mode: str,
    rows: int,
) -> list[str]:
    """Loud, explicit warnings when the comparison is not apples-to-apples."""

    warnings: list[str] = []
    if record.dataset.digest not in {dataset.digest, _LEGACY_PLACEHOLDER}:
        warnings.append(
            "baseline was recorded on a different dataset version (digest "
            f"{record.dataset.digest} vs {dataset.digest}); deltas are not "
            "apples-to-apples - consider re-establishing the baseline"
        )
    if record.scope.rows and (record.scope.mode != mode or record.scope.rows != rows):
        warnings.append(
            f"baseline scope was {record.scope.mode}/{record.scope.rows} "
            f"rows but this run scores {mode}/{rows} rows; deltas compare "
            "different row sets"
        )
    return warnings


def _baseline_from_run(
    run_id: str, origin: str, mlflow_module: Any | None
) -> BaselineRecord:
    mlflow = mlflow_module
    if mlflow is None:
        try:
            import mlflow  # type: ignore[no-redef]
        except ImportError as error:
            from aai_core.agentkit.errors import missing_extra

            raise missing_extra(
                f"Fetching the {origin} baseline run", "genai"
            ) from error
    try:
        run = mlflow.get_run(run_id)
    except Exception as error:
        raise BaselineMissingError(
            f"could not fetch baseline run {run_id!r} ({origin}): {error}",
            remediation="Check the run id and your MLflow authentication.",
        ) from error
    tags = dict(getattr(run.data, "tags", {}) or {})
    metrics = dict(getattr(run.data, "metrics", {}) or {})
    scorers: dict[str, int] = {}
    for pair in tags.get("aai.scorer_versions", "").split(","):
        name, _, version = pair.partition("=")
        if name and version.isdigit():
            scorers[name] = int(version)
    return BaselineRecord(
        schema_version=1,
        run_id=run_id,
        experiment_id=str(getattr(run.info, "experiment_id", "")) or None,
        recorded_at=tags.get("aai.recorded_at", _LEGACY_PLACEHOLDER),
        dataset=BaselineDataset(
            ref=tags.get("aai.dataset", _LEGACY_PLACEHOLDER),
            digest=tags.get("aai.dataset_digest", _LEGACY_PLACEHOLDER),
            rows=int(tags.get("aai.dataset_rows", "0") or 0),
        ),
        scope=BaselineScope(
            mode="full", rows=int(tags.get("aai.dataset_rows", "0") or 0)
        ),
        metrics=metrics,
        versions=BaselineVersions(
            agent=tags.get("aai.agent_target", _LEGACY_PLACEHOLDER),
            scorers=scorers,
            judge_model=tags.get("aai.judge_model"),
            aai_core=tags.get("aai.agentkit_version", _LEGACY_PLACEHOLDER),
        ),
        recorded_by=origin,
        change_id=tags.get("aai.change_id", _LEGACY_PLACEHOLDER),
    )
