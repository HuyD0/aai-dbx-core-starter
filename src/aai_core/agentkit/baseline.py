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


def comparability_failures(
    record: BaselineRecord,
    *,
    dataset: LoadedDataset,
    mode: str,
    rows: int,
    scorers: Mapping[str, int] | None = None,
    judge_model: str | None = None,
    judge_prompts: Mapping[str, str] | None = None,
    judges_enabled: bool = True,
) -> list[str]:
    """Why this baseline cannot be the reference for this run.

    A delta is only evidence when both sides measured the same rows with
    the same scorers. Change the dataset, the scope, the scorer set, a
    scorer version, the judge model, or a judge prompt, and the number
    still subtracts cleanly while meaning nothing — which is worse than no
    comparison, because it looks like one.

    Everything here is checkable before the first judge call. The prompt
    map arrives on a second pass, because resolving it needs MLflow, but
    that still happens before the run opens.
    """

    failures: list[str] = []
    # A sample carries the digest of the dataset it was drawn from, and a
    # baseline recorded on that whole dataset is a baseline for the same
    # questions. Only the scope differs — which the next check reports, and
    # reports truthfully. Rejecting the sample as "changed data" would be a
    # refusal for a reason that is not the case.
    same_data = {dataset.digest, dataset.sampled_from, _LEGACY_PLACEHOLDER}
    if record.dataset.digest not in same_data:
        failures.append(
            "the dataset changed since the baseline was recorded (digest "
            f"{record.dataset.digest} -> {dataset.digest})"
        )
    if record.scope.rows and (record.scope.mode != mode or record.scope.rows != rows):
        failures.append(
            f"the baseline scored {record.scope.mode}/{record.scope.rows} "
            f"rows but this run scores {mode}/{rows}"
        )
    current = dict(scorers or {})
    for name, version in sorted(current.items()):
        recorded = record.versions.scorers.get(name)
        if recorded is not None and recorded != version:
            failures.append(
                f"scorer {name} is v{version} but the baseline used "
                f"v{recorded}, so the two scores do not mean the same thing"
            )
    if scorers is not None:
        # A caller that names no scorers has said nothing about membership;
        # only a caller that supplies the plan can have one compared.
        failures.extend(_membership_failures(record, current, judges_enabled))
    recorded_judge = record.versions.judge_model
    if judge_model and recorded_judge and recorded_judge != judge_model:
        failures.append(f"the judge model changed ({recorded_judge} -> {judge_model})")
    if judge_prompts is not None:
        failures.extend(_prompt_failures(record, dict(judge_prompts)))
    return failures


def _prompt_failures(record: BaselineRecord, current: Mapping[str, str]) -> list[str]:
    """How this run's judge instructions differ from the baseline's.

    Both directions count. A registered prompt whose alias is deleted stops
    resolving, so the scorer quietly falls back to its bundled
    instructions — the judge changed, and comparing only the names this run
    resolved would never look at it. The reverse, a judge that gained a
    registered prompt since the baseline, is the same change in the other
    direction.

    A baseline that recorded no prompts at all says nothing about
    membership — legacy records and judge-free runs both look like that —
    so only shared names are version-compared there.
    """

    recorded = dict(record.versions.judge_prompts)
    failures = [
        f"the {name} judge prompt moved ({recorded[name]} -> {current[name]}), "
        "so it is not the same judge that scored the baseline"
        for name in sorted(set(recorded) & set(current))
        if recorded[name] != current[name]
    ]
    if not recorded:
        return failures
    for name in sorted(set(recorded) - set(current)):
        failures.append(
            f"the {name} judge prompt {recorded[name]} no longer resolves, so "
            "that judge now scores with its bundled instructions instead of "
            "the ones the baseline used"
        )
    for name in sorted(set(current) - set(recorded)):
        failures.append(
            f"the {name} judge prompt is now {current[name]}, but the "
            "baseline scored with that judge's bundled instructions"
        )
    return failures


def _membership_failures(
    record: BaselineRecord,
    current: Mapping[str, int],
    judges_enabled: bool,
) -> list[str]:
    """Scorers one side ran and the other did not.

    Comparing versions alone misses the bigger change: removing a scorer
    also removes its registry-default threshold from the policy, so the
    comparison passes without that evidence and nothing says so.

    A judge-free run is not a mismatch, though. ``smoke`` runs code
    scorers only by design, so a baseline's judge scorers are ignored when
    judges are off — a scorer missing because of the *mode* is different
    from one removed by *configuration*, and only the latter is a control
    being weakened.
    """

    from aai_core.agentkit.catalog import get_spec
    from aai_core.agentkit.errors import UnknownScorerError

    recorded = dict(record.versions.scorers)
    if not recorded:
        # A legacy baseline records no scorers; nothing to compare against.
        return []

    def _judged(name: str) -> bool:
        try:
            return get_spec(name).judge is not None
        except UnknownScorerError:
            return False

    expected = {name for name in recorded if judges_enabled or not _judged(name)}
    failures = []
    missing = sorted(expected - set(current))
    added = sorted(set(current) - set(recorded))
    if missing:
        failures.append(
            f"the baseline scored {', '.join(missing)} but this run does "
            "not, so the comparison would be missing that evidence"
        )
    if added:
        failures.append(
            f"this run scores {', '.join(added)} but the baseline never "
            "did, so there is nothing to compare it against"
        )
    return failures


def drift_warnings(
    record: BaselineRecord,
    *,
    dataset: LoadedDataset,
    mode: str,
    rows: int,
) -> list[str]:
    """Non-blocking differences worth naming in the run's warnings."""

    warnings: list[str] = []
    if record.dataset.digest == _LEGACY_PLACEHOLDER:
        warnings.append(
            "the baseline predates dataset digests, so it cannot be checked "
            "against this dataset; re-establish it to restore the check"
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
    scorers = {
        name: int(version)
        for name, version in _tag_pairs(tags.get("aai.scorer_versions"))
        if version.isdigit()
    }
    # The run recorded which judge instructions it used; not reading them
    # back would leave every run-fetched baseline with an empty prompt map,
    # and a check that cannot fire is not a check.
    judge_prompts = dict(_tag_pairs(tags.get("aai.judge_prompt_versions")))
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
        # The run records its own scope. Assuming "full" would make a
        # sampled baseline fetched by run id incomparable with the very
        # sampled run that produced it.
        scope=BaselineScope(
            mode="sample" if tags.get("aai.scope_mode") == "sample" else "full",
            rows=int(tags.get("aai.scope_rows") or tags.get("aai.dataset_rows") or 0),
        ),
        metrics=metrics,
        versions=BaselineVersions(
            agent=tags.get("aai.agent_target", _LEGACY_PLACEHOLDER),
            scorers=scorers,
            judge_model=tags.get("aai.judge_model"),
            judge_prompts=judge_prompts,
            aai_core=tags.get("aai.agentkit_version", _LEGACY_PLACEHOLDER),
        ),
        recorded_by=origin,
        change_id=tags.get("aai.change_id", _LEGACY_PLACEHOLDER),
    )


def _tag_pairs(value: str | None) -> list[tuple[str, str]]:
    """``"a=1,b=2"`` as pairs — the shape both version tags are written in.

    A prompt URI (``prompts:/name/3``) contains no ``=``, so partitioning on
    the first one keeps the whole value.
    """

    pairs = []
    for item in (value or "").split(","):
        name, _, recorded = item.partition("=")
        if name.strip() and recorded.strip():
            pairs.append((name.strip(), recorded.strip()))
    return pairs
