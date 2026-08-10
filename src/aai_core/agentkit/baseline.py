"""The committed baseline record — what every comparison compares against.

``evals/baseline.json`` is reviewed evidence: which run, on which dataset
version, at which scope, produced which metrics with which scorer and
prompt versions. ``compare`` refuses to score into a vacuum; when no
baseline exists it says so and offers ``--establish-baseline``, which
records that the current version IS the baseline.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, ValidationError, field_serializer, field_validator

from aai_core.agentkit.datasets import LoadedDataset
from aai_core.agentkit.errors import (
    BaselineIncomparableError,
    BaselineMissingError,
    ConfigError,
)
from aai_core.contracts import ContractModel, freeze_value, thaw_value

_LEGACY_PLACEHOLDER = "unknown"
_REMOTE_BASELINE_TAGS = (
    "aai.run_purpose",
    "aai.agentkit_version",
    "aai.dataset",
    "aai.dataset_digest",
    "aai.dataset_rows",
    "aai.scope_mode",
    "aai.scope_rows",
    "aai.agent_target",
    "aai.recorded_at",
    "aai.change_id",
    "aai.scorer_versions",
    "aai.gate_passed",
    "aai.decision",
)


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
    # What the judge endpoint actually served, when it could be read. The
    # endpoint URI is a stable name for a mutable thing; this is the thing.
    judge_model_identity: str | None = None
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
        non_finite = sorted(
            name for name, item in value.items() if not math.isfinite(item)
        )
        if non_finite:
            raise ValueError(
                "metric values must be finite (invalid: " + ", ".join(non_finite) + ")"
            )
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
    text = (
        json.dumps(
            record.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
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
    judge_model_identity: str | None = None,
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
    # The endpoint name can stay put while the model behind it moves, so
    # the resolved identity is what actually says "same judge".
    was_identity = record.versions.judge_model_identity
    if was_identity and judge_model_identity and was_identity != judge_model_identity:
        failures.append(
            f"the judge endpoint now serves {judge_model_identity} but the "
            f"baseline was scored by {was_identity}, so it is not the same "
            "judge"
        )
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

    An empty prompt map is ambiguous for a legacy or judge-free baseline,
    but scorer versions disambiguate a prompt-backed judge that ran with
    bundled instructions. A registered prompt appearing later is drift for
    that scorer even though the old prompt map was empty.
    """

    recorded = dict(record.versions.judge_prompts)
    failures = [
        f"the {name} judge prompt moved ({recorded[name]} -> {current[name]}), "
        "so it is not the same judge that scored the baseline"
        for name in sorted(set(recorded) & set(current))
        if recorded[name] != current[name]
    ]
    if not recorded:
        # An empty prompt map is ambiguous only when the baseline says
        # nothing about which scorers ran. Once the scorer versions name a
        # prompt-backed judge, an empty entry means that judge used its
        # bundled instructions. If the same scorer resolves a registered
        # prompt now, its instructions changed even though the old map was
        # empty.
        current = {
            name: uri
            for name, uri in current.items()
            if name in record.versions.scorers
        }
        if not current:
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
    info = getattr(run, "info", None)
    returned_run_id = str(getattr(info, "run_id", "") or "")
    if returned_run_id != run_id:
        raise _invalid_remote_baseline(
            run_id,
            origin,
            f"the tracking store returned run {returned_run_id!r}",
        )
    status = str(getattr(info, "status", "") or "").upper()
    if status != "FINISHED":
        raise _invalid_remote_baseline(
            run_id,
            origin,
            f"its status is {status or 'unknown'}, not FINISHED",
        )
    tags = dict(getattr(run.data, "tags", {}) or {})
    run_metrics = dict(getattr(run.data, "metrics", {}) or {})
    missing = [
        name
        for name in _REMOTE_BASELINE_TAGS
        if not isinstance(tags.get(name), str) or not tags[name].strip()
    ]
    if missing:
        raise _invalid_remote_baseline(
            run_id,
            origin,
            "it lacks AgentKit baseline lineage tags: " + ", ".join(missing),
        )
    if tags["aai.run_purpose"] != "baseline":
        raise _invalid_remote_baseline(
            run_id,
            origin,
            f"aai.run_purpose is {tags['aai.run_purpose']!r}, not 'baseline'",
        )
    scope_mode = tags["aai.scope_mode"]
    if scope_mode not in {"full", "sample"}:
        raise _invalid_remote_baseline(
            run_id,
            origin,
            f"aai.scope_mode is {scope_mode!r}, not 'full' or 'sample'",
        )
    dataset_rows = _positive_tag_int(tags, "aai.dataset_rows", run_id, origin)
    scope_rows = _positive_tag_int(tags, "aai.scope_rows", run_id, origin)
    if not run_metrics:
        raise _invalid_remote_baseline(run_id, origin, "it carries no scored metrics")
    scorer_pairs = _strict_tag_pairs(
        tags["aai.scorer_versions"],
        name="aai.scorer_versions",
        run_id=run_id,
        origin=origin,
    )
    if any(not version.isdigit() or int(version) < 1 for _, version in scorer_pairs):
        raise _invalid_remote_baseline(
            run_id,
            origin,
            "aai.scorer_versions must assign a positive integer to every scorer",
        )
    scorers = {name: int(version) for name, version in scorer_pairs}
    # The run recorded which judge instructions it used; not reading them
    # back would leave every run-fetched baseline with an empty prompt map,
    # and a check that cannot fire is not a check.
    judge_prompts = dict(_tag_pairs(tags.get("aai.judge_prompt_versions")))
    evidence = _remote_results_evidence(mlflow, run_id, origin)
    expected_experiment_id = str(getattr(info, "experiment_id", "")) or None
    expected_versions = BaselineVersions(
        agent=tags["aai.agent_target"],
        scorers=scorers,
        judge_model=tags.get("aai.judge_model"),
        judge_model_identity=tags.get("aai.judge_model_identity"),
        judge_prompts=judge_prompts,
        aai_core=tags["aai.agentkit_version"],
    )
    expected_dataset = BaselineDataset(
        ref=tags["aai.dataset"],
        digest=tags["aai.dataset_digest"],
        rows=dataset_rows,
    )
    expected_scope = BaselineScope(mode=scope_mode, rows=scope_rows)
    mismatches: list[str] = []
    for name, actual, expected in (
        ("experiment_id", evidence.experiment_id, expected_experiment_id),
        ("recorded_at", evidence.recorded_at, tags["aai.recorded_at"]),
        ("agent", evidence.agent, tags["aai.agent_target"]),
        ("dataset", evidence.dataset, expected_dataset),
        ("scope", evidence.scope, expected_scope),
        ("versions", evidence.versions, expected_versions),
        ("change_id", evidence.change_id, tags["aai.change_id"]),
        ("gate_passed", str(evidence.gate_passed).lower(), tags["aai.gate_passed"]),
        ("decision", evidence.decision, tags["aai.decision"]),
    ):
        if actual != expected:
            mismatches.append(name)
    if not evidence.established_baseline:
        mismatches.append("established_baseline")
    if evidence.baseline_run_id is not None or evidence.baseline_metrics:
        mismatches.append("baseline_lineage")
    evidence_metrics = dict(evidence.metrics)
    if not evidence_metrics:
        mismatches.append("metrics")
    elif evidence_metrics != run_metrics:
        mismatches.append("metrics")
    if mismatches:
        raise _invalid_remote_baseline(
            run_id,
            origin,
            "canonical agentkit/results.json disagrees with the run's "
            "lineage: " + ", ".join(sorted(set(mismatches))),
        )
    return BaselineRecord(
        schema_version=1,
        run_id=run_id,
        experiment_id=expected_experiment_id,
        recorded_at=evidence.recorded_at,
        dataset=evidence.dataset,
        scope=evidence.scope,
        metrics=evidence_metrics,
        versions=evidence.versions,
        recorded_by=origin,
        change_id=evidence.change_id,
    )


def _invalid_remote_baseline(
    run_id: str, origin: str, reason: str
) -> BaselineIncomparableError:
    return BaselineIncomparableError(
        f"run {run_id!r} ({origin}) is not valid AgentKit baseline evidence: {reason}",
        remediation=(
            "Use the run id emitted by a completed "
            "`agentkit compare --establish-baseline` run. Committed legacy "
            "baseline files remain readable, but ungoverned remote runs do not."
        ),
    )


def _remote_results_evidence(mlflow: Any, run_id: str, origin: str) -> Any:
    # Local import avoids the module cycle: ResultsRecord itself uses the
    # baseline dataset/scope/version contracts defined above.
    from aai_core.agentkit.results import fetch_results

    try:
        return fetch_results(run_id, mlflow_module=mlflow)
    except ConfigError as error:
        raise _invalid_remote_baseline(
            run_id,
            origin,
            f"its canonical agentkit/results.json is unavailable or invalid ({error})",
        ) from error


def _positive_tag_int(
    tags: Mapping[str, Any], name: str, run_id: str, origin: str
) -> int:
    try:
        value = int(tags[name])
    except (TypeError, ValueError) as error:
        raise _invalid_remote_baseline(
            run_id, origin, f"{name} is not an integer"
        ) from error
    if value < 1:
        raise _invalid_remote_baseline(run_id, origin, f"{name} must be at least 1")
    return value


def _strict_tag_pairs(
    value: str, *, name: str, run_id: str, origin: str
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for item in value.split(","):
        key, separator, recorded = item.partition("=")
        if not separator or not key.strip() or not recorded.strip():
            raise _invalid_remote_baseline(
                run_id, origin, f"{name} contains a malformed entry {item!r}"
            )
        pairs.append((key.strip(), recorded.strip()))
    return pairs


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
