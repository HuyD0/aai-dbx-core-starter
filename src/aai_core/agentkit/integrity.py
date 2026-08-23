"""Per-run judge-integrity checks: self-consistency and frozen anchors.

The gate compares this run's judged metrics against a baseline's, and both
sides assume the judge is a fixed instrument. It is not: the endpoint behind
a stable judge name can be repointed, and even a pinned judge is stochastic.
This module measures the instrument inside the run so the gate can tell
"the agent regressed" apart from "the judge moved":

- **Self-consistency** re-scores a small deterministic sample of this run's
  own outputs with the same judges. The judge's disagreement with itself is
  an upper bound on how much of any observed delta is signal.
- **Anchor drift** re-scores a frozen set of baseline outputs whose judge
  scores were recorded when the baseline was established. The agent is not
  in the loop at all, so movement here is the judge changing.

Both are a few extra judge calls, not a second evaluation: the outputs
already exist, so only the judge half runs again. "Anchors", not "canary" —
this repository's dependency canary (AGENTS.md section 8) is an unrelated
mechanism, and the name says what these rows do: they pin the judge.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import random
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Literal, cast

from pydantic import Field, ValidationError, field_serializer, field_validator

from aai_core.agentkit._values import numeric_score
from aai_core.agentkit.errors import ConfigError
from aai_core.contracts import ContractModel, freeze_value, thaw_value
from aai_core.evaluation import MetricDirection, MetricRule

_INTEGRITY_SEGMENT = "/integrity/"
SELF_INCONSISTENCY_METRIC = "judge/integrity/self_inconsistency"
ANCHOR_DRIFT_METRIC = "judge/integrity/anchor_drift"
RESCORE_FAILURES_METRIC = "judge/integrity/rescore_failures"
# Deterministic like SMOKE_SEED: the sampled rows must be the same rows on
# every run, or the flip rate would measure sampling noise as well as the
# judge.
INTEGRITY_SEED = 20260819
DEFAULT_ANCHOR_ROWS = 12

ANCHOR_DRIFT_EXPLANATION = (
    "the judges scored the frozen anchor outputs differently than when the "
    "anchors were recorded - the judge changed, not the agent. This is not "
    "an agent regression: verify the judge endpoint and prompt pins, or, "
    "after a deliberate judge release, re-establish the baseline and "
    "anchors with `agentkit compare --establish-baseline`."
)


class IntegrityConfig(ContractModel):
    """Project-owned policy for measuring the judge inside a run.

    ``consistency_sample`` rows of this run are re-judged every judged run;
    0 (the default) turns the check and its gate rule off. Anchors are
    report-only until ``require_anchors`` is set — flip it in the same
    change that commits the first frozen anchors. ``require_calibration``
    additionally demands a passing calibration record for every judge
    scorer in use (see ``aai_core.agentkit.calibration``).
    """

    consistency_sample: int = Field(default=0, ge=0, le=200)
    max_self_inconsistency: float = Field(default=0.2, ge=0.0, le=1.0)
    anchors: str = Field(default="evals/judge_anchors.json", min_length=1)
    max_anchor_drift: float = Field(default=0.1, ge=0.0, le=1.0)
    require_anchors: bool = False
    require_calibration: bool = False
    calibration_dir: str = Field(default="evals/judges", min_length=1)


class JudgeConsistencyEvidence(ContractModel):
    """What re-judging a sample of this run's own outputs measured."""

    sample_size: int = Field(ge=0)
    seed: int
    flip_rates: Mapping[str, float] = Field(default_factory=dict)
    overall: float = Field(ge=0.0)
    rescore_failures: int = Field(default=0, ge=0)

    @field_validator("flip_rates", mode="before")
    @classmethod
    def coerce_rates(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                key: float(item) if isinstance(item, int) else item
                for key, item in value.items()
                if not isinstance(item, bool)
            }
        return value

    @field_validator("flip_rates", mode="after")
    @classmethod
    def freeze_rates(cls, value: Mapping[str, float]) -> Mapping[str, float]:
        return cast(Mapping[str, float], freeze_value(value))

    @field_serializer("flip_rates")
    def serialize_rates(self, value: Mapping[str, float]) -> dict[str, float]:
        return cast(dict[str, float], thaw_value(value))


class AnchorDriftEvidence(ContractModel):
    """What re-judging the frozen anchor outputs measured."""

    anchors_ref: str = Field(min_length=1)
    anchors_digest: str = Field(min_length=1)
    rows: int = Field(ge=0)
    drift_by_scorer: Mapping[str, float] = Field(default_factory=dict)
    overall: float = Field(ge=0.0)
    rescore_failures: int = Field(default=0, ge=0)

    @field_validator("drift_by_scorer", mode="before")
    @classmethod
    def coerce_drift(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                key: float(item) if isinstance(item, int) else item
                for key, item in value.items()
                if not isinstance(item, bool)
            }
        return value

    @field_validator("drift_by_scorer", mode="after")
    @classmethod
    def freeze_drift(cls, value: Mapping[str, float]) -> Mapping[str, float]:
        return cast(Mapping[str, float], freeze_value(value))

    @field_serializer("drift_by_scorer")
    def serialize_drift(self, value: Mapping[str, float]) -> dict[str, float]:
        return cast(dict[str, float], thaw_value(value))


class IntegrityEvidence(ContractModel):
    schema_version: Literal[1] = 1
    consistency: JudgeConsistencyEvidence | None = None
    anchor_drift: AnchorDriftEvidence | None = None


class AnchorRow(ContractModel):
    """One frozen row: the question, the frozen answer, the frozen scores."""

    inputs: Mapping[str, Any] = Field(default_factory=dict)
    outputs: Any = None
    expectations: Mapping[str, Any] = Field(default_factory=dict)
    # Recorded judge scores by registry scorer NAME (not metric key), from
    # the run that froze this row.
    scores: Mapping[str, float] = Field(default_factory=dict)

    @field_validator("scores", mode="before")
    @classmethod
    def coerce_scores(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                key: float(item) if isinstance(item, int) else item
                for key, item in value.items()
                if not isinstance(item, bool)
            }
        return value

    @field_validator("inputs", "expectations", "scores", "outputs", mode="after")
    @classmethod
    def freeze_fields(cls, value: Any) -> Any:
        return freeze_value(value)

    @field_serializer("inputs", "expectations", "scores", "outputs")
    def serialize_fields(self, value: Any) -> Any:
        return thaw_value(value)


class JudgeAnchors(ContractModel):
    """The committed frozen-anchor file, digest-bound against edits."""

    schema_version: Literal[1] = 1
    recorded_at: str = Field(min_length=1)
    recorded_by: str = Field(min_length=1)
    change_id: str = Field(min_length=1)
    judge_model: str | None = None
    judge_model_identity: str | None = None
    judge_prompts: Mapping[str, str] = Field(default_factory=dict)
    scorer_versions: Mapping[str, int] = Field(default_factory=dict)
    rows: tuple[AnchorRow, ...] = ()
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("rows", mode="before")
    @classmethod
    def coerce_rows(cls, value: Any) -> Any:
        if isinstance(value, list):
            return tuple(
                AnchorRow(**item) if isinstance(item, Mapping) else item
                for item in value
            )
        return value

    @field_validator("judge_prompts", "scorer_versions", mode="after")
    @classmethod
    def freeze_pins(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        return cast(Mapping[str, Any], freeze_value(value))

    @field_serializer("judge_prompts", "scorer_versions")
    def serialize_pins(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return cast(dict[str, Any], thaw_value(value))


@dataclass(frozen=True)
class RowJudge:
    """A row-level judge as the runner built it, ready to re-invoke."""

    name: str
    metric: str
    scorer: Any


def integrity_metric(metric: str, component: str) -> str:
    """The stable synthetic metric name used by the gate policy."""

    return f"{metric}{_INTEGRITY_SEGMENT}{component}"


def is_integrity_metric(metric: str) -> bool:
    return _INTEGRITY_SEGMENT in metric


def is_row_level_judge(spec: Any) -> bool:
    """Whether a catalog spec is a judge this module can re-invoke.

    Duck-typed on purpose: importing the catalog here would close an import
    cycle (config imports this module; the catalog reads the config). v1
    re-invokes only row-level judges that need no trace — trace-fanout
    judges would need the trace object and per-span fan-out to re-score.
    """

    if getattr(spec, "judge", None) is None:
        return False
    fanout = getattr(getattr(spec, "fanout", None), "value", None)
    if fanout not in (None, "row"):
        return False
    trace = getattr(getattr(spec, "needs_trace", None), "value", None)
    return trace in (None, "none")


def extend_rules_with_integrity(
    rules: Sequence[MetricRule],
    config: IntegrityConfig,
    *,
    judges_enabled: bool,
) -> tuple[MetricRule, ...]:
    """Add the judge-integrity rules without changing the original ones.

    Deliberately a function of configuration alone (plus whether judges ran):
    the anchor rule must not flap with the anchor file's presence, or a
    deleted file would silently relax the gate instead of failing it closed
    through the missing metric.
    """

    base = tuple(rule for rule in rules if not is_integrity_metric(rule.metric))
    if not judges_enabled:
        return base
    augmented = list(base)
    if config.consistency_sample > 0:
        augmented.append(
            MetricRule(
                metric=SELF_INCONSISTENCY_METRIC,
                direction=MetricDirection.LOWER,
                required=config.max_self_inconsistency,
            )
        )
    if config.require_anchors:
        augmented.append(
            MetricRule(
                metric=ANCHOR_DRIFT_METRIC,
                direction=MetricDirection.LOWER,
                required=config.max_anchor_drift,
            )
        )
    return tuple(augmented)


def estimate_integrity_calls(
    config: IntegrityConfig,
    *,
    row_judges: int,
    dataset_rows: int,
    anchor_rows: int,
) -> int:
    """Extra judge calls the integrity checks will add to this run."""

    if row_judges == 0:
        return 0
    sampled = min(config.consistency_sample, dataset_rows)
    return row_judges * (sampled + anchor_rows)


def invoke_judge(judge: RowJudge, row: Mapping[str, Any], outputs: Any) -> float | None:
    """Re-invoke one judge on one already-answered row.

    Mirrors how ``Scorer.run`` dispatches: only the keyword arguments the
    scorer's ``__call__`` accepts are passed. Returns the numeric verdict,
    or ``None`` when the judge declined the row; raises when the judge
    invocation itself failed.
    """

    provided = {
        "inputs": dict(row.get("inputs") or {}),
        "outputs": outputs,
        "expectations": dict(row.get("expectations") or {}),
    }
    try:
        parameters = inspect.signature(judge.scorer.__call__).parameters
    except (TypeError, ValueError):
        parameters = None
    if parameters is None or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        kwargs = provided
    else:
        kwargs = {key: value for key, value in provided.items() if key in parameters}
    return numeric_feedback(judge.scorer(**kwargs))


def numeric_feedback(value: Any) -> float | None:
    """The numeric verdict inside a native scorer return value.

    Judges return a Feedback (or a list of them); code paths may return a
    bare bool, number, or yes/no string. All collapse through the same
    mapping the per-row samples use, so a re-score is compared in the same
    units as the original.
    """

    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    unwrapped = getattr(value, "value", value)
    return numeric_score(unwrapped)


def eligible_row_indices(
    rows: Sequence[Mapping[str, Any]],
    outputs_by_row: Sequence[Any],
    metric_samples: Mapping[str, Sequence[float | None]],
    judges: Sequence[RowJudge],
) -> list[int]:
    """Rows that can be re-judged: an output exists and a judge scored it."""

    eligible = []
    for index in range(min(len(rows), len(outputs_by_row))):
        if outputs_by_row[index] is None:
            continue
        for judge in judges:
            samples = metric_samples.get(judge.metric) or ()
            if index < len(samples) and samples[index] is not None:
                eligible.append(index)
                break
    return eligible


def sample_indices(eligible: Sequence[int], n: int, *, seed: int) -> list[int]:
    if n >= len(eligible):
        return list(eligible)
    shuffled = list(eligible)
    random.Random(seed).shuffle(shuffled)
    return sorted(shuffled[:n])


def run_integrity_checks(
    *,
    config: IntegrityConfig,
    rows: Sequence[Mapping[str, Any]],
    outputs_by_row: Sequence[Any],
    metric_samples: Mapping[str, Sequence[float | None]],
    judges: Sequence[RowJudge],
    anchors: JudgeAnchors | None,
) -> tuple[IntegrityEvidence | None, dict[str, float], list[str]]:
    """Run the configured checks; return (evidence, metrics, warnings).

    Fail-open is not an option here: when a check is configured but cannot
    produce its metric, the metric stays absent and the corresponding gate
    rule fails closed with "metric is missing". The warnings say why.
    """

    warnings: list[str] = []
    if not judges:
        if config.consistency_sample > 0 or config.require_anchors or anchors:
            warnings.append(
                "judge integrity checks were configured but no row-level "
                "judge ran, so nothing was re-scored"
            )
        return None, {}, warnings

    consistency = None
    anchor_drift = None
    metrics: dict[str, float] = {}
    rescore_failures = 0

    if config.consistency_sample > 0:
        consistency, consistency_metrics, consistency_warnings = _consistency_check(
            config,
            rows=rows,
            outputs_by_row=outputs_by_row,
            metric_samples=metric_samples,
            judges=judges,
        )
        metrics.update(consistency_metrics)
        warnings.extend(consistency_warnings)
        if consistency is not None:
            rescore_failures += consistency.rescore_failures

    if anchors is not None:
        anchor_drift, anchor_metrics, anchor_warnings = _anchor_check(
            config, anchors=anchors, judges=judges
        )
        metrics.update(anchor_metrics)
        warnings.extend(anchor_warnings)
        if anchor_drift is not None:
            rescore_failures += anchor_drift.rescore_failures
    elif config.require_anchors:
        warnings.append(
            f"integrity.require_anchors is set but {config.anchors} does not "
            "exist; freeze anchors with a judged `agentkit compare "
            "--establish-baseline` run and commit the file"
        )

    if consistency is None and anchor_drift is None:
        return None, metrics, warnings
    metrics[RESCORE_FAILURES_METRIC] = float(rescore_failures)
    if rescore_failures:
        warnings.append(
            f"{rescore_failures} judge re-invocation(s) failed during the "
            "integrity checks; their rows were left out of the flip-rate "
            "and drift means"
        )
    return (
        IntegrityEvidence(consistency=consistency, anchor_drift=anchor_drift),
        metrics,
        warnings,
    )


def _consistency_check(
    config: IntegrityConfig,
    *,
    rows: Sequence[Mapping[str, Any]],
    outputs_by_row: Sequence[Any],
    metric_samples: Mapping[str, Sequence[float | None]],
    judges: Sequence[RowJudge],
) -> tuple[JudgeConsistencyEvidence | None, dict[str, float], list[str]]:
    eligible = eligible_row_indices(rows, outputs_by_row, metric_samples, judges)
    if not eligible:
        return (
            None,
            {},
            [
                "judge self-consistency could not be measured: no row carries "
                "a recoverable output and a recorded judge score, so the "
                f"required metric {SELF_INCONSISTENCY_METRIC} is absent"
            ],
        )
    selected = sample_indices(eligible, config.consistency_sample, seed=INTEGRITY_SEED)
    deltas_by_judge: dict[str, list[float]] = {}
    failures = 0
    for index in selected:
        row = rows[index]
        outputs = outputs_by_row[index]
        for judge in judges:
            samples = metric_samples.get(judge.metric) or ()
            first = samples[index] if index < len(samples) else None
            if first is None:
                continue
            try:
                second = invoke_judge(judge, row, outputs)
            except Exception:  # noqa: BLE001 - judge endpoints fail like networks
                failures += 1
                continue
            if second is None:
                continue
            deltas_by_judge.setdefault(judge.name, []).append(abs(first - second))
    all_deltas = [delta for deltas in deltas_by_judge.values() for delta in deltas]
    metrics: dict[str, float] = {}
    warnings: list[str] = []
    if not all_deltas:
        warnings.append(
            "judge self-consistency could not be measured: every re-scoring "
            f"call failed or was declined, so {SELF_INCONSISTENCY_METRIC} "
            "is absent"
        )
        evidence = JudgeConsistencyEvidence(
            sample_size=len(selected),
            seed=INTEGRITY_SEED,
            flip_rates={},
            overall=0.0,
            rescore_failures=failures,
        )
        return (evidence if failures else None), metrics, warnings
    flip_rates = {name: fmean(deltas) for name, deltas in deltas_by_judge.items()}
    overall = fmean(all_deltas)
    for judge in judges:
        rate = flip_rates.get(judge.name)
        if rate is not None:
            metrics[integrity_metric(judge.metric, "flip_rate")] = rate
    metrics[SELF_INCONSISTENCY_METRIC] = overall
    if overall > config.max_self_inconsistency:
        warnings.append(
            f"the judges disagreed with themselves on re-scored outputs "
            f"(flip rate {overall:.3f} > {config.max_self_inconsistency:g}). "
            "The judged metrics of this run are noisier than the gate's "
            "deltas assume - investigate the judge before reading the agent "
            "metrics"
        )
    evidence = JudgeConsistencyEvidence(
        sample_size=len(selected),
        seed=INTEGRITY_SEED,
        flip_rates=flip_rates,
        overall=overall,
        rescore_failures=failures,
    )
    return evidence, metrics, warnings


def _anchor_check(
    config: IntegrityConfig,
    *,
    anchors: JudgeAnchors,
    judges: Sequence[RowJudge],
) -> tuple[AnchorDriftEvidence | None, dict[str, float], list[str]]:
    judged_names = {judge.name for judge in judges}
    recorded_names = {
        name for row in anchors.rows for name in row.scores if name in judged_names
    }
    if not anchors.rows or not recorded_names:
        return (
            None,
            {},
            [
                f"the frozen anchors in {config.anchors} record no scores for "
                "any judge in this run, so drift could not be measured "
                f"({ANCHOR_DRIFT_METRIC} is absent); re-establish the "
                "baseline to freeze anchors for the current judges"
            ],
        )
    deltas_by_judge: dict[str, list[float]] = {}
    failures = 0
    for row in anchors.rows:
        for judge in judges:
            recorded = row.scores.get(judge.name)
            if recorded is None:
                continue
            try:
                rescored = invoke_judge(
                    judge,
                    {"inputs": row.inputs, "expectations": row.expectations},
                    thaw_value(row.outputs),
                )
            except Exception:  # noqa: BLE001 - judge endpoints fail like networks
                failures += 1
                continue
            if rescored is None:
                continue
            deltas_by_judge.setdefault(judge.name, []).append(abs(recorded - rescored))
    all_deltas = [delta for deltas in deltas_by_judge.values() for delta in deltas]
    metrics: dict[str, float] = {}
    warnings: list[str] = []
    if not all_deltas:
        warnings.append(
            "anchor drift could not be measured: every anchor re-scoring "
            f"call failed or was declined, so {ANCHOR_DRIFT_METRIC} is absent"
        )
        evidence = AnchorDriftEvidence(
            anchors_ref=config.anchors,
            anchors_digest=anchors.digest,
            rows=len(anchors.rows),
            drift_by_scorer={},
            overall=0.0,
            rescore_failures=failures,
        )
        return (evidence if failures else None), metrics, warnings
    drift_by_scorer = {name: fmean(deltas) for name, deltas in deltas_by_judge.items()}
    overall = fmean(all_deltas)
    for judge in judges:
        drift = drift_by_scorer.get(judge.name)
        if drift is not None:
            metrics[integrity_metric(judge.metric, "anchor_drift")] = drift
    metrics[ANCHOR_DRIFT_METRIC] = overall
    if overall > config.max_anchor_drift:
        warnings.append(
            f"anchor drift {overall:.3f} exceeds {config.max_anchor_drift:g}: "
            + ANCHOR_DRIFT_EXPLANATION
        )
    evidence = AnchorDriftEvidence(
        anchors_ref=config.anchors,
        anchors_digest=anchors.digest,
        rows=len(anchors.rows),
        drift_by_scorer=drift_by_scorer,
        overall=overall,
        rescore_failures=failures,
    )
    return evidence, metrics, warnings


def build_anchor_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
    outputs_by_row: Sequence[Any],
    metric_samples: Mapping[str, Sequence[float | None]],
    judges: Sequence[RowJudge],
    limit: int = DEFAULT_ANCHOR_ROWS,
    seed: int = INTEGRITY_SEED,
) -> tuple[AnchorRow, ...]:
    """Freeze a deterministic sample of this run's rows with their scores."""

    eligible = eligible_row_indices(rows, outputs_by_row, metric_samples, judges)
    selected = sample_indices(eligible, limit, seed=seed)
    frozen: list[AnchorRow] = []
    for index in selected:
        row = rows[index]
        scores: dict[str, float] = {}
        for judge in judges:
            samples = metric_samples.get(judge.metric) or ()
            value = samples[index] if index < len(samples) else None
            if value is not None:
                scores[judge.name] = float(value)
        if not scores:
            continue
        frozen.append(
            AnchorRow(
                inputs=dict(row.get("inputs") or {}),
                outputs=outputs_by_row[index],
                expectations=dict(row.get("expectations") or {}),
                scores=scores,
            )
        )
    return tuple(frozen)


def anchor_rows_digest(rows: Sequence[AnchorRow]) -> str:
    canonical = json.dumps(
        [row.model_dump(mode="json") for row in rows],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_anchors(
    path: Path,
    *,
    rows: Sequence[AnchorRow],
    recorded_at: str,
    recorded_by: str,
    change_id: str,
    judge_model: str | None,
    judge_model_identity: str | None,
    judge_prompts: Mapping[str, str],
    scorer_versions: Mapping[str, int],
) -> JudgeAnchors:
    """Atomic, sorted, newline-terminated write (review-friendly diffs)."""

    anchors = JudgeAnchors(
        recorded_at=recorded_at,
        recorded_by=recorded_by,
        change_id=change_id,
        judge_model=judge_model,
        judge_model_identity=judge_model_identity,
        judge_prompts=dict(judge_prompts),
        scorer_versions=dict(scorer_versions),
        rows=tuple(rows),
        digest=anchor_rows_digest(rows),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        json.dumps(
            anchors.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    descriptor, scratch_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    scratch = Path(scratch_name)
    try:
        scratch.write_text(text, encoding="utf-8")
        os.replace(scratch, path)
    finally:
        scratch.unlink(missing_ok=True)
    return anchors


def load_anchors(path: Path) -> JudgeAnchors:
    """Load and digest-verify the committed anchors.

    A hand-edited or regenerated-without-freezing file is refused: anchors
    only separate judge drift from agent change while their outputs and
    recorded scores are exactly what the establishing run froze.
    """

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ConfigError(f"could not read judge anchors {path}: {error}") from error
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ConfigError(f"{path} is not valid UTF-8 JSON: {error}") from error
    if not isinstance(document, Mapping):
        raise ConfigError(f"{path} must contain a JSON object")
    try:
        anchors = JudgeAnchors(**document)
    except ValidationError as error:
        raise ConfigError(f"{path} is not a valid anchors file: {error}") from error
    recomputed = anchor_rows_digest(anchors.rows)
    if recomputed != anchors.digest:
        raise ConfigError(
            f"{path} changed after it was frozen (digest mismatch)",
            remediation=(
                "Anchors are immutable evidence. Restore the committed file, "
                "or re-freeze them with a judged `agentkit compare "
                "--establish-baseline` run."
            ),
        )
    return anchors
