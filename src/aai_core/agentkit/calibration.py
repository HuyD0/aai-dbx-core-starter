"""Judge calibration: measuring the instrument against the people it stands for.

A judge score gates releases, so "our agent scores 0.87 on correctness" is
only auditable as "0.87 under a judge that agrees with our subject-matter
experts at κ = 0.71 on a named calibration set". This module computes that
record and persists it as a committed, review-friendly platform artifact:

- chance-adjusted agreement (Cohen's κ) between the judge and the SME
  consensus — raw percent agreement is inflated by class imbalance;
- the human ceiling: pairwise inter-annotator κ. A judge cannot be more
  consistent than the humans defining the target, and a low ceiling means
  the rubric is the problem, not the judge.

Calibration is out-of-band from the commit gate (`agentkit judge
calibrate`, run on judge releases, not per commit). The per-commit path
only checks that a current, passing record exists for the pinned judge
when ``integrity.require_calibration`` is set. Alignment/optimization
(``Scorer.align``) deliberately stays a native MLflow API — see
docs/genai-lifecycle.md — this module measures; it does not tune.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, ValidationError, field_validator

from aai_core.agentkit._values import numeric_score
from aai_core.agentkit.errors import ConfigError
from aai_core.contracts import ContractModel

DEFAULT_MINIMUM_KAPPA = 0.60
MINIMUM_LABELS = 20


class AnnotatorVerdict(ContractModel):
    annotator: str = Field(min_length=1)
    value: str | int | float | bool

    @field_validator("annotator")
    @classmethod
    def refuse_personal_identity(cls, value: str) -> str:
        if "@" in value:
            raise ValueError(
                "annotator must be a non-personal identity such as a group "
                "or reviewer alias, never an email address"
            )
        return value.strip()


class CalibrationLabel(ContractModel):
    """One calibration example: the judge's verdict and the SME verdicts."""

    example_id: str = Field(min_length=1)
    judge_value: str | int | float | bool
    annotations: tuple[AnnotatorVerdict, ...] = Field(min_length=1)

    @field_validator("annotations", mode="before")
    @classmethod
    def coerce_annotations(cls, value: Any) -> Any:
        if isinstance(value, list):
            return tuple(
                AnnotatorVerdict(**item) if isinstance(item, Mapping) else item
                for item in value
            )
        return value


class CalibrationRecord(ContractModel):
    """The committed calibration evidence for one registry judge."""

    schema_version: Literal[1] = 1
    scorer: str = Field(min_length=1)
    scorer_version: int = Field(ge=1)
    judge_model: str | None = None
    judge_model_identity: str | None = None
    judge_prompt_uri: str | None = None
    labels_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_size: int = Field(ge=1)
    annotator_count: int = Field(ge=1)
    tie_count: int = Field(default=0, ge=0)
    percent_agreement: float = Field(ge=0.0, le=1.0)
    kappa: float = Field(ge=-1.0, le=1.0)
    human_ceiling_kappa: float | None = Field(default=None, ge=-1.0, le=1.0)
    minimum_kappa: float = Field(default=DEFAULT_MINIMUM_KAPPA, ge=0.0, le=1.0)
    passed: bool
    recorded_at: str = Field(min_length=1)
    decided_by: str | None = Field(default=None, min_length=1)

    @field_validator("decided_by")
    @classmethod
    def refuse_personal_identity(cls, value: str | None) -> str | None:
        if value is not None and "@" in value:
            raise ValueError(
                "decided_by must be a non-personal identity such as a group "
                "name, never an email address"
            )
        return value


def canonical_verdict(value: Any) -> str:
    """One category label per verdict, shared with the scoring path.

    ``yes``/``true``/1 and ``no``/``false``/0 collapse to the same
    categories the per-row samples use, so a judge answering "yes" agrees
    with an SME labelling ``true``. Anything else is its own category.
    """

    numeric = numeric_score(value)
    if numeric == 1.0:
        return "pass"
    if numeric == 0.0:
        return "fail"
    if numeric is not None:
        return f"{numeric:g}"
    return str(value).strip().casefold()


def percent_agreement(first: Sequence[str], second: Sequence[str]) -> float:
    if len(first) != len(second) or not first:
        raise ConfigError("agreement needs two equal, non-empty label sequences")
    matches = sum(1 for a, b in zip(first, second, strict=True) if a == b)
    return matches / len(first)


def cohen_kappa(first: Sequence[str], second: Sequence[str]) -> float:
    """Chance-adjusted agreement between two raters over the same items.

    κ = (pₒ − pₑ) / (1 − pₑ). When chance agreement is already 1 (both
    raters use a single identical category) the statistic is undefined;
    perfect observed agreement reports 1.0 and anything else 0.0, which is
    the conservative reading either way.
    """

    observed = percent_agreement(first, second)
    total = len(first)
    categories = set(first) | set(second)
    expected = sum(
        (sum(1 for item in first if item == category) / total)
        * (sum(1 for item in second if item == category) / total)
        for category in categories
    )
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def consensus_verdicts(
    labels: Sequence[CalibrationLabel],
) -> tuple[list[str], list[str], int]:
    """(judge, SME-consensus) category pairs, and how many ties were dropped.

    Consensus is a strict majority of the annotators on that example. A
    tie carries no target for the judge to agree with, so tied examples
    are excluded from the comparison and counted — many ties are a rubric
    problem the record should surface, not paper over.
    """

    judge: list[str] = []
    humans: list[str] = []
    ties = 0
    for label in labels:
        votes: dict[str, int] = {}
        for annotation in label.annotations:
            category = canonical_verdict(annotation.value)
            votes[category] = votes.get(category, 0) + 1
        ranked = sorted(votes.items(), key=lambda item: (-item[1], item[0]))
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            ties += 1
            continue
        judge.append(canonical_verdict(label.judge_value))
        humans.append(ranked[0][0])
    return judge, humans, ties


def human_ceiling(labels: Sequence[CalibrationLabel]) -> float | None:
    """Mean pairwise inter-annotator κ — the target a judge cannot beat."""

    by_annotator: dict[str, dict[str, str]] = {}
    for label in labels:
        for annotation in label.annotations:
            by_annotator.setdefault(annotation.annotator, {})[label.example_id] = (
                canonical_verdict(annotation.value)
            )
    kappas: list[float] = []
    for first, second in combinations(sorted(by_annotator), 2):
        shared = sorted(set(by_annotator[first]) & set(by_annotator[second]))
        if len(shared) < 2:
            continue
        kappas.append(
            cohen_kappa(
                [by_annotator[first][example] for example in shared],
                [by_annotator[second][example] for example in shared],
            )
        )
    if not kappas:
        return None
    return sum(kappas) / len(kappas)


def load_labels(path: Path) -> tuple[CalibrationLabel, ...]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ConfigError(
            f"could not read calibration labels {path}: {error}"
        ) from error
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ConfigError(f"{path} is not valid UTF-8 JSON: {error}") from error
    if not isinstance(document, list):
        raise ConfigError(f"{path} must contain a JSON list of labelled examples")
    labels = []
    for index, item in enumerate(document):
        if not isinstance(item, Mapping):
            raise ConfigError(f"{path} entry {index} must be a JSON object")
        try:
            labels.append(CalibrationLabel(**item))
        except ValidationError as error:
            raise ConfigError(
                f"{path} entry {index} is not a valid calibration label: {error}"
            ) from error
    return tuple(labels)


def labels_digest(labels: Sequence[CalibrationLabel]) -> str:
    canonical = json.dumps(
        [label.model_dump(mode="json") for label in labels],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def calibrate(
    *,
    scorer: str,
    scorer_version: int,
    labels: Sequence[CalibrationLabel],
    minimum_kappa: float = DEFAULT_MINIMUM_KAPPA,
    judge_model: str | None = None,
    judge_model_identity: str | None = None,
    judge_prompt_uri: str | None = None,
    recorded_at: str,
    decided_by: str | None = None,
) -> CalibrationRecord:
    """Measure the judge against the SME labels and build the record."""

    if len(labels) < MINIMUM_LABELS:
        raise ConfigError(
            f"calibration needs at least {MINIMUM_LABELS} labelled examples; "
            f"got {len(labels)}. κ over a handful of labels is noise dressed "
            "as evidence.",
            remediation=(
                "Collect more SME labels — stratified by failure mode, not "
                "sampled uniformly — before releasing the judge."
            ),
        )
    judge, humans, ties = consensus_verdicts(labels)
    if not judge:
        raise ConfigError(
            "every calibration example was an annotator tie, so there is no "
            "consensus for the judge to agree with",
            remediation=(
                "The rubric is under-specified: annotators cannot agree "
                "with each other. Fix the rubric before measuring the judge."
            ),
        )
    kappa = cohen_kappa(judge, humans)
    annotators = {
        annotation.annotator for label in labels for annotation in label.annotations
    }
    return CalibrationRecord(
        scorer=scorer,
        scorer_version=scorer_version,
        judge_model=judge_model,
        judge_model_identity=judge_model_identity,
        judge_prompt_uri=judge_prompt_uri,
        labels_digest=labels_digest(labels),
        sample_size=len(judge),
        annotator_count=len(annotators),
        tie_count=ties,
        percent_agreement=percent_agreement(judge, humans),
        kappa=kappa,
        human_ceiling_kappa=human_ceiling(labels),
        minimum_kappa=minimum_kappa,
        passed=kappa >= minimum_kappa,
        recorded_at=recorded_at,
        decided_by=decided_by,
    )


def calibration_path(root: Path, directory: str, scorer: str) -> Path:
    return root / directory / f"{scorer}.json"


def write_calibration(path: Path, record: CalibrationRecord) -> None:
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


def load_calibration(path: Path) -> CalibrationRecord:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ConfigError(
            f"could not read calibration record {path}: {error}"
        ) from error
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ConfigError(f"{path} is not valid UTF-8 JSON: {error}") from error
    if not isinstance(document, Mapping):
        raise ConfigError(f"{path} must contain a JSON object")
    try:
        return CalibrationRecord(**document)
    except ValidationError as error:
        raise ConfigError(
            f"{path} is not a valid calibration record: {error}"
        ) from error


def calibration_failures(
    *,
    root: Path,
    directory: str,
    judge_scorers: Mapping[str, int],
) -> list[str]:
    """Why the pinned judges are not calibration-covered, if they are not.

    Enforced only when ``integrity.require_calibration`` is set. Existence,
    a passing κ, and the scorer version are checkable offline; prompt-URI
    and served-identity drift against the record are surfaced by evidence
    rather than blocking here (comparability already blocks a judge that
    moved since the baseline).
    """

    failures: list[str] = []
    for name in sorted(judge_scorers):
        path = calibration_path(root, directory, name)
        if not path.is_file():
            failures.append(
                f"judge scorer {name!r} has no calibration record at "
                f"{directory}/{name}.json; run `agentkit judge calibrate "
                f"--scorer {name} --labels <sme-labels.json>`"
            )
            continue
        record = load_calibration(path)
        if record.scorer != name:
            failures.append(
                f"{directory}/{name}.json records scorer {record.scorer!r}, "
                f"not {name!r}"
            )
            continue
        if record.scorer_version != judge_scorers[name]:
            failures.append(
                f"judge scorer {name!r} is v{judge_scorers[name]} but its "
                f"calibration record measured v{record.scorer_version}; "
                "re-calibrate under the released scorer version"
            )
            continue
        if not record.passed:
            failures.append(
                f"judge scorer {name!r} failed calibration "
                f"(κ {record.kappa:.3f} < {record.minimum_kappa:g}); fix the "
                "rubric or judge and re-calibrate before gating with it"
            )
    return failures


def calibration_status(
    *,
    root: Path,
    directory: str,
    judge_scorers: Mapping[str, int],
    judge_prompts: Mapping[str, str] | None = None,
    judge_model_identity: str | None = None,
) -> list[dict[str, Any]]:
    """Per-judge calibration facts for the evidence pack (never blocking)."""

    rows: list[dict[str, Any]] = []
    for name in sorted(judge_scorers):
        path = calibration_path(root, directory, name)
        if not path.is_file():
            rows.append({"scorer": name, "status": "uncalibrated"})
            continue
        try:
            record = load_calibration(path)
        except ConfigError as error:
            rows.append({"scorer": name, "status": "unreadable", "reason": str(error)})
            continue
        row: dict[str, Any] = {
            "scorer": name,
            "status": "passed" if record.passed else "failed",
            "kappa": record.kappa,
            "human_ceiling_kappa": record.human_ceiling_kappa,
            "minimum_kappa": record.minimum_kappa,
            "sample_size": record.sample_size,
            "recorded_at": record.recorded_at,
        }
        if record.scorer_version != judge_scorers[name]:
            row["stale"] = (
                f"measured scorer v{record.scorer_version}, "
                f"run used v{judge_scorers[name]}"
            )
        prompt = (judge_prompts or {}).get(name)
        if prompt and record.judge_prompt_uri and prompt != record.judge_prompt_uri:
            row["stale_prompt"] = (
                f"measured {record.judge_prompt_uri}, run used {prompt}"
            )
        if (
            judge_model_identity
            and record.judge_model_identity
            and judge_model_identity != record.judge_model_identity
        ):
            row["stale_judge"] = (
                f"measured {record.judge_model_identity}, "
                f"run judged by {judge_model_identity}"
            )
        rows.append(row)
    return rows
