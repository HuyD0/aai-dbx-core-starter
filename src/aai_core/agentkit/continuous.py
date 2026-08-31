"""Continuous logprob-weighted judge scoring — the LLM-as-a-verifier path.

A discrete judge collapses its verdict into the one label it emits, so two
answers of different quality tie whenever they round to the same label —
which makes discrete judges nearly useless for *ranking*. This module reads
the judge's uncertainty instead: it prompts for a single-token score label,
takes ``exp()`` of the top logprobs at that position, keeps only the valid
score tokens, renormalizes by their retained mass, and returns the
probability-weighted average mapped to ``[0, 1]``. A judge that emits "B"
at 80% confidence now scores differently from one that emits "B" at 55%.

Design decisions, in this platform's terms:

- **Letters, not digits.** Score labels are single uppercase letters
  (``A`` = worst … up to 26 points), because every common tokenizer emits
  one letter as one token; multi-digit scores split across tokens and the
  distribution at the first position stops meaning "the score".
- **Granularity and repeats are parameters**, not constants — the point is
  to settle them empirically (``scripts/sweep_continuous_scoring.py``), so
  every run records them and the sweep compares 5/10/20-point scales.
- **Criteria decomposition.** Each row is judged per criterion and the
  criterion scores are averaged, instead of one "is this good" call. The
  criteria are versioned platform assets like judge prompts: projects never
  redefine them.
- **The verifier makes its own chat calls.** MLflow's built-in judges never
  expose logprobs, so this path calls the configured OpenAI-compatible
  model directly (``provider: databricks`` or ``azure_apim``) through the
  governed :class:`~aai_core.providers.types.ChatModel` adapter. The
  Anthropic API exposes no ``top_logprobs``, so an Anthropic-backed
  endpoint cannot serve this path; support is probed at runtime and an
  incapable backend falls back to the discrete path with a warning rather
  than failing the run.
- **Report-only until validated.** The continuous metrics land beside the
  discrete ones (which keep gating); nothing here is registered in the
  scorer catalog or given a threshold until the sweep has settled the
  configuration. A continuous scorer that *crashes* still fails the gate
  through ``<scorer>/error_count`` like any scorer — an unsupported
  backend is the one condition that degrades instead of erroring, because
  it is detected before any row is scored.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from statistics import fmean
from threading import Lock
from typing import TYPE_CHECKING, Any

from pydantic import Field

from aai_core.agentkit.errors import ConfigError
from aai_core.contracts import ContractModel
from aai_core.providers.types import ProviderRequestError
from aai_core.scorers import _output_text

if TYPE_CHECKING:
    from aai_core.agentkit.datasets import DatasetShape
    from aai_core.providers.types import ChatModel
    from aai_core.runtime import PlatformSettings

__all__ = [
    "CONTINUOUS_SCORER_NAME",
    "CONTINUOUS_SCORING_VERSION",
    "CORRECTNESS_CRITERIA",
    "ContinuousJudgment",
    "ContinuousScoringConfig",
    "ContinuousVerifier",
    "CriterionScore",
    "JudgeCriterion",
    "PointwiseScore",
    "VerifierStats",
    "detect_logprob_support",
    "kendall_tau_b",
    "score_labels",
    "tie_rate",
    "top_logprob_pairs",
    "weigh_top_logprobs",
]

_LOGGER = logging.getLogger(__name__)

CONTINUOUS_SCORER_NAME = "correctness_continuous"
CONTINUOUS_SCORING_VERSION = 1
# The OpenAI-compatible ceiling; Azure OpenAI and Databricks FMAPI accept it.
TOP_LOGPROBS = 20
# Provider statuses that mean "this backend refused the logprobs request
# parameters" — a capability signal, not a transient failure. 401/403/429
# and 5xx are real failures and always propagate.
_UNSUPPORTED_STATUSES = frozenset({400, 404, 422})

_MAX_GRANULARITY = 26  # single uppercase letters A..Z


class ContinuousScoringConfig(ContractModel):
    """Project-owned switches for the continuous scoring experiment.

    Off by default; enabling it adds the continuous verifier beside the
    discrete judges without replacing them. ``judge_model`` names the
    logical model the verifier calls (any ``providers.models`` entry in
    ``aai-platform.yml``); when unset the ordinary judge model is used.
    """

    enabled: bool = False
    judge_model: str | None = Field(default=None, min_length=1)
    granularity: int = Field(default=20, ge=2, le=_MAX_GRANULARITY)
    repeats: int = Field(default=1, ge=1, le=16)
    low_mass_threshold: float = Field(default=0.5, gt=0.0, le=1.0)


@dataclass(frozen=True)
class JudgeCriterion:
    """One versioned criterion of a decomposed judgment."""

    key: str
    instruction: str


# Versioned platform assets, like judge prompts: a project selects the
# continuous scorer, never redefines what its criteria mean. Bump
# CONTINUOUS_SCORING_VERSION when these change.
CORRECTNESS_CRITERIA: tuple[JudgeCriterion, ...] = (
    JudgeCriterion(
        key="factual_agreement",
        instruction=(
            "The response agrees with the expected answer on every fact it "
            "states and contradicts nothing in it."
        ),
    ),
    JudgeCriterion(
        key="coverage",
        instruction=(
            "The response covers every substantive point the expected "
            "answer contains."
        ),
    ),
    JudgeCriterion(
        key="no_fabrication",
        instruction=(
            "The response adds no material claims that the expected answer "
            "does not support."
        ),
    ),
)

PAIRWISE_CRITERION = JudgeCriterion(
    key="pairwise_preference",
    instruction=(
        "Which response answers the request better overall — more "
        "accurately, more completely, and without fabrication?"
    ),
)


def score_labels(granularity: int) -> tuple[str, ...]:
    """The single-token score alphabet for a scale of ``granularity`` points."""

    if not 2 <= granularity <= _MAX_GRANULARITY:
        raise ConfigError(
            f"continuous granularity must be between 2 and {_MAX_GRANULARITY}"
            f"; got {granularity}",
            remediation="Set scorers.continuous.granularity in agentkit.yaml.",
        )
    return tuple(chr(ord("A") + index) for index in range(granularity))


@dataclass(frozen=True)
class ContinuousJudgment:
    """One verifier call, read from the score position's top logprobs."""

    score: float
    normalization_mass: float
    top_label: str
    low_mass: bool
    # The granularity travels with the judgment so the discrete parse does
    # not need a second source of truth for the scale.
    granularity: int

    @property
    def discrete_score(self) -> float:
        """What a discrete parse of the same call would have scored."""

        return _label_value(self.top_label, self.granularity)


def weigh_top_logprobs(
    pairs: Sequence[tuple[str, float]],
    *,
    granularity: int,
    low_mass_threshold: float,
) -> ContinuousJudgment | None:
    """Confidence-weighted score from the score position's top logprobs.

    ``exp()`` of each logprob, filtered to the valid score alphabet,
    normalized by the retained mass. Token variants that differ only in
    surrounding whitespace or case ("B", " B", "b") are the same label and
    their probabilities sum. ``None`` when no score token appears at all —
    the prompt did not steer the model to the scale, and inventing a score
    from unrelated tokens would put noise under a metric name. The
    *normalization mass* (the sum before dividing) is the calibration
    signal: below ``low_mass_threshold`` the judgment is flagged.
    """

    labels = score_labels(granularity)
    positions = {label: index for index, label in enumerate(labels)}
    mass_by_label: dict[str, float] = {}
    for token, logprob in pairs:
        label = token.strip().upper()
        if label not in positions:
            continue
        mass_by_label[label] = mass_by_label.get(label, 0.0) + math.exp(logprob)
    mass = sum(mass_by_label.values())
    if mass <= 0.0:
        return None
    top = granularity - 1
    weighted = sum(
        probability * (positions[label] / top)
        for label, probability in mass_by_label.items()
    )
    top_label = max(mass_by_label, key=lambda label: mass_by_label[label])
    return ContinuousJudgment(
        score=weighted / mass,
        normalization_mass=mass,
        top_label=top_label,
        low_mass=mass < low_mass_threshold,
        granularity=granularity,
    )


def _label_value(label: str, granularity: int) -> float:
    labels = score_labels(granularity)
    return labels.index(label) / (granularity - 1)


def top_logprob_pairs(raw_response: Any) -> list[tuple[str, float]] | None:
    """The (token, logprob) alternatives at the first generated position.

    Tolerates both attribute-shaped SDK objects and plain mappings, because
    the native client is provider-owned. ``None`` when the response carries
    no readable top logprobs — the capability signal the fallback reads.
    """

    choices = _read(raw_response, "choices")
    if not isinstance(choices, Sequence) or not choices:
        return None
    logprobs = _read(choices[0], "logprobs")
    content = _read(logprobs, "content")
    if not isinstance(content, Sequence) or not content:
        return None
    alternatives = _read(content[0], "top_logprobs")
    if not isinstance(alternatives, Sequence):
        return None
    pairs: list[tuple[str, float]] = []
    for alternative in alternatives:
        token = _read(alternative, "token")
        logprob = _read(alternative, "logprob")
        if isinstance(token, str) and isinstance(logprob, (int, float)):
            pairs.append((token, float(logprob)))
    return pairs or None


def _read(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


class VerifierStats:
    """Thread-safe per-run instrument telemetry.

    MLflow runs scorers on worker threads, so every verifier call records
    here under a lock and the runner reads one snapshot after the run.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self.calls = 0
        self.invalid_calls = 0
        self.low_mass_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self._mass_total = 0.0
        self._mass_min: float | None = None

    def record(
        self,
        judgment: ContinuousJudgment | None,
        usage: Mapping[str, int],
    ) -> None:
        with self._lock:
            self.calls += 1
            self.input_tokens += int(
                usage.get("input_tokens", usage.get("prompt_tokens", 0))
            )
            self.output_tokens += int(
                usage.get("output_tokens", usage.get("completion_tokens", 0))
            )
            if judgment is None:
                self.invalid_calls += 1
                return
            if judgment.low_mass:
                self.low_mass_calls += 1
            self._mass_total += judgment.normalization_mass
            if self._mass_min is None:
                self._mass_min = judgment.normalization_mass
            else:
                self._mass_min = min(self._mass_min, judgment.normalization_mass)

    def metrics(self) -> dict[str, float]:
        with self._lock:
            metrics = {
                "continuous/judge_calls": float(self.calls),
                "continuous/input_tokens": float(self.input_tokens),
                "continuous/output_tokens": float(self.output_tokens),
            }
            valid = self.calls - self.invalid_calls
            if self.calls:
                metrics["continuous/invalid_rate"] = self.invalid_calls / self.calls
                metrics["continuous/low_mass_rate"] = self.low_mass_calls / self.calls
            if valid and self._mass_min is not None:
                metrics["continuous/normalization_mass_mean"] = self._mass_total / valid
                metrics["continuous/normalization_mass_min"] = self._mass_min
            return metrics


@dataclass(frozen=True)
class CriterionScore:
    """One criterion's verdict, averaged over the configured repeats."""

    continuous: float
    discrete: float
    low_mass: bool
    normalization_mass: float


@dataclass(frozen=True)
class PointwiseScore:
    """One row's decomposed verdict: the mean over its scored criteria."""

    continuous: float
    discrete: float
    criteria_scored: int


_SYSTEM_PROMPT = (
    "You are a strict evaluation verifier. Answer with exactly one letter "
    "and nothing else."
)


@dataclass
class ContinuousVerifier:
    """The logprob-weighted verifier over one governed chat model.

    ``repeats`` runs each judgment K times and averages; for pairwise
    comparisons the candidate in the A slot alternates across repeats so
    positional bias cancels instead of accumulating.
    """

    model: ChatModel
    granularity: int = 20
    repeats: int = 1
    low_mass_threshold: float = 0.5
    criteria: tuple[JudgeCriterion, ...] = CORRECTNESS_CRITERIA
    stats: VerifierStats = field(default_factory=VerifierStats)

    def judge_once(self, prompt: str) -> ContinuousJudgment | None:
        response = self.model.generate(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=1,
            provider_options={"logprobs": True, "top_logprobs": TOP_LOGPROBS},
        )
        pairs = top_logprob_pairs(response.raw)
        judgment = (
            None
            if pairs is None
            else weigh_top_logprobs(
                pairs,
                granularity=self.granularity,
                low_mass_threshold=self.low_mass_threshold,
            )
        )
        self.stats.record(judgment, response.usage)
        return judgment

    def score_criterion(
        self,
        *,
        request: str,
        response: str,
        expected: str,
        criterion: JudgeCriterion,
    ) -> CriterionScore | None:
        prompt = self._pointwise_prompt(
            request=request,
            response=response,
            expected=expected,
            criterion=criterion,
        )
        judgments = [self.judge_once(prompt) for _ in range(self.repeats)]
        valid = [judgment for judgment in judgments if judgment is not None]
        if not valid:
            return None
        return CriterionScore(
            continuous=fmean(judgment.score for judgment in valid),
            discrete=fmean(judgment.discrete_score for judgment in valid),
            low_mass=any(judgment.low_mass for judgment in valid),
            normalization_mass=fmean(judgment.normalization_mass for judgment in valid),
        )

    def score_response(
        self,
        *,
        request: str,
        response: str,
        expected: str,
    ) -> PointwiseScore | None:
        """Criteria-decomposed verdict: score each criterion, average."""

        scores = [
            score
            for criterion in self.criteria
            if (
                score := self.score_criterion(
                    request=request,
                    response=response,
                    expected=expected,
                    criterion=criterion,
                )
            )
            is not None
        ]
        if not scores:
            return None
        return PointwiseScore(
            continuous=fmean(score.continuous for score in scores),
            discrete=fmean(score.discrete for score in scores),
            criteria_scored=len(scores),
        )

    def compare(
        self,
        *,
        request: str,
        first: str,
        second: str,
        expected: str | None = None,
        criterion: JudgeCriterion = PAIRWISE_CRITERION,
    ) -> float | None:
        """Continuous preference for ``first`` over ``second`` in ``[0, 1]``.

        0.5 is a tie. Each repeat alternates which candidate sits in the A
        slot; a swapped repeat's score is mirrored (``1 - score``) before
        averaging, so positional bias cancels across repeats.
        """

        preferences: list[float] = []
        for repeat in range(self.repeats):
            swapped = repeat % 2 == 1
            slot_a, slot_b = (second, first) if swapped else (first, second)
            prompt = self._pairwise_prompt(
                request=request,
                slot_a=slot_a,
                slot_b=slot_b,
                expected=expected,
                criterion=criterion,
            )
            judgment = self.judge_once(prompt)
            if judgment is None:
                continue
            preferences.append(1.0 - judgment.score if swapped else judgment.score)
        if not preferences:
            return None
        return fmean(preferences)

    def _scale_line(self, *, worst: str, best: str) -> str:
        last = score_labels(self.granularity)[-1]
        return (
            f"Rate on a {self.granularity}-letter scale from A to {last}: "
            f"A means {worst}, {last} means {best}, and the letters between "
            "are evenly spaced."
        )

    def _pointwise_prompt(
        self,
        *,
        request: str,
        response: str,
        expected: str,
        criterion: JudgeCriterion,
    ) -> str:
        last = score_labels(self.granularity)[-1]
        scale = self._scale_line(
            worst="the criterion is completely unsatisfied",
            best="it is perfectly satisfied",
        )
        return (
            f"{scale}\n\n"
            f"Criterion: {criterion.instruction}\n\n"
            f"Request: {request}\n\n"
            f"Expected answer: {expected}\n\n"
            f"Response to rate: {response}\n\n"
            f"Answer with exactly one letter from A to {last}."
        )

    def _pairwise_prompt(
        self,
        *,
        request: str,
        slot_a: str,
        slot_b: str,
        expected: str | None,
        criterion: JudgeCriterion,
    ) -> str:
        last = score_labels(self.granularity)[-1]
        scale = self._scale_line(
            worst="response A is decisively worse than response B",
            best="response A is decisively better than response B",
        )
        reference = f"\nExpected answer: {expected}\n" if expected else ""
        return (
            f"{scale} The middle of the scale means they are equal.\n\n"
            f"Criterion: {criterion.instruction}\n\n"
            f"Request: {request}\n{reference}\n"
            f"Response A: {slot_a}\n\n"
            f"Response B: {slot_b}\n\n"
            f"Answer with exactly one letter from A to {last}."
        )


def detect_logprob_support(model: ChatModel) -> bool:
    """Whether the configured backend returns top logprobs at all.

    One tiny probe call. A 400/404/422 from the provider means the request
    parameters were refused — the capability answer, not an outage — as
    does a well-formed response carrying no logprobs (backends that
    silently drop the parameter). Authentication, permission, rate-limit,
    and server failures propagate: they say nothing about capability, and
    degrading on them would silently turn a broken deployment into a
    discrete-only run.
    """

    try:
        response = model.generate(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": "Reply with the letter A."},
            ],
            temperature=0.0,
            max_tokens=1,
            provider_options={"logprobs": True, "top_logprobs": TOP_LOGPROBS},
        )
    except ProviderRequestError as error:
        if error.status_code in _UNSUPPORTED_STATUSES:
            return False
        raise
    return top_logprob_pairs(response.raw) is not None


def tie_rate(values: Sequence[float | None]) -> float | None:
    """Fraction of unordered row pairs whose scores tie.

    The instrument-comparison number: a discrete judge's whole problem is
    that this is high. Scores are compared after rounding away float noise
    so a continuous instrument is not credited for 1e-17 differences.
    """

    scored = [round(value, 9) for value in values if value is not None]
    pairs = len(scored) * (len(scored) - 1) // 2
    if pairs == 0:
        return None
    tied = 0
    counts: dict[float, int] = {}
    for value in scored:
        tied += counts.get(value, 0)
        counts[value] = counts.get(value, 0) + 1
    return tied / pairs


def kendall_tau_b(
    first: Sequence[float],
    second: Sequence[float],
) -> float | None:
    """Kendall's tau-b between two score sequences over the same items.

    Tau-b corrects for ties, which matters here twice over: the reference
    ranking may hold none while a coarse instrument produces many, and
    plain tau would reward the instrument for every tie it breaks by
    luck. ``None`` when either side is constant — agreement with a
    ranking that ranks nothing is undefined, not zero.
    """

    if len(first) != len(second) or len(first) < 2:
        return None
    concordant = discordant = 0
    for index, (a1, b1) in enumerate(zip(first, second, strict=True)):
        for a2, b2 in zip(first[index + 1 :], second[index + 1 :], strict=True):
            product = _sign(a1 - a2) * _sign(b1 - b2)
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
    pairs = len(first) * (len(first) - 1) / 2
    tied_first = _tied_pairs(first)
    tied_second = _tied_pairs(second)
    if pairs in (tied_first, tied_second):
        return None
    return (concordant - discordant) / math.sqrt(
        (pairs - tied_first) * (pairs - tied_second)
    )


def _sign(delta: float) -> int:
    if round(delta, 9) == 0:
        return 0
    return 1 if delta > 0 else -1


def _tied_pairs(values: Sequence[float]) -> float:
    counts: dict[float, int] = {}
    for value in values:
        rounded = round(value, 9)
        counts[rounded] = counts.get(rounded, 0) + 1
    return sum(count * (count - 1) / 2 for count in counts.values())


# --- Runner integration ------------------------------------------------------
#
# Deliberately outside the scorer catalog for now: granularity and repeats
# change what the number means, and a registry entry whose meaning varies by
# project configuration would break "0.8 means the same thing everywhere".
# The metrics are report-only; registration and thresholds come after the
# sweep settles the configuration (docs/continuous-scoring.md).

_EXPECTATION_NEEDS = frozenset({"expected_facts", "expected_response"})


@dataclass(frozen=True)
class ContinuousRunPlan:
    """What the continuous path will do in one scoring run, decided upfront
    so the cost message and the budget ceiling cover it before any spend."""

    config: ContinuousScoringConfig
    judge_model_name: str
    rows: int
    judge_calls: int
    blocked: str | None = None

    @property
    def active(self) -> bool:
        return self.blocked is None and self.judge_calls > 0

    def message(self) -> str:
        if self.blocked is not None:
            return f"Continuous scoring is configured but skipped: {self.blocked}"
        return (
            f"Continuous scoring adds ~{self.judge_calls} verifier call(s) "
            f"({self.rows} rows x {len(CORRECTNESS_CRITERIA)} criteria x "
            f"{self.config.repeats} repeat(s), plus one capability probe); "
            "the budget ceiling covers them too."
        )


def plan_run(
    config: ContinuousScoringConfig,
    *,
    default_judge_model: str,
    shape: DatasetShape,
    judges_enabled: bool,
) -> ContinuousRunPlan | None:
    """The continuous plan for this run, or ``None`` when the path is off.

    Judge-free runs (smoke) stay free: the verifier is a judge and never
    opts a credential-free path into spend. The expectation contract is
    correctness's — a choice of ``expected_facts`` or ``expected_response``
    on every row — because the criteria compare against the expected
    answer, and a row without one has nothing to verify.
    """

    if not config.enabled or not judges_enabled:
        return None
    from aai_core.agentkit.catalog import _every_row_satisfies

    judge_model_name = config.judge_model or default_judge_model
    if not _every_row_satisfies(
        shape, set(_EXPECTATION_NEEDS), set(shape.expectation_keys)
    ):
        return ContinuousRunPlan(
            config=config,
            judge_model_name=judge_model_name,
            rows=shape.row_count,
            judge_calls=0,
            blocked=(
                "every row must provide expectations.expected_facts or "
                "expectations.expected_response for the verifier to score "
                "against"
            ),
        )
    calls = shape.row_count * len(CORRECTNESS_CRITERIA) * config.repeats + 1
    return ContinuousRunPlan(
        config=config,
        judge_model_name=judge_model_name,
        rows=shape.row_count,
        judge_calls=calls,
    )


@dataclass
class ActiveContinuousScoring:
    """The built continuous path for one run: scorers, telemetry, teardown."""

    plan: ContinuousRunPlan
    verifier: ContinuousVerifier | None
    scorers: list[Any]
    fallback: bool
    warnings: list[str]
    _owned_context: Any = None

    def parameters(self) -> dict[str, str]:
        config = self.plan.config
        return {
            "continuous_scoring_version": str(CONTINUOUS_SCORING_VERSION),
            "continuous_granularity": str(config.granularity),
            "continuous_repeats": str(config.repeats),
            "continuous_criteria": str(len(CORRECTNESS_CRITERIA)),
            "continuous_judge_model": self.plan.judge_model_name,
            "continuous_low_mass_threshold": f"{config.low_mass_threshold:g}",
        }

    def tags(self) -> dict[str, str]:
        return {
            "aai.continuous_scoring": (
                "fallback-discrete" if self.fallback else "logprob-weighted"
            ),
            "aai.continuous_judge_model": self.plan.judge_model_name,
        }

    def finalize(
        self,
        metric_samples: Mapping[str, tuple[float | None, ...]],
    ) -> tuple[dict[str, float], list[str]]:
        """Instrument metrics and warnings, read once after the run."""

        metrics: dict[str, float] = {
            "continuous/fallback": 1.0 if self.fallback else 0.0
        }
        warnings = list(self.warnings)
        if self.verifier is None:
            return metrics, warnings
        stats = self.verifier.stats
        metrics.update(stats.metrics())
        for metric_key, samples_key in (
            (f"{CONTINUOUS_SCORER_NAME}/tie_rate", f"{CONTINUOUS_SCORER_NAME}/mean"),
            ("correctness/tie_rate", "correctness/mean"),
        ):
            values = metric_samples.get(samples_key)
            rate = tie_rate(values) if values is not None else None
            if rate is not None:
                metrics[metric_key] = rate
        if stats.low_mass_calls:
            warnings.append(
                f"{stats.low_mass_calls} of {stats.calls} continuous verifier "
                "call(s) kept less than "
                f"{self.plan.config.low_mass_threshold:g} probability mass on "
                "score tokens - the prompt is not reliably steering the "
                "verifier to the scale"
            )
        if stats.invalid_calls:
            warnings.append(
                f"{stats.invalid_calls} continuous verifier call(s) produced "
                "no score token at all; those judgments were skipped"
            )
        return metrics, warnings

    def close(self) -> None:
        if self._owned_context is not None:
            self._owned_context.close()
            self._owned_context = None


def activate_run(
    plan: ContinuousRunPlan,
    *,
    settings: PlatformSettings,
    mlflow_module: Any,
    model: Any | None = None,
) -> ActiveContinuousScoring:
    """Resolve the verifier model, probe for logprobs, build the scorer.

    A backend that returns no top logprobs — the Anthropic API among them —
    degrades to the already-running discrete path with a clear warning
    instead of failing the run; every other provider failure propagates.
    ``model`` injects a caller-owned verifier (tests, notebooks); otherwise
    the logical name resolves through the governed provider configuration.
    """

    owned_context: Any = None
    if model is None:
        from aai_core.context import PlatformContext
        from aai_core.providers.types import ProviderConfigurationError

        owned_context = PlatformContext(settings)
        try:
            model = owned_context.providers.model(plan.judge_model_name)
        except ProviderConfigurationError as error:
            owned_context.close()
            raise ConfigError(
                f"continuous scoring cannot resolve verifier model "
                f"{plan.judge_model_name!r}: {error}",
                remediation=(
                    "Point scorers.continuous.judge_model at a "
                    "providers.models entry in aai-platform.yml (an Azure "
                    "OpenAI deployment behind azure_apim, or a "
                    "Databricks-served model that returns logprobs)."
                ),
            ) from error
    try:
        supported = detect_logprob_support(model)
    except Exception:
        if owned_context is not None:
            owned_context.close()
        raise
    if not supported:
        message = (
            f"the verifier model {plan.judge_model_name!r} returned no top "
            "logprobs (the Anthropic API never does; some gateways drop the "
            "parameter), so continuous scoring falls back to the discrete "
            "path for this run"
        )
        _LOGGER.warning("continuous scoring: %s", message)
        if owned_context is not None:
            owned_context.close()
        return ActiveContinuousScoring(
            plan=plan,
            verifier=None,
            scorers=[],
            fallback=True,
            warnings=[f"continuous scoring: {message}"],
        )
    verifier = ContinuousVerifier(
        model=model,
        granularity=plan.config.granularity,
        repeats=plan.config.repeats,
        low_mass_threshold=plan.config.low_mass_threshold,
    )
    return ActiveContinuousScoring(
        plan=plan,
        verifier=verifier,
        scorers=[_build_pointwise_scorer(verifier, mlflow_module)],
        fallback=False,
        warnings=[],
        _owned_context=owned_context,
    )


def _build_pointwise_scorer(verifier: ContinuousVerifier, mlflow_module: Any) -> Any:
    scorer_decorator = mlflow_module.genai.scorers.scorer

    @scorer_decorator(name=CONTINUOUS_SCORER_NAME)
    def correctness_continuous(
        inputs: Any = None,
        outputs: Any = None,
        expectations: Mapping[str, Any] | None = None,
    ) -> float | list[Any]:
        request = _request_text(inputs)
        expected = _expected_text(expectations)
        response = outputs if isinstance(outputs, str) else _output_text(outputs)
        if request is None or expected is None or response is None:
            # Outside the contract (or nothing was answered): skip with
            # MLflow's empty-feedback convention rather than inventing a
            # zero, exactly like the retrieval wrapper.
            return []
        result = verifier.score_response(
            request=request, response=response, expected=expected
        )
        if result is None:
            return []
        return result.continuous

    return correctness_continuous


def _request_text(inputs: Any) -> str | None:
    if isinstance(inputs, str):
        return inputs.strip() or None
    if isinstance(inputs, Mapping):
        if not inputs:
            return None
        if len(inputs) == 1:
            value = next(iter(inputs.values()))
            return str(value).strip() or None
        return json.dumps(
            {str(key): inputs[key] for key in sorted(inputs, key=str)},
            ensure_ascii=False,
            default=str,
        )
    return None


def _expected_text(expectations: Mapping[str, Any] | None) -> str | None:
    if not isinstance(expectations, Mapping):
        return None
    expected = expectations.get("expected_response")
    if isinstance(expected, str) and expected.strip():
        return expected.strip()
    facts = expectations.get("expected_facts")
    if isinstance(facts, Sequence) and not isinstance(facts, (str, bytes)):
        joined = "; ".join(str(fact).strip() for fact in facts if str(fact).strip())
        return joined or None
    return None


# --- Graded candidates for the granularity/repeats sweep --------------------


@dataclass(frozen=True)
class GradedCandidate:
    """One synthetic candidate answer with a known quality rank (higher is
    better). The sweep measures whether a scorer configuration recovers
    this ordering — accuracy against a known reference, not just
    instrument-vs-instrument consistency."""

    rank: int
    kind: str
    text: str


_FABRICATED_RIDER = (
    "Note that this guidance was withdrawn last year and no longer applies."
)


def graded_candidates(expected: str, *, wrong: str) -> tuple[GradedCandidate, ...]:
    """Four deterministic candidates per gold row, best to worst.

    verbatim (3) — the expected answer itself; paraphrase (2) — the same
    content restated, still fully correct; degraded (1) — half the content
    plus a fabricated contradiction; wrong (0) — another row's answer,
    fluent but off-question. Deterministic on purpose: the reference
    ordering must not depend on a generator model.
    """

    parts = [part.strip() for part in expected.replace("; ", ". ").split(". ")]
    parts = [part.rstrip(".") for part in parts if part]
    paraphrase = "In short: " + ". ".join(reversed(parts)) + "."
    words = expected.split()
    degraded = " ".join(words[: max(1, len(words) // 2)]).rstrip(".,;")
    degraded = f"{degraded}. {_FABRICATED_RIDER}"
    return (
        GradedCandidate(rank=3, kind="verbatim", text=expected),
        GradedCandidate(rank=2, kind="paraphrase", text=paraphrase),
        GradedCandidate(rank=1, kind="degraded", text=degraded),
        GradedCandidate(rank=0, kind="wrong", text=wrong),
    )
