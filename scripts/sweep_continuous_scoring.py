"""Sweep continuous-scoring granularity and repeats on a gold dataset.

The experiment that settles the continuous-scoring configuration
(docs/continuous-scoring.md): for every (granularity, repeats) combination
it scores graded candidate answers — built deterministically from each gold
row's expected answer, so their quality ordering is *known* — and reports
how well each configuration recovers that ordering (Kendall's tau-b),
alongside each instrument's tie rate, the normalization-mass health of the
verifier calls, and what the combination cost in calls and tokens.

Two instruments are read from the *same* verifier calls, so the comparison
adds no spend: the logprob-weighted continuous score, and the discrete
parse (the top token alone — what a parse-the-emitted-label judge sees).

The verifier must return top logprobs (Azure OpenAI behind ``azure_apim``,
or a Databricks-served model that supports ``top_logprobs``; the Anthropic
API never does). The script probes once and refuses early otherwise.

Usage:

    # credential-free plumbing check with a simulated verifier
    python scripts/sweep_continuous_scoring.py --simulate

    # the real experiment, against the configured platform judge
    python scripts/sweep_continuous_scoring.py \
        --dataset evals/data/golden_cases.json --model judge-model

Exit codes follow the agentkit contract: 0 success, 1 error.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aai_core.agentkit.continuous import (  # noqa: E402
    ContinuousVerifier,
    GradedCandidate,
    VerifierStats,
    detect_logprob_support,
    graded_candidates,
    kendall_tau_b,
    score_labels,
    tie_rate,
)
from aai_core.providers.types import ModelResponse  # noqa: E402

DEFAULT_DATASET = "templates/evaluation-project/template/evals/data/golden_cases.json"
DEFAULT_GRANULARITIES = (5, 10, 20)
DEFAULT_REPEATS = (1, 2, 4)
_SCALE_PATTERN = re.compile(r"on a (\d+)-letter scale")
_RESPONSE_PATTERN = re.compile(
    r"Response to rate: (?P<text>.*?)\n\nAnswer with", re.DOTALL
)
_REQUEST_PATTERN = re.compile(r"Request: (?P<text>.*?)\n\nExpected answer:", re.DOTALL)


@dataclass(frozen=True)
class GoldItem:
    request: str
    expected: str
    candidates: tuple[GradedCandidate, ...]


@dataclass(frozen=True)
class ComboReport:
    granularity: int
    repeats: int
    tau_continuous: float | None
    tau_discrete: float | None
    tie_rate_continuous: float | None
    tie_rate_discrete: float | None
    normalization_mass_mean: float | None
    low_mass_rate: float | None
    rows_scored: int
    rows_skipped: int
    judge_calls: int
    input_tokens: int
    output_tokens: int

    def as_dict(self) -> dict[str, Any]:
        return dict(vars(self))


class SimulatedVerifierModel:
    """Deterministic logprob-shaped verifier for ``--simulate`` runs.

    Scores the graded candidates by their registered rank with a fixed
    spread, so the report machinery can be exercised — and its output
    format seen — with zero credentials and zero spend. It is a plumbing
    check, not evidence: repeats of an identical prompt return identical
    distributions, unlike a real backend.
    """

    def __init__(self) -> None:
        # Keyed by (request, response text): the same text is a right
        # answer to its own question and a wrong answer to another's, so
        # text alone cannot carry the rank.
        self.rank_by_key: dict[tuple[str, str], int] = {}
        self.calls = 0

    def register(self, request: str, candidate: GradedCandidate) -> None:
        self.rank_by_key[(request, candidate.text)] = candidate.rank

    def generate(self, messages: list[dict[str, str]], **options: Any) -> ModelResponse:
        self.calls += 1
        prompt = messages[-1]["content"]
        scale_match = _SCALE_PATTERN.search(prompt)
        granularity = int(scale_match.group(1)) if scale_match else 20
        labels = score_labels(granularity)
        response_match = _RESPONSE_PATTERN.search(prompt)
        request_match = _REQUEST_PATTERN.search(prompt)
        rank = 2
        if response_match and request_match:
            rank = self.rank_by_key.get(
                (
                    request_match.group("text").strip(),
                    response_match.group("text").strip(),
                ),
                2,
            )
        center = round(rank / 3 * (granularity - 1))
        pairs = [(labels[center], math.log(0.7))]
        if center > 0:
            pairs.append((labels[center - 1], math.log(0.15)))
        if center + 1 < granularity:
            pairs.append((labels[center + 1], math.log(0.1)))
        alternatives = [
            SimpleNamespace(token=token, logprob=logprob) for token, logprob in pairs
        ]
        raw = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=pairs[0][0]),
                    logprobs=SimpleNamespace(
                        content=[
                            SimpleNamespace(
                                token=pairs[0][0], top_logprobs=alternatives
                            )
                        ]
                    ),
                )
            ]
        )
        return ModelResponse(
            content=pairs[0][0],
            provider="simulated",
            logical_name="simulated-verifier",
            model="simulated",
            latency_ms=0.1,
            usage={"prompt_tokens": len(prompt) // 4, "completion_tokens": 1},
            raw=raw,
        )


def load_gold_items(path: Path, *, max_rows: int | None) -> list[GoldItem]:
    """Gold rows -> (request, expected, four graded candidates each).

    The "wrong" candidate is the *next* row's expected answer — fluent,
    plausible, and answering a different question — so the reference
    ordering never depends on a generator model.
    """

    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, list) or not loaded:
        raise SystemExit(f"error: {path} must be a non-empty JSON list of rows")
    expected_texts: list[str] = []
    requests: list[str] = []
    for index, row in enumerate(loaded):
        if not isinstance(row, dict):
            raise SystemExit(f"error: row {index} in {path} is not an object")
        inputs = row.get("inputs") or {}
        request = str(next(iter(inputs.values()), "")).strip()
        expectations = row.get("expectations") or {}
        expected = str(expectations.get("expected_response") or "").strip()
        if not expected:
            facts = expectations.get("expected_facts") or []
            expected = "; ".join(str(fact) for fact in facts).strip()
        if not request or not expected:
            continue
        requests.append(request)
        expected_texts.append(expected)
    if len(expected_texts) < 2:
        raise SystemExit(
            f"error: {path} needs at least 2 rows with inputs and "
            "expectations.expected_response/expected_facts"
        )
    if max_rows is not None:
        requests = requests[:max_rows]
        expected_texts = expected_texts[: max(2, max_rows)]
    items = []
    for index, (request, expected) in enumerate(
        zip(requests, expected_texts, strict=False)
    ):
        wrong = expected_texts[(index + 1) % len(expected_texts)]
        items.append(
            GoldItem(
                request=request,
                expected=expected,
                candidates=graded_candidates(expected, wrong=wrong),
            )
        )
    return items


def sweep_combination(
    model: Any,
    items: list[GoldItem],
    *,
    granularity: int,
    repeats: int,
    low_mass_threshold: float,
) -> ComboReport:
    stats = VerifierStats()
    verifier = ContinuousVerifier(
        model=model,
        granularity=granularity,
        repeats=repeats,
        low_mass_threshold=low_mass_threshold,
        stats=stats,
    )
    taus_continuous: list[float] = []
    taus_discrete: list[float] = []
    row_ties_continuous: list[float] = []
    row_ties_discrete: list[float] = []
    skipped = 0
    for item in items:
        ranks: list[float] = []
        continuous: list[float] = []
        discrete: list[float] = []
        complete = True
        for candidate in item.candidates:
            result = verifier.score_response(
                request=item.request,
                response=candidate.text,
                expected=item.expected,
            )
            if result is None:
                complete = False
                break
            ranks.append(float(candidate.rank))
            continuous.append(result.continuous)
            discrete.append(result.discrete)
        if not complete:
            skipped += 1
            continue
        tau_continuous = kendall_tau_b(ranks, continuous)
        tau_discrete = kendall_tau_b(ranks, discrete)
        if tau_continuous is not None:
            taus_continuous.append(tau_continuous)
        if tau_discrete is not None:
            taus_discrete.append(tau_discrete)
        continuous_ties = tie_rate(continuous)
        discrete_ties = tie_rate(discrete)
        if continuous_ties is not None:
            row_ties_continuous.append(continuous_ties)
        if discrete_ties is not None:
            row_ties_discrete.append(discrete_ties)
    metrics = stats.metrics()
    return ComboReport(
        granularity=granularity,
        repeats=repeats,
        tau_continuous=fmean(taus_continuous) if taus_continuous else None,
        tau_discrete=fmean(taus_discrete) if taus_discrete else None,
        tie_rate_continuous=(
            fmean(row_ties_continuous) if row_ties_continuous else None
        ),
        tie_rate_discrete=fmean(row_ties_discrete) if row_ties_discrete else None,
        normalization_mass_mean=metrics.get("continuous/normalization_mass_mean"),
        low_mass_rate=metrics.get("continuous/low_mass_rate"),
        rows_scored=len(items) - skipped,
        rows_skipped=skipped,
        judge_calls=stats.calls,
        input_tokens=stats.input_tokens,
        output_tokens=stats.output_tokens,
    )


def render_reports(reports: list[ComboReport]) -> str:
    header = (
        "granularity",
        "K",
        "tau_cont",
        "tau_disc",
        "ties_cont",
        "ties_disc",
        "mass",
        "low_mass",
        "calls",
        "tokens",
    )

    def cell(value: float | None) -> str:
        return "-" if value is None else f"{value:.3f}"

    rows = [header]
    for report in reports:
        rows.append(
            (
                str(report.granularity),
                str(report.repeats),
                cell(report.tau_continuous),
                cell(report.tau_discrete),
                cell(report.tie_rate_continuous),
                cell(report.tie_rate_discrete),
                cell(report.normalization_mass_mean),
                cell(report.low_mass_rate),
                str(report.judge_calls),
                str(report.input_tokens + report.output_tokens),
            )
        )
    widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    lines = [
        "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip()
        for row in rows
    ]
    lines.append("")
    lines.append(
        "tau_*: mean Kendall tau-b against the known candidate ordering "
        "(1.0 = perfect ranking); ties_*: mean within-row tie rate. "
        "_cont reads the logprob-weighted score, _disc the same calls' "
        "top-token parse."
    )
    return "\n".join(lines)


def resolve_model(logical_name: str, platform_config: str | None) -> Any:
    from aai_core.context import PlatformContext
    from aai_core.runtime import PlatformSettings

    settings = PlatformSettings.load(platform_config)
    return PlatformContext(settings).providers.model(logical_name)


def _parse_int_list(text: str, *, flag: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part) for part in text.split(",") if part.strip())
    except ValueError as error:
        raise SystemExit(f"error: {flag} must be comma-separated integers") from error
    if not values:
        raise SystemExit(f"error: {flag} names no values")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep continuous-scoring granularity and repeats on a gold "
            "dataset; report ranking agreement and tie rate per combination."
        )
    )
    parser.add_argument(
        "--dataset",
        default=str(ROOT / DEFAULT_DATASET),
        help="gold dataset (JSON rows with inputs + expectations)",
    )
    parser.add_argument(
        "--model",
        default="judge-model",
        help="logical verifier model from aai-platform.yml",
    )
    parser.add_argument(
        "--platform-config",
        default=None,
        help="path to aai-platform.yml (default: ordinary discovery)",
    )
    parser.add_argument(
        "--granularities", default=",".join(map(str, DEFAULT_GRANULARITIES))
    )
    parser.add_argument("--repeats", default=",".join(map(str, DEFAULT_REPEATS)))
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--low-mass-threshold", type=float, default=0.5)
    parser.add_argument("--output", default=None, help="write the JSON report here")
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="use the built-in deterministic verifier (no credentials, no spend)",
    )
    arguments = parser.parse_args(argv)

    granularities = _parse_int_list(arguments.granularities, flag="--granularities")
    repeats_values = _parse_int_list(arguments.repeats, flag="--repeats")
    items = load_gold_items(Path(arguments.dataset), max_rows=arguments.max_rows)

    if arguments.simulate:
        simulated = SimulatedVerifierModel()
        for item in items:
            for candidate in item.candidates:
                simulated.register(item.request, candidate)
        model: Any = simulated
    else:
        model = resolve_model(arguments.model, arguments.platform_config)
        if not detect_logprob_support(model):
            print(
                f"error: verifier model {arguments.model!r} returned no top "
                "logprobs. Continuous scoring needs an Azure OpenAI "
                "deployment (azure_apim) or a Databricks-served model that "
                "supports top_logprobs; the Anthropic API never exposes "
                "them.",
                file=sys.stderr,
            )
            return 1

    calls_per_combo = len(items) * 4 * 3  # rows x candidates x criteria
    print(
        f"Sweeping {len(granularities)}x{len(repeats_values)} combinations "
        f"over {len(items)} gold rows (4 graded candidates each); about "
        f"{sum(calls_per_combo * k for k in repeats_values) * len(granularities)} "
        "verifier calls in total.\n"
    )
    reports = []
    for granularity in granularities:
        for repeats in repeats_values:
            reports.append(
                sweep_combination(
                    model,
                    items,
                    granularity=granularity,
                    repeats=repeats,
                    low_mass_threshold=arguments.low_mass_threshold,
                )
            )
    print(render_reports(reports))
    skipped = sum(report.rows_skipped for report in reports)
    if skipped:
        print(
            f"\nwarning: {skipped} row-combination(s) produced no valid "
            "score token and were skipped; check low_mass/invalid rates."
        )
    if arguments.output:
        payload = {
            "dataset": str(arguments.dataset),
            "model": "simulated" if arguments.simulate else arguments.model,
            "rows": len(items),
            "combinations": [report.as_dict() for report in reports],
        }
        Path(arguments.output).write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nJSON report written to {arguments.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
