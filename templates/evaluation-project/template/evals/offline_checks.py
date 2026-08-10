"""Tier-1 gate: deterministic checks plus a code-scorer evaluation, offline.

Runs in pull-request CI with zero credentials: validates configuration and
datasets, then scores the recorded answer sheet with the deterministic code
scorers and applies THEIR thresholds through the shared gate engine. Judge
metrics cannot execute here, but this tier verifies that every gated metric is
categorized and that report-only judges are not accidentally promoted into a
release threshold. Judges execute only in evals/evaluate.py.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from aai_core.evaluation import GatePolicy, MetricRule, apply_gate

ROOT = Path(__file__).resolve().parents[1]
MIN_GOLDEN_CASES = 10
PLACEHOLDER_MARKERS = ("replace this", "replace-with", "todo", "changeme")


def load_gate() -> tuple[list[MetricRule], set[str], set[str], set[str]]:
    config = json.loads((ROOT / "evals" / "gate_config.json").read_text("utf-8"))
    thresholds = [MetricRule(**threshold) for threshold in config["thresholds"]]
    return (
        thresholds,
        set(config["code_metrics"]),
        set(config["judge_metrics"]),
        set(config["report_only_judge_metrics"]),
    )


def main() -> int:
    sys.path.insert(0, str(ROOT / "src"))
    from app import targets
    from app.scorers import score_all

    thresholds, code_metrics, judge_metrics, report_only_judge_metrics = load_gate()
    failures = _gate_configuration_failures(
        thresholds,
        code_metrics,
        judge_metrics,
        report_only_judge_metrics,
    )
    golden, predict, data_failures = _reviewed_data(targets)
    failures.extend(data_failures)

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1

    # Deterministic code-scorer evaluation over the answer sheet.
    totals: dict[str, float] = {}
    for case in golden:
        question = case["inputs"]["question"]
        scores = score_all(predict(question), case["expectations"])
        for name, value in scores.items():
            totals[f"{name}/mean"] = totals.get(f"{name}/mean", 0.0) + value
    metrics = {name: value / len(golden) for name, value in totals.items()}
    code_thresholds = [t for t in thresholds if t.metric in code_metrics]
    report = apply_gate(
        metrics,
        policy=GatePolicy(
            rules=tuple(code_thresholds),
            # Tier 1 enforces absolute deterministic thresholds only. The
            # governed tier-2 run owns comparison to the approved baseline.
            allow_missing_regression_baseline=True,
        ),
    )
    print({"tier": "offline", "metrics": report.metrics})
    report.require_passed()
    print(f"offline release-gate checks passed ({len(golden)} golden cases)")
    return 0


def _gate_configuration_failures(
    thresholds: list[MetricRule],
    code_metrics: set[str],
    judge_metrics: set[str],
    report_only_judge_metrics: set[str],
) -> list[str]:
    failures: list[str] = []
    gated = {threshold.metric for threshold in thresholds}
    gated_categories = code_metrics | judge_metrics
    all_categories = [code_metrics, judge_metrics, report_only_judge_metrics]
    overlap = set()
    for index, category in enumerate(all_categories):
        for other in all_categories[index + 1 :]:
            overlap.update(category & other)
    if overlap:
        failures.append(
            "metrics must appear in exactly one code, judge, or report-only "
            f"category: {sorted(overlap)}"
        )
    for metric in gated_categories:
        if metric not in gated:
            failures.append(f"categorized metric {metric} is not gated")
    for metric in gated - gated_categories:
        failures.append(f"gated metric {metric} is not categorized")
    for metric in report_only_judge_metrics & gated:
        failures.append(f"report-only judge metric {metric} must not be gated")
    return failures


def _reviewed_data(
    targets: Any,
) -> tuple[list[dict[str, Any]], Callable[[str], str], list[str]]:
    failures: list[str] = []
    golden = targets.load_evaluation_cases(
        ROOT / "evals" / "data" / "golden_cases.json"
    )
    if len(golden) < MIN_GOLDEN_CASES:
        failures.append(
            f"golden_cases.json has {len(golden)} cases; keep at least "
            f"{MIN_GOLDEN_CASES} (grow toward 150+ as the suite matures)"
        )
    for index, case in enumerate(golden):
        question = case.get("inputs", {}).get("question", "")
        expected = case.get("expectations", {}).get("expected_response", "")
        if not question.strip() or not expected.strip():
            failures.append(f"golden case {index} is missing question/expectation")
        elif any(m in f"{question} {expected}".lower() for m in PLACEHOLDER_MARKERS):
            failures.append(f"golden case {index} still contains placeholder text")

    targets.load_edge_cases(ROOT / "evals" / "data" / "edge_cases.json")
    predict = targets.answer_sheet_predict_fn(
        ROOT / "evals" / "data" / "answer_sheet.json",
        expected_cases=golden,
    )
    return golden, predict, failures


if __name__ == "__main__":
    raise SystemExit(main())
