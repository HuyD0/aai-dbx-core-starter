"""Tier-1 gate: deterministic checks plus a code-scorer evaluation, offline.

Runs in pull-request CI with zero credentials: validates configuration and
datasets, then scores the recorded answer sheet with the deterministic code
scorers and applies THEIR thresholds through the shared gate engine. Judge
metrics gate only in the full run (evals/evaluate.py).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from aai_core.evaluation import QualityThreshold, apply_thresholds

ROOT = Path(__file__).resolve().parents[1]
MIN_GOLDEN_CASES = 10
PLACEHOLDER_MARKERS = ("replace this", "replace-with", "todo", "changeme")


def load_gate() -> tuple[list[QualityThreshold], set[str]]:
    config = json.loads((ROOT / "evals" / "gate_config.json").read_text("utf-8"))
    thresholds = [QualityThreshold(**threshold) for threshold in config["thresholds"]]
    return thresholds, set(config["code_metrics"])


def main() -> int:
    sys.path.insert(0, str(ROOT / "src"))
    from app.scorers import score_all

    failures: list[str] = []
    thresholds, code_metrics = load_gate()
    gated = {threshold.metric for threshold in thresholds}
    for metric in code_metrics:
        if metric not in gated:
            failures.append(f"code metric {metric} is not gated in gate_config.json")

    golden = json.loads(
        (ROOT / "evals" / "data" / "golden_cases.json").read_text("utf-8")
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

    edge = json.loads((ROOT / "evals" / "data" / "edge_cases.json").read_text("utf-8"))
    if not isinstance(edge, list):
        failures.append("edge_cases.json must be a list (capability suite)")

    sheet = json.loads(
        (ROOT / "evals" / "data" / "answer_sheet.json").read_text("utf-8")
    )
    answers = {record["question"]: record["answer"] for record in sheet}
    missing = [
        case["inputs"]["question"]
        for case in golden
        if case["inputs"]["question"] not in answers
    ]
    if missing:
        failures.append(f"answer_sheet.json lacks answers for: {missing}")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1

    # Deterministic code-scorer evaluation over the answer sheet.
    totals: dict[str, float] = {}
    for case in golden:
        question = case["inputs"]["question"]
        scores = score_all(answers[question], case["expectations"])
        for name, value in scores.items():
            totals[f"{name}/mean"] = totals.get(f"{name}/mean", 0.0) + value
    metrics = {name: value / len(golden) for name, value in totals.items()}
    code_thresholds = [t for t in thresholds if t.metric in code_metrics]
    report = apply_thresholds(metrics, code_thresholds)
    print({"tier": "offline", "metrics": report.metrics})
    report.require_passed()
    print(f"offline release-gate checks passed ({len(golden)} golden cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
