"""Deterministic, credential-free release-gate checks for pull-request CI.

Pull requests must stay credential-free, so the LLM-judge evaluation cannot
run there. These checks catch the failures that do not need a model: broken
gate configuration, missing or placeholder evaluation cases, and a corrupt
baseline. The full gate (evals/evaluate.py) runs on the credentialed path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from aai_core.evaluation import QualityThreshold

ROOT = Path(__file__).resolve().parents[1]
MIN_CASES = 10
REQUIRED_GATED_METRICS = (
    "safety/mean",
    "retrieval_groundedness/mean",
    "relevance_to_query/mean",
)
PLACEHOLDER_MARKERS = ("replace this", "replace-with", "todo", "changeme")


def main() -> int:
    failures: list[str] = []

    config = json.loads((ROOT / "evals" / "gate_config.json").read_text("utf-8"))
    thresholds = [QualityThreshold(**threshold) for threshold in config["thresholds"]]
    gated = {threshold.metric for threshold in thresholds}
    for metric in REQUIRED_GATED_METRICS:
        if metric not in gated:
            failures.append(f"gate_config.json does not gate {metric}")

    cases = json.loads(
        (ROOT / "evals" / "data" / "release_cases.json").read_text("utf-8")
    )
    if len(cases) < MIN_CASES:
        failures.append(
            f"release_cases.json has {len(cases)} cases; keep at least "
            f"{MIN_CASES} (grow toward 150+ as the application matures)"
        )
    for index, case in enumerate(cases):
        question = case.get("inputs", {}).get("question", "")
        expected = case.get("expectations", {}).get("expected_response", "")
        if not isinstance(question, str) or not question.strip():
            failures.append(f"case {index} has no inputs.question")
        if not isinstance(expected, str) or not expected.strip():
            failures.append(f"case {index} has no expectations.expected_response")
        text = f"{question} {expected}".lower()
        if any(marker in text for marker in PLACEHOLDER_MARKERS):
            failures.append(f"case {index} still contains placeholder text")

    baseline = ROOT / "evals" / "baseline.json"
    if baseline.exists():
        metrics = json.loads(baseline.read_text("utf-8")).get("metrics")
        if not isinstance(metrics, dict) or not metrics:
            failures.append("baseline.json exists but has no metrics mapping")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print(f"offline release-gate checks passed ({len(cases)} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
