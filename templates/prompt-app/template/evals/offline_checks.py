"""Deterministic, credential-free release-gate checks for pull-request CI.

Validates gate configuration, the prompt source file, and the evaluation
cases without a model or workspace. The LLM-judge gate (evals/evaluate.py)
runs on the credentialed path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from aai_core.evaluation import MetricRule

ROOT = Path(__file__).resolve().parents[1]
MIN_CASES = 10
REQUIRED_GATED_METRICS = (
    "safety/mean",
    "correctness/mean",
    "relevance_to_query/mean",
)
PLACEHOLDER_MARKERS = ("replace this", "replace-with", "todo", "changeme")


def main() -> int:  # noqa: C901 - linear, independent contract assertions
    failures: list[str] = []

    config = json.loads((ROOT / "evals" / "gate_config.json").read_text("utf-8"))
    thresholds = [MetricRule(**threshold) for threshold in config["thresholds"]]
    gated = {threshold.metric for threshold in thresholds}
    for metric in REQUIRED_GATED_METRICS:
        if metric not in gated:
            failures.append(f"gate_config.json does not gate {metric}")

    prompt = json.loads((ROOT / "prompts" / "system_prompt.json").read_text("utf-8"))
    roles = [message.get("role") for message in prompt.get("messages", [])]
    if "system" not in roles or "user" not in roles:
        failures.append("system_prompt.json needs system and user messages")
    user_content = " ".join(
        message.get("content", "")
        for message in prompt.get("messages", [])
        if message.get("role") == "user"
    )
    if "{{question}}" not in user_content:
        failures.append("system_prompt.json user message must use {{question}}")

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
