"""Deterministic, credential-free release-gate checks for pull-request CI.

Validates gate configuration, the prompt source, the evaluation cases, and
every exact expected tool name/argument object. The judge metrics and traced
tool calls gate in evals/evaluate.py on the credentialed path.
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
    "correctness/mean",
    "tool_call_correctness/mean",
)
PLACEHOLDER_MARKERS = ("replace this", "replace-with", "todo", "changeme")


def main() -> int:
    sys.path.insert(0, str(ROOT / "src"))
    from app.tools import build_registry

    failures: list[str] = []

    config = json.loads((ROOT / "evals" / "gate_config.json").read_text("utf-8"))
    thresholds = [QualityThreshold(**threshold) for threshold in config["thresholds"]]
    gated = {threshold.metric for threshold in thresholds}
    for metric in REQUIRED_GATED_METRICS:
        if metric not in gated:
            failures.append(f"gate_config.json does not gate {metric}")

    prompt = json.loads((ROOT / "prompts" / "system_prompt.json").read_text("utf-8"))
    roles = [message.get("role") for message in prompt.get("messages", [])]
    if "system" not in roles or "user" not in roles:
        failures.append("system_prompt.json needs system and user messages")

    registered = set(build_registry().names())
    cases = json.loads(
        (ROOT / "evals" / "data" / "release_cases.json").read_text("utf-8")
    )
    if len(cases) < MIN_CASES:
        failures.append(
            f"release_cases.json has {len(cases)} cases; keep at least "
            f"{MIN_CASES} (grow toward 150+ as the application matures)"
        )
    tool_cases = 0
    for index, case in enumerate(cases):
        question = case.get("inputs", {}).get("question", "")
        expectations = case.get("expectations", {})
        expected = expectations.get("expected_response", "")
        if not question.strip() or not str(expected).strip():
            failures.append(f"case {index} is missing question/expectation")
            continue
        if any(m in f"{question} {expected}".lower() for m in PLACEHOLDER_MARKERS):
            failures.append(f"case {index} still contains placeholder text")
        expected_tool_calls = expectations.get("expected_tool_calls")
        if expected_tool_calls is None:
            failures.append(f"case {index} is missing expectations.expected_tool_calls")
            continue
        if not isinstance(expected_tool_calls, list):
            failures.append(
                f"case {index} expectations.expected_tool_calls must be a list"
            )
            continue
        for call_index, call in enumerate(expected_tool_calls):
            label = f"case {index} expected_tool_calls[{call_index}]"
            if not isinstance(call, dict):
                failures.append(f"{label} must be an object")
                continue
            if set(call) != {"name", "arguments"}:
                failures.append(f"{label} must contain exactly name and arguments")
            name = call.get("name")
            if not isinstance(name, str) or not name:
                failures.append(f"{label}.name must be a non-empty string")
            elif name not in registered:
                failures.append(f"{label} references unregistered tool {name!r}")
            if not isinstance(call.get("arguments"), dict):
                failures.append(f"{label}.arguments must be an object")
        if expected_tool_calls:
            tool_cases += 1
    if cases and tool_cases == 0:
        failures.append("no case exercises a tool trajectory")
    if cases and tool_cases == len(cases):
        failures.append("no case covers the no-tool path")

    baseline = ROOT / "evals" / "baseline.json"
    if baseline.exists():
        metrics = json.loads(baseline.read_text("utf-8")).get("metrics")
        if not isinstance(metrics, dict) or not metrics:
            failures.append("baseline.json exists but has no metrics mapping")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print(
        f"offline release-gate checks passed ({len(cases)} cases, "
        f"{tool_cases} with tool trajectories)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
