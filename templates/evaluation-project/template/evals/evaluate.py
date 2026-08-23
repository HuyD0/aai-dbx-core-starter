"""Tier-2 gate: the full suite with code scorers and LLM judges.

Runs on the credentialed path, locally with workspace authentication or as
the bundle's ``release_gate`` job. AgentKit scores every row, compares the run
with the recorded baseline, applies every threshold, and writes governed
evidence to MLflow.

    python evals/evaluate.py                      # score and compare
    python evals/evaluate.py --update-baseline    # establish this baseline

This wrapper preserves the path used by the bundle and Makefile while routing
behavior through ``agentkit eval``. Exit codes are stable: 0 passed, 2 a
threshold failed, and 1 a configuration or runtime error.
"""

from __future__ import annotations

import argparse
import sys
from importlib.util import find_spec
from pathlib import Path

from aai_core.agentkit.cli import main

ROOT = Path(__file__).resolve().parents[1]


def build_arguments(argv: list[str] | None = None) -> list[str]:
    """Translate the stable template wrapper flags to AgentKit arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["answer-sheet", "endpoint", "live"],
        default=None,
        help="answer-sheet replays recorded answers; endpoint/live calls the "
        "agent configured in agentkit.yaml.",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="record this run as the baseline future runs compare against.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="skip the spend confirmation (for non-interactive jobs).",
    )
    parser.add_argument(
        "--decision",
        choices=["adopt", "reject", "inconclusive"],
        default=None,
        help="the conclusion this comparison supports.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Unity Catalog model to score (supplied by the deployment job).",
    )
    parser.add_argument(
        "--model-version",
        default=None,
        help="model version to score (supplied by the deployment job).",
    )
    parsed = parser.parse_args(argv)

    if bool(parsed.model_name) != bool(parsed.model_version):
        parser.error("--model-name and --model-version must be given together")

    arguments = ["eval"]
    if parsed.yes:
        arguments.append("--yes")
    mode = parsed.mode
    if parsed.model_name:
        # Score exactly the version that triggered promotion. Falling back to
        # the configured agent could approve an unrelated target.
        arguments += [
            "--agent",
            f"models:/{parsed.model_name}/{parsed.model_version}",
        ]
        mode = mode or "live"
    if mode is not None:
        arguments += [
            "--mode",
            "live" if mode in {"endpoint", "live"} else "answer-sheet",
        ]
    if parsed.update_baseline:
        arguments.append("--establish-baseline")
    if parsed.decision is not None:
        arguments += ["--decision", parsed.decision]
    return arguments


def validate_bundled_inputs(arguments: list[str]) -> None:
    """Fail local case and answer-sheet drift before target or judge work."""

    # Imported lazily so tests can exercise this plain wrapper without a
    # rendered project's ``src`` directory on sys.path.
    from app.targets import validate_bundled_data

    include_answer_sheet = any(
        arguments[index : index + 2] == ["--mode", "answer-sheet"]
        for index in range(len(arguments) - 1)
    )
    validate_bundled_data(ROOT, include_answer_sheet=include_answer_sheet)


def publish_evidence_run_id(root: Path | None = None) -> str | None:
    """Hand the approval task the exact run this evaluation recorded.

    Searching MLflow for the newest run against a model version can select a
    concurrent or manual evaluation with different data or policy. A task
    value preserves the exact evidence identity. Writing it is best effort so
    local runs outside Databricks remain usable.
    """

    from aai_core.agentkit.results import load_latest_results

    directory = (root or Path.cwd()) / ".aai" / "agentkit" / "results"
    loaded = load_latest_results(directory)
    run_id = loaded[0].run_id if loaded else None
    if not run_id:
        return None
    # ``databricks.sdk.runtime`` creates a remote dbutils client when the
    # Databricks runtime namespace is absent. Importing it on a developer
    # machine can therefore authenticate and retry instead of failing fast.
    # The ``dbruntime`` module is the same boundary the certified SDK uses to
    # distinguish an in-runtime namespace from its local fallback.
    if find_spec("dbruntime") is None:
        return run_id
    try:
        from databricks.sdk.runtime import dbutils

        dbutils.jobs.taskValues.set(key="evidence_run_id", value=run_id)
    except Exception as error:  # noqa: BLE001 - no task values outside a job
        # Do not echo a provider exception message into job logs; exception
        # messages are not a trusted redaction boundary.
        print(
            "could not hand the run id to the approval task "
            f"({type(error).__name__})"
        )
    return run_id


if __name__ == "__main__":
    agentkit_arguments = build_arguments(sys.argv[1:])
    validate_bundled_inputs(agentkit_arguments)
    exit_code = main(agentkit_arguments)
    publish_evidence_run_id()
    raise SystemExit(exit_code)
