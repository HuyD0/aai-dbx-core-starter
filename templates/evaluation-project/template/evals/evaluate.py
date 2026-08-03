"""Tier-2 gate: the full suite with code scorers AND LLM judges.

Runs on the credentialed path — locally with workspace auth, or as the
bundle's `release_gate` job so compute goes to the data. Scores every row,
compares against the recorded baseline, applies every threshold, and writes
the run to MLflow with its dataset, scorer, prompt, and model versions
attached.

    python evals/evaluate.py                      # score and compare
    python evals/evaluate.py --update-baseline    # record this run AS the baseline

Equivalent to `agentkit eval`; this wrapper keeps the file path the bundle
job and the Makefile refer to. Exit codes: 0 passed, 2 a threshold failed,
1 a configuration or runtime error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aai_core.agentkit.cli import main


def build_arguments(argv: list[str] | None = None) -> list[str]:
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

    arguments = ["eval", "--yes"]
    mode = parsed.mode
    if parsed.model_name:
        # Score the version that triggered promotion, whatever agentkit.yaml
        # names. Without this the gate could approve an unrelated target.
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


def publish_evidence_run_id(root: Path | None = None) -> str | None:
    """Hand the approval task the exact run this evaluation recorded.

    The approval task needs to name the run whose evidence supports it.
    Searching MLflow for the newest run against this model version would
    pick up a concurrent or manual evaluation of the same version instead,
    and point a reviewer at a different dataset, config, or gate. The run
    id travels as a Databricks task value, which is exact.

    Best effort on the *write* only: outside a job there is no task value
    to set, and a laptop run must not fail because of it. The approval
    task falls back to a search when the value is absent, and says so.
    """

    from aai_core.agentkit.results import load_latest_results

    directory = (root or Path.cwd()) / ".aai" / "agentkit" / "results"
    loaded = load_latest_results(directory)
    run_id = loaded[0].run_id if loaded else None
    if not run_id:
        return None
    try:
        from databricks.sdk.runtime import dbutils

        dbutils.jobs.taskValues.set(key="evidence_run_id", value=run_id)
    except Exception as error:  # noqa: BLE001 - no task values outside a job
        # Not fatal: the approval task falls back to searching for the run
        # and says that is what it did. Worth a line in the job log, since
        # it is why the approver sees the less exact message.
        print(f"could not hand the run id to the approval task: {error}")
    return run_id


if __name__ == "__main__":
    code = main(build_arguments(sys.argv[1:]))
    publish_evidence_run_id()
    raise SystemExit(code)
