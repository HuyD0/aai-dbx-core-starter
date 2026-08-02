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
    parsed = parser.parse_args(argv)

    arguments = ["eval", "--yes"]
    if parsed.mode is not None:
        arguments += [
            "--mode",
            "live" if parsed.mode in {"endpoint", "live"} else "answer-sheet",
        ]
    if parsed.update_baseline:
        arguments.append("--establish-baseline")
    if parsed.decision is not None:
        arguments += ["--decision", parsed.decision]
    return arguments


if __name__ == "__main__":
    raise SystemExit(main(build_arguments(sys.argv[1:])))
