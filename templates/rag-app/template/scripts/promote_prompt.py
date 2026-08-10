"""Promote a governed prompt version to the validation or production alias.

The application loads the ``production`` alias in prod environments and
``development`` elsewhere (src/app/rag.py). Nothing promotes automatically:
after the release gate (evals/evaluate.py) passes for a prompt version, a
human runs this script — first ``--to validation``, then ``--to production``
once the release is approved. The release gate emits a decision run for the
exact pinned version; this script requires that run and promotion verifies its
persisted evidence. Aliases remain deployment pointers only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from aai_core import bootstrap

ROOT = Path(__file__).resolve().parents[1]
PROMPT_NAME = "agent-system"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        required=True,
        type=int,
        help="Prompt version that passed the release gate.",
    )
    parser.add_argument(
        "--to",
        required=True,
        choices=["validation", "production"],
        dest="alias",
        help="Alias to move to the given version.",
    )
    parser.add_argument(
        "--decision-run-id",
        required=True,
        help="Finished decision run emitted by evals/evaluate.py.",
    )
    args = parser.parse_args()

    context = bootstrap(ROOT / "aai-platform.yml")
    context.prompts.promote(
        PROMPT_NAME,
        alias=args.alias,
        version=args.version,
        decision_run_id=args.decision_run_id,
    )
    print(
        {
            "name": PROMPT_NAME,
            "alias": args.alias,
            "version": args.version,
            "decision_run_id": args.decision_run_id,
        }
    )


if __name__ == "__main__":
    main()
