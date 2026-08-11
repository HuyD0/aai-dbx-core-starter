"""Promote an evaluated prompt version to the validation or production alias.

Production deploys load the `production` alias, so a version must be promoted
before the first prod deploy. The release gate (evals/evaluate.py) records a
decision run for the exact version; this script requires that run and
promotion verifies its persisted evidence.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from aai_core import bootstrap
from app.config import PROMPT_NAME

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, type=int)
    parser.add_argument(
        "--to", required=True, choices=["validation", "production"], dest="alias"
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
