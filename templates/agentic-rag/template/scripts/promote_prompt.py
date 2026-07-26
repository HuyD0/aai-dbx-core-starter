"""Promote a governed prompt version to the candidate or production alias.

The application loads the ``production`` alias in prod environments and
``development`` elsewhere (src/app/agent.py). Nothing promotes automatically:
after the release gate (evals/evaluate.py) passes for a prompt version, a
human runs this script — first ``--to candidate``, then ``--to production``
once the release is approved. Evaluations should pin exact versions
(``prompts:/name/version``); aliases are deployment pointers only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from aai_core import bootstrap
from aai_core.prompts import PromptManager

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
        choices=["candidate", "production"],
        dest="alias",
        help="Alias to move to the given version.",
    )
    args = parser.parse_args()

    context = bootstrap(ROOT / "aai-platform.yml")
    prompts = PromptManager(
        context=context.tags,
        catalog=context.settings.catalog,
        schema=context.settings.schema,
    )
    prompts.set_alias(PROMPT_NAME, alias=args.alias, version=args.version)
    print({"name": PROMPT_NAME, "alias": args.alias, "version": args.version})


if __name__ == "__main__":
    main()
