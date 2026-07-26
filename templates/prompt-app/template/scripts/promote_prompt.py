"""Promote an evaluated prompt version to the candidate or production alias.

Production deploys load the `production` alias (src/app/assistant.py), so a
version must be promoted before the first prod deploy. Run this only after
the release gate (evals/evaluate.py) passed for exactly this version.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from aai_core import bootstrap
from aai_core.prompts import PromptManager
from app.config import PROMPT_NAME

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, type=int)
    parser.add_argument(
        "--to", required=True, choices=["candidate", "production"], dest="alias"
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
