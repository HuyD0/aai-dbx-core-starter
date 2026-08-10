"""Tier-1 gate: deterministic checks plus code scorers, fully offline.

Pull-request CI runs this with zero credentials. The template first enforces
its strict stable-case-ID and answer-sheet contract, then ``agentkit smoke``
validates the configured dataset, selects versioned code scorers from the
shared registry, and applies their thresholds. Judge scorers run only in
``evals/evaluate.py`` on the credentialed path.

Exit codes are the CI contract: 0 passed, 2 a threshold failed, and 1 a
configuration or runtime error.
"""

from __future__ import annotations

import sys
from pathlib import Path

from aai_core.agentkit.cli import main as agentkit_main

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    """Validate reviewed local inputs, then run the shared smoke gate."""

    sys.path.insert(0, str(ROOT / "src"))
    from app.targets import validate_bundled_data

    validate_bundled_data(ROOT, include_answer_sheet=True)
    return agentkit_main(["smoke"])


if __name__ == "__main__":
    raise SystemExit(main())
