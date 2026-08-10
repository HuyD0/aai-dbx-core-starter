"""Tier-1 gate: deterministic checks plus code scorers, offline.

Runs in pull-request CI with zero credentials. `agentkit smoke` validates
the dataset, scores it with the deterministic code scorers from the shared
registry, and applies their thresholds. Judge scorers cannot execute here —
they run in evals/evaluate.py on the credentialed path.

Exit codes are the CI contract: 0 passed, 2 a threshold failed, 1 a
configuration or runtime error.
"""

from __future__ import annotations

from aai_core.agentkit.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["smoke"]))
