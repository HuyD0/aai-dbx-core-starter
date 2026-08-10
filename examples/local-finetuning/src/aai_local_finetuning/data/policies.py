"""Versioned, deterministic labeling policies for the learning dataset.

The source flags have no semantics inferred here: they are normalized and retained
for slice analysis. ``difficulty`` is an explicit curriculum heuristic, not Bitext
ground truth. Escalation is likewise a narrow learning policy, not production advice.
Changing either rule requires a new policy version and a regenerated manifest.
"""

from __future__ import annotations

import re

from .schemas import Difficulty

ESCALATION_POLICY_VERSION = "bitext-escalation-v1"
DIFFICULTY_POLICY_VERSION = "bitext-difficulty-v1"
RESPONSE_POLICY_VERSION = "bitext-safe-response-v1"

# Requiring both the documented category and intent prevents a mislabeled row from
# silently acquiring a policy-critical target.
ESCALATION_RULES: frozenset[tuple[str, str]] = frozenset(
    {
        ("account", "registration_problems"),
        ("contact", "contact_human_agent"),
        ("feedback", "complaint"),
        ("payment", "payment_issue"),
    }
)

_FLAG_CHARACTER = re.compile(r"[A-Z0-9]")


def parse_flags(value: str) -> tuple[str, ...]:
    """Return unique source flag characters in a stable order."""

    return tuple(sorted(set(_FLAG_CHARACTER.findall(value.upper()))))


def classify_difficulty(instruction: str, flags: tuple[str, ...]) -> Difficulty:
    """Apply the v1 length/flag-count heuristic used only for evaluation slices.

    * hard: at least 12 whitespace-delimited input words or at least 6 flags;
    * easy: at most 6 words and at most 2 flags;
    * standard: every other record.
    """

    word_count = len(instruction.split())
    if word_count >= 12 or len(flags) >= 6:
        return Difficulty.HARD
    if word_count <= 6 and len(flags) <= 2:
        return Difficulty.EASY
    return Difficulty.STANDARD


def requires_escalation(*, category: str, intent: str) -> bool:
    """Apply the v1 category-and-intent escalation target policy."""

    return (category, intent) in ESCALATION_RULES


def render_training_response(*, intent: str, escalation: bool) -> str:
    """Render the v1 brief, policy-safe target instead of copying source prose."""

    if escalation:
        return "I'll route this request to a support specialist for review."
    topic = intent.replace("_", " ")
    return f"I can help with your {topic} request."


def policy_versions() -> dict[str, str]:
    return {
        "difficulty": DIFFICULTY_POLICY_VERSION,
        "requires_escalation": ESCALATION_POLICY_VERSION,
        "training_response": RESPONSE_POLICY_VERSION,
    }
