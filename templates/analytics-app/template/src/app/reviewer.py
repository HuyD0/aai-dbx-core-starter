"""Optional adversarial review pass over a drafted answer.

Published tradeoff from production use of this pattern: roughly +6%
accuracy for +32% tokens and +72% latency — which is why it ships opt-in
and OFF by default (the adversarial_review template input, or the
AAI_ANALYTICS_ADVERSARIAL_REVIEW environment variable, turns it on). The
reviewer only sees the tool-recorded evidence, so it can strike numbers the
draft cannot support but can never introduce new ones.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict


class ReviewVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    approved: bool
    revised_answer: str | None = None
    objections: tuple[str, ...] = ()


def load_reviewer_messages(path: str | Path) -> tuple[dict[str, Any], ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return tuple(dict(message) for message in payload["messages"])


def render_reviewer_messages(
    messages: tuple[dict[str, Any], ...],
    *,
    question: str,
    answer: str,
    evidence: str,
) -> list[dict[str, Any]]:
    rendered = []
    for message in messages:
        content = str(message.get("content", ""))
        content = content.replace("{{question}}", question)
        content = content.replace("{{answer}}", answer)
        content = content.replace("{{evidence}}", evidence)
        rendered.append({"role": message["role"], "content": content})
    return rendered


def review_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": ReviewVerdict.__name__,
            "schema": ReviewVerdict.model_json_schema(),
            "strict": True,
        },
    }


def apply_verdict(draft: str, verdict: ReviewVerdict) -> str:
    """The reviewed prose: revised when rejected, annotated when objections
    stand without a rewrite."""

    if verdict.approved:
        return draft
    if verdict.revised_answer and verdict.revised_answer.strip():
        return verdict.revised_answer.strip()
    if verdict.objections:
        notes = "; ".join(verdict.objections)
        return f"{draft}\n\nReview notes: {notes}"
    return draft


# Kept next to the verdict model so tests pin the schema the reviewer must
# return; a drifting schema fails fast instead of silently approving.
VERDICT_SCHEMA_FIELDS = tuple(sorted(ReviewVerdict.model_fields))
