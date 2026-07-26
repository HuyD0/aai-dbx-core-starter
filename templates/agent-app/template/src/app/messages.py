"""Text-only normalization for MLflow Responses API messages."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def response_message_text(item: Any) -> str:
    """Return text from a Responses message or reject unsupported media."""

    content = getattr(item, "content", None)
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise TypeError("Responses messages must contain text")

    parts: list[str] = []
    for part in content:
        part_type = (
            part.get("type")
            if isinstance(part, Mapping)
            else getattr(part, "type", None)
        )
        if part_type not in {"input_text", "output_text"}:
            raise ValueError(
                "This agent is text-only; input_image and input_file parts "
                "require an explicitly governed multimodal implementation"
            )
        text = (
            part.get("text")
            if isinstance(part, Mapping)
            else getattr(part, "text", None)
        )
        if not isinstance(text, str):
            raise TypeError("Responses text parts require a string text field")
        parts.append(text)
    return "\n".join(parts)
