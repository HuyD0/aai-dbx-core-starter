"""Responses API message normalization tests."""

from types import SimpleNamespace

import pytest

from app.messages import response_message_text


def test_response_message_text_joins_text_parts():
    message = SimpleNamespace(
        content=[
            {"type": "input_text", "text": "First"},
            SimpleNamespace(type="input_text", text="Second"),
        ]
    )

    assert response_message_text(message) == "First\nSecond"


def test_response_message_text_rejects_unsupported_multimodal_parts():
    message = SimpleNamespace(
        content=[{"type": "input_image", "image_url": "https://example.invalid"}]
    )

    with pytest.raises(ValueError, match="text-only"):
        response_message_text(message)
