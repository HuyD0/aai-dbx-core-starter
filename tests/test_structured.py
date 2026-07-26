"""Typed structured-output boundary tests."""

import pytest
from pydantic import BaseModel, ConfigDict

from aai_core.structured import StructuredOutputError, generate_typed
from aai_core.testing import FakeChatModel


class ShipmentAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    answer: str
    confidence: float


def test_generate_typed_uses_model_schema_and_strict_validation():
    model = FakeChatModel(reply='{"answer":"shipped","confidence":0.9}')

    answer = generate_typed(
        model,
        [{"role": "user", "content": "Where is the order?"}],
        response_model=ShipmentAnswer,
    )

    assert answer == ShipmentAnswer(answer="shipped", confidence=0.9)
    response_format = model.requests[0]["response_format"]
    assert response_format["json_schema"]["name"] == "ShipmentAnswer"
    assert (
        response_format["json_schema"]["schema"]["properties"]["answer"]["type"]
        == "string"
    )
    assert response_format["json_schema"]["strict"] is True


def test_generate_typed_sanitizes_validation_failures():
    sensitive_content = '{"answer":"private customer content","confidence":"high"}'
    model = FakeChatModel(reply=sensitive_content)

    with pytest.raises(StructuredOutputError) as exc_info:
        generate_typed(
            model,
            [{"role": "user", "content": "question"}],
            response_model=ShipmentAnswer,
        )

    assert "invalid structured output" in str(exc_info.value)
    assert sensitive_content not in str(exc_info.value)
    assert "private customer content" not in repr(exc_info.value)
    assert exc_info.value.__cause__ is None
