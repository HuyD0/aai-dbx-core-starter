"""Structured-output generation with validation.

A thin, dependency-free layer over the OpenAI-compatible
``response_format=json_schema`` path: capability-checked by the adapter,
parsed and required-keys-validated here, with a stable error type instead of
a raw ``JSONDecodeError`` from deep inside an application.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel, ValidationError

from aai_core.exceptions import AaiCoreError

if TYPE_CHECKING:
    from aai_core.providers.types import ChatModel

StructuredModel = TypeVar("StructuredModel", bound=BaseModel)


class StructuredOutputError(AaiCoreError):
    code = "aai_core.structured.invalid"


def generate_structured(
    model: ChatModel,
    messages: Sequence[Mapping[str, Any]],
    *,
    json_schema: Mapping[str, Any],
    name: str = "response",
    **options: Any,
) -> dict[str, Any]:
    """Generate and return a validated JSON object matching ``json_schema``.

    Validation covers JSON parse plus the schema's top-level ``required``
    keys — enough to fail fast and loudly. The model must declare the
    ``structured_output`` capability (enforced by the adapter).
    """

    response = model.generate(
        messages,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": name,
                "schema": dict(json_schema),
                "strict": True,
            },
        },
        **options,
    )
    try:
        parsed = json.loads(response.content)
    except json.JSONDecodeError:
        raise StructuredOutputError(
            f"Model returned invalid JSON for {name!r}",
            remediation="Verify the endpoint supports json_schema response "
            "format; simplify the schema or lower the temperature.",
        ) from None
    if not isinstance(parsed, dict):
        raise StructuredOutputError(
            f"Model returned a JSON {type(parsed).__name__}, not an object"
        )
    missing = [key for key in json_schema.get("required", []) if key not in parsed]
    if missing:
        raise StructuredOutputError(
            f"Structured response is missing required keys: {missing}"
        )
    return parsed


def generate_typed(
    model: ChatModel,
    messages: Sequence[Mapping[str, Any]],
    *,
    response_model: type[StructuredModel],
    name: str | None = None,
    **options: Any,
) -> StructuredModel:
    """Generate and strictly validate a Pydantic response model.

    The Pydantic class is the application-owned boundary: its JSON Schema is
    sent to the provider and the returned JSON is validated with Pydantic v2.
    Validation errors and model content are deliberately omitted from the
    raised error so prompts, personal data, and provider details do not leak.
    """

    schema_name = name or response_model.__name__
    response = model.generate(
        messages,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "schema": response_model.model_json_schema(),
                "strict": True,
            },
        },
        **options,
    )
    try:
        return response_model.model_validate_json(response.content, strict=True)
    except (ValidationError, ValueError, TypeError):
        raise StructuredOutputError(
            f"Model returned invalid structured output for {schema_name!r}",
            remediation="Inspect the governed trace, verify endpoint structured-"
            "output support, and simplify or correct the response model.",
        ) from None
