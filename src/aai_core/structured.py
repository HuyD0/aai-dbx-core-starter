"""Structured-output generation with validation.

A thin, dependency-free layer over the OpenAI-compatible
``response_format=json_schema`` path: capability-checked by the adapter,
parsed and required-keys-validated here, with a stable error type instead of
a raw ``JSONDecodeError`` from deep inside an application.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from aai_core.exceptions import AaiCoreError

if TYPE_CHECKING:
    from aai_core.providers.types import ChatModel


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
    except json.JSONDecodeError as error:
        raise StructuredOutputError(
            f"Model returned invalid JSON for {name!r}",
            remediation="Verify the endpoint supports json_schema response "
            "format; simplify the schema or lower the temperature.",
        ) from error
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
