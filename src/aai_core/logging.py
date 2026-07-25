"""Structured logging with process-local secret redaction."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from threading import RLock
from typing import Any

from aai_core.tags import ResourceContext


class Redactor:
    """Redact registered secret values without exposing them through repr."""

    def __init__(self) -> None:
        self._values: set[str] = set()
        self._lock = RLock()

    def register(self, value: str) -> None:
        if value:
            with self._lock:
                self._values.add(value)

    def redact(self, value: Any) -> Any:
        if isinstance(value, str):
            with self._lock:
                redacted = value
                for secret in sorted(self._values, key=len, reverse=True):
                    redacted = redacted.replace(secret, "[REDACTED]")
                return redacted
        if isinstance(value, Mapping):
            return {key: self.redact(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(self.redact(item) for item in value)
        if isinstance(value, list):
            return [self.redact(item) for item in value]
        return value


class RedactingFilter(logging.Filter):
    def __init__(self, redactor: Redactor) -> None:
        super().__init__()
        self.redactor = redactor

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = self.redactor.redact(record.msg)
        record.args = self.redactor.redact(record.args)
        return True


class JsonFormatter(logging.Formatter):
    def __init__(self, context: ResourceContext) -> None:
        super().__init__()
        self._context = context

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            **self._context.for_trace(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, sort_keys=True, default=str)


def configure_logging(
    context: ResourceContext,
    *,
    level: int | str = logging.INFO,
    redactor: Redactor | None = None,
) -> Redactor:
    """Configure the root logger once and return the active redactor."""

    active_redactor = redactor or Redactor()
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter(context))
    handler.addFilter(RedactingFilter(active_redactor))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    return active_redactor
