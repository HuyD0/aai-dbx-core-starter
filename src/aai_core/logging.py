"""Structured logging with process-local secret redaction."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from threading import RLock
from typing import Any, TextIO, cast

from aai_core.tags import ResourceContext

__all__ = ["JsonFormatter", "RedactingFilter", "Redactor", "configure_logging"]


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
    """Redact a record before any attached formatter can render it."""

    def __init__(self, redactor: Redactor) -> None:
        super().__init__()
        self.redactor = redactor

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = self.redactor.redact(record.msg)
        record.args = self.redactor.redact(record.args)
        if record.exc_info:
            exception = logging.Formatter().formatException(record.exc_info)
            record.exc_text = self.redactor.redact(exception)
            # A later formatter must not reconstruct an unredacted traceback.
            record.exc_info = None
        elif record.exc_text:
            record.exc_text = self.redactor.redact(record.exc_text)
        if record.stack_info:
            record.stack_info = self.redactor.redact(record.stack_info)
        return True


class JsonFormatter(logging.Formatter):
    """Render bounded platform context and a redacted log record as JSON."""

    def __init__(
        self,
        context: ResourceContext,
        *,
        redactor: Redactor | None = None,
    ) -> None:
        super().__init__()
        self._context = context
        self._redactor = redactor or Redactor()

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
        elif record.exc_text:
            payload["exception"] = record.exc_text
        if record.stack_info:
            payload["stack"] = record.stack_info
        # Redact once more after formatting so exception text, custom string
        # conversion, and JSON escaping cannot bypass the record filter.
        rendered = json.dumps(
            self._redactor.redact(payload),
            sort_keys=True,
            default=str,
        )
        return cast(str, self._redactor.redact(rendered))


class _AAICoreHandler(logging.StreamHandler[TextIO]):
    """Marker class for the one handler owned by aai-core."""


def configure_logging(
    context: ResourceContext,
    *,
    level: int | str = logging.INFO,
    redactor: Redactor | None = None,
) -> Redactor:
    """Add or update the SDK handler without replacing application handlers.

    Existing handlers receive the same redacting filter so a registered secret
    cannot leak through a host formatter that observes the record before the
    SDK handler does.
    """

    active_redactor = redactor or Redactor()
    root = logging.getLogger()
    for existing in root.handlers:
        _install_redacting_filter(existing, active_redactor)

    handler = next(
        (
            existing
            for existing in root.handlers
            if isinstance(existing, _AAICoreHandler)
        ),
        None,
    )
    if handler is None:
        handler = _AAICoreHandler()
        root.addHandler(handler)
    handler.setFormatter(JsonFormatter(context, redactor=active_redactor))
    _install_redacting_filter(handler, active_redactor)
    root.setLevel(level)
    return active_redactor


def _install_redacting_filter(
    handler: logging.Handler,
    redactor: Redactor,
) -> None:
    if any(
        isinstance(candidate, RedactingFilter) and candidate.redactor is redactor
        for candidate in handler.filters
    ):
        return
    handler.addFilter(RedactingFilter(redactor))
