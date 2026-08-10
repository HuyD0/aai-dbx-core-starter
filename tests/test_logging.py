"""Structured logging ownership and secret-redaction tests."""

from __future__ import annotations

import io
import json
import logging
import sys

from aai_core.logging import JsonFormatter, RedactingFilter, Redactor, configure_logging
from aai_core.tags import ResourceContext


def _context() -> ResourceContext:
    return ResourceContext(
        application="logging-test",
        project="sdk",
        environment="dev",
        team="platform",
        owner_group="group:platform-owners",
        cost_center="CC-0000",
        data_classification="internal",
        lifecycle="experimental",
        repository="org/repo",
        release="dev",
    )


def _exception_record(secret: str) -> logging.LogRecord:
    try:
        raise ValueError(f"native provider included {secret}")
    except ValueError:
        return logging.LogRecord(
            "test",
            logging.ERROR,
            __file__,
            1,
            "request failed with %s",
            (secret,),
            sys.exc_info(),
        )


def test_json_formatter_redacts_message_and_formatted_traceback() -> None:
    secret = "secret-with-special-characters-+/="
    redactor = Redactor()
    redactor.register(secret)
    record = _exception_record(secret)

    RedactingFilter(redactor).filter(record)
    rendered = JsonFormatter(_context(), redactor=redactor).format(record)
    payload = json.loads(rendered)

    assert secret not in rendered
    assert payload["message"] == "request failed with [REDACTED]"
    assert "ValueError: native provider included [REDACTED]" in payload["exception"]


def test_configure_logging_preserves_and_protects_host_handlers(monkeypatch) -> None:
    root = logging.Logger("isolated-root")
    host_output = io.StringIO()
    host_handler = logging.StreamHandler(host_output)
    host_handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(host_handler)
    monkeypatch.setattr("aai_core.logging.logging.getLogger", lambda: root)

    redactor = Redactor()
    secret = "host-handler-secret"
    redactor.register(secret)
    configure_logging(_context(), redactor=redactor)
    configure_logging(_context(), redactor=redactor)

    assert host_handler in root.handlers
    assert len(root.handlers) == 2

    try:
        raise RuntimeError(f"provider returned {secret}")
    except RuntimeError:
        root.exception("request used %s", secret)

    rendered = host_output.getvalue()
    assert secret not in rendered
    assert "request used [REDACTED]" in rendered
    assert "RuntimeError: provider returned [REDACTED]" in rendered
