"""Earliest-possible import/exec audit boundary for execution evidence."""

from __future__ import annotations

import sys

_PREEXISTING_MODULES = dict(sys.modules)
_PENDING_EVENTS: list[tuple[str, tuple[object, ...]]] = []
_PENDING_EVENT_LIMIT = 65_536
_PENDING_EVENTS_OVERFLOWED = False
_DELEGATE = None


def _bootstrap_audit_hook(event: str, args: tuple[object, ...]) -> None:
    global _PENDING_EVENTS_OVERFLOWED
    if event not in {"exec", "import"}:
        return
    delegate = _DELEGATE
    if delegate is None:
        if len(_PENDING_EVENTS) >= _PENDING_EVENT_LIMIT:
            _PENDING_EVENTS_OVERFLOWED = True
        else:
            _PENDING_EVENTS.append((event, args))
    else:
        delegate(event, args)


sys.addaudithook(_bootstrap_audit_hook)


def activate(delegate: object) -> None:
    """Route queued and future events to the complete training audit hook."""

    global _DELEGATE
    if _DELEGATE is not None:
        if _DELEGATE is not delegate:
            raise RuntimeError("runtime audit delegate is already installed")
        return
    if not callable(delegate):
        raise TypeError("runtime audit delegate must be callable")
    if _PENDING_EVENTS_OVERFLOWED:
        raise RuntimeError("runtime bootstrap audit event buffer overflowed")
    _DELEGATE = delegate
    pending = tuple(_PENDING_EVENTS)
    _PENDING_EVENTS.clear()
    for event, args in pending:
        delegate(event, args)


def was_preexisting(name: str, module: object) -> bool:
    """Return whether the same module object predates this audit boundary."""

    return _PREEXISTING_MODULES.get(name) is module
