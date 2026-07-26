"""Shared helpers for hermetic tests.

Provider extras (openai, azure-*, databricks-*) are deliberately absent from
the dev environment, so tests that exercise real wiring install lightweight
fake modules instead of talking to any cloud.
"""

import sys
from types import ModuleType


def install_fake_module(monkeypatch, name: str, **attributes):
    """Register a fake module (and empty parent packages) in sys.modules."""

    parts = name.split(".")
    for index in range(1, len(parts)):
        parent = ".".join(parts[:index])
        if parent not in sys.modules:
            monkeypatch.setitem(sys.modules, parent, ModuleType(parent))
    module = ModuleType(name)
    for attribute, value in attributes.items():
        setattr(module, attribute, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module
