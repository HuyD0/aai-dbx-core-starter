from __future__ import annotations

import pytest

from aai_local_classification.settings import load_settings


@pytest.fixture(scope="session")
def settings():
    return load_settings()
