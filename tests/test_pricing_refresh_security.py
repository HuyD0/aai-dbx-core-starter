"""Security contracts for the maintainer-only Azure pricing updater."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "refresh_pricing_snapshot",
    ROOT / "scripts" / "refresh_pricing_snapshot.py",
)
assert SPEC and SPEC.loader
pricing = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pricing)


@pytest.mark.parametrize(
    "url",
    [
        "http://prices.azure.com/api/retail/prices",
        "https://example.invalid/api/retail/prices",
        "https://prices.azure.com.evil.invalid/api/retail/prices",
        "https://prices.azure.com@evil.invalid/api/retail/prices",
        "https://prices.azure.com:444/api/retail/prices",
        "https://prices.azure.com/private",
    ],
)
def test_rejects_untrusted_pagination_urls(url):
    with pytest.raises(ValueError, match="untrusted URL"):
        pricing._validated_api_url(url)


def test_rejects_pagination_cycle(monkeypatch):
    def fake_get_json(url):
        return {"Items": [], "NextPageLink": url}

    monkeypatch.setattr(pricing, "_get_json", fake_get_json)

    with pytest.raises(ValueError, match="cycle"):
        pricing.fetch_region_rates("canadacentral", ["Standard_D2s_v5"])


def test_rejects_oversized_response(monkeypatch):
    class Response:
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, size):
            assert size == pricing.MAX_RESPONSE_BYTES + 1
            return b"x" * size

    monkeypatch.setattr(pricing.urllib.request, "urlopen", lambda *_a, **_k: Response())

    with pytest.raises(ValueError, match="size limit"):
        pricing._get_json(pricing.API_ENDPOINT)


def test_accepts_bounded_json_object(monkeypatch):
    expected = {"Items": [], "NextPageLink": None}

    class Response:
        headers = {"Content-Length": str(len(json.dumps(expected)))}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, _size):
            return json.dumps(expected).encode()

    monkeypatch.setattr(pricing.urllib.request, "urlopen", lambda *_a, **_k: Response())
    assert pricing._get_json(pricing.API_ENDPOINT) == expected
