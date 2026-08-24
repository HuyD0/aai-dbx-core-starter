"""Refresh the console's Azure VM list-price rates from the public retail API.

The cost estimator's snapshot at
src/platform_app/aai_console/pricing_data/azure_prices.json has two kinds of
tables:

1. ``vm_rates`` — Azure VM on-demand/spot $/hour. This script rewrites that
   section (and ``metadata.sources.vm_rates.as_of``) from the public,
   unauthenticated Azure Retail Prices API for every region × instance the
   snapshot lists.
2. Everything else (DBU $/rates, instance DBU/hour, warehouse ladders, FMAPI
   token rates) — hand-curated from the public pricing pages cited in
   ``metadata.sources``. This script never touches those; re-check them against
   their cited URLs and bump ``metadata.as_of`` when you refresh.

Maintainer-run only: it performs outbound HTTPS, so it must never be wired into
CI, a workflow, or the app runtime. Standard library only, so it runs anywhere.

Usage:
    python scripts/refresh_pricing_snapshot.py          # rewrite vm_rates
    python scripts/refresh_pricing_snapshot.py --check  # verify, exit 1 on gaps
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_FILE = (
    REPO_ROOT
    / "src"
    / "platform_app"
    / "aai_console"
    / "pricing_data"
    / "azure_prices.json"
)
API_ENDPOINT = "https://prices.azure.com/api/retail/prices"
API_VERSION = "2023-01-01-preview"
SKUS_PER_REQUEST = 8  # keeps the $filter short; the API throttles greedily
MAX_ATTEMPTS = 6
MAX_PAGES_PER_REQUEST = 1_000
MAX_RESPONSE_BYTES = 10 * 1024 * 1024
_ALLOWED_API_HOST = "prices.azure.com"


def _validated_api_url(url: str) -> str:
    """Allow only the documented Azure Retail Prices HTTPS endpoint."""

    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != _ALLOWED_API_HOST
        or parsed.port not in (None, 443)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path != "/api/retail/prices"
        or parsed.fragment
    ):
        raise ValueError("Retail API pagination returned an untrusted URL")
    return url


def _get_json(url: str) -> dict:
    """GET with retry: the retail API rate-limits with 429 fairly eagerly."""

    safe_url = _validated_api_url(url)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(safe_url, timeout=60) as response:
                declared_size = response.headers.get("Content-Length")
                if declared_size and int(declared_size) > MAX_RESPONSE_BYTES:
                    raise ValueError("Retail API response exceeds the size limit")
                body = response.read(MAX_RESPONSE_BYTES + 1)
                if len(body) > MAX_RESPONSE_BYTES:
                    raise ValueError("Retail API response exceeds the size limit")
                payload = json.loads(body)
                if not isinstance(payload, dict):
                    raise ValueError("Retail API response must be a JSON object")
                return payload
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503) or attempt == MAX_ATTEMPTS:
                raise
            retry_after = error.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after else 2.0**attempt
            except ValueError:
                delay = 2.0**attempt
            delay = min(max(delay, 0.0), 60.0)
            print(f"  HTTP {error.code}; retrying in {delay:.0f}s", file=sys.stderr)
            time.sleep(delay)
    raise AssertionError("unreachable")


def fetch_region_rates(region: str, instances: list[str]) -> dict[str, dict]:
    """Return {instance: {"on_demand": rate, "spot": rate|None}} for one region."""

    rates: dict[str, dict] = {name: {} for name in instances}
    for start in range(0, len(instances), SKUS_PER_REQUEST):
        chunk = instances[start : start + SKUS_PER_REQUEST]
        sku_filter = " or ".join(f"armSkuName eq '{name}'" for name in chunk)
        query = urllib.parse.urlencode(
            {
                "api-version": API_VERSION,
                "currencyCode": "USD",
                "$filter": (
                    "serviceName eq 'Virtual Machines' and "
                    "priceType eq 'Consumption' "
                    f"and armRegionName eq '{region}' and ({sku_filter})"
                ),
            }
        )
        url = f"{API_ENDPOINT}?{query}"
        seen_pages: set[str] = set()
        page_count = 0
        while url:
            url = _validated_api_url(url)
            if url in seen_pages:
                raise ValueError("Retail API pagination contains a cycle")
            seen_pages.add(url)
            page_count += 1
            if page_count > MAX_PAGES_PER_REQUEST:
                raise ValueError("Retail API pagination exceeds the page limit")
            payload = _get_json(url)
            _merge_items(rates, payload.get("Items", []))
            next_page = payload.get("NextPageLink")
            if next_page is not None and not isinstance(next_page, str):
                raise ValueError("Retail API pagination URL must be a string")
            url = next_page
    return rates


def _merge_items(rates: dict[str, dict], items: list[dict]) -> None:
    for item in items:
        name = item.get("armSkuName")
        sku_name = item.get("skuName", "")
        product = item.get("productName", "")
        price = item.get("unitPrice")
        if name not in rates or not price:
            continue
        if "Windows" in product or "Low Priority" in sku_name:
            continue
        key = "spot" if "Spot" in sku_name else "on_demand"
        # Duplicate meters occasionally appear; keep the lowest list price.
        current = rates[name].get(key)
        if current is None or price < current:
            rates[name][key] = price


def refresh(snapshot: dict, *, today: dt.date) -> tuple[dict, list[str]]:
    regions = [region["id"] for region in snapshot["regions"]]
    instances = sorted(snapshot["instances"])
    previous = snapshot.get("vm_rates", {})
    missing: list[str] = []
    vm_rates: dict[str, dict] = {}
    for region in regions:
        fetched = fetch_region_rates(region, instances)
        vm_rates[region] = {}
        for name in instances:
            rate = fetched.get(name, {})
            on_demand = rate.get("on_demand")
            if on_demand is None:
                missing.append(f"{region}/{name}: no on-demand rate returned")
                continue
            entry: dict = {"on_demand": on_demand}
            if rate.get("spot") is not None:
                entry["spot"] = rate["spot"]
            old = previous.get(region, {}).get(name, {})
            if old.get("on_demand") not in (None, on_demand):
                print(
                    f"  {region}/{name}: on-demand "
                    f"{old['on_demand']} -> {on_demand}"
                )
            vm_rates[region][name] = entry
    snapshot["vm_rates"] = vm_rates
    snapshot["metadata"]["sources"]["vm_rates"]["as_of"] = today.isoformat()
    return snapshot, missing


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fetch and compare without writing; exit 1 on any gap",
    )
    args = parser.parse_args(argv)

    snapshot = json.loads(SNAPSHOT_FILE.read_text(encoding="utf-8"))
    snapshot, missing = refresh(snapshot, today=dt.date.today())
    for gap in missing:
        print(f"missing: {gap}", file=sys.stderr)
    if missing:
        print(
            "Some rates were not returned; the snapshot was not written. "
            "Remove the instance/region or curate the rate by hand.",
            file=sys.stderr,
        )
        return 1
    if args.check:
        print("all vm_rates resolvable; snapshot not written (--check)")
        return 0
    SNAPSHOT_FILE.write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {SNAPSHOT_FILE.relative_to(REPO_ROOT)}")
    print(
        "Reminder: DBU tables are hand-curated. Re-check the URLs in "
        "metadata.sources and bump metadata.as_of when refreshing them."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
