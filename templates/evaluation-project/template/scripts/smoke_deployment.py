#!/usr/bin/env python3
"""Post-deploy verification of the Databricks App this bundle serves.

The release gate proved the change scores; this proves the deployed app
actually serves it. Two tiers:

1. **Status** (always): poll ``databricks apps get`` until the app reports
   RUNNING, bounded by ``--timeout-seconds``. Needs no permission beyond
   what the deploy workflow already uses.
2. **Golden probes** (opt-in, by committing ``evals/data/live_probes.json``):
   POST each probe's request to the live app URL with a workspace bearer
   token and assert HTTP 200, required content, and a latency budget.
   Probing requires the CI principal to hold CAN_USE on the app — an
   external platform grant (see docs/uat-promotion.md).

Verdicts, latencies, and probe ids are logged; request and response
bodies never are — app traffic may carry governed content.

Probe file shape (a JSON list)::

    [{"id": "gp-001",
      "path": "/invocations",
      "request": {"input": [{"role": "user", "content": "..."}]},
      "must_contain": ["expected text"],
      "max_seconds": 30}]

Standard library only: this runs inside the credentialed deploy workflow,
which installs nothing beyond the Databricks CLI.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

TEMPLATE_STAMP = ".aai-template.json"
DEFAULT_PROBES = "evals/data/live_probes.json"
FAILED_STATES = {"ERROR", "CRASHED"}
ACCESS_HINT = (
    "the app rejected the workspace token (HTTP {code}). Golden probes need "
    "the CI principal to hold CAN_USE on the app - an external platform "
    "grant (see docs/uat-promotion.md). The status check above already "
    "verified the app is RUNNING; remove evals/data/live_probes.json to "
    "run status-only smoke until the grant exists."
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target", required=True, choices=("dev", "uat"))
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--probes", default=DEFAULT_PROBES)
    arguments = parser.parse_args()

    app_name = _app_name(arguments.target)
    payload = _wait_until_running(
        app_name,
        timeout_seconds=arguments.timeout_seconds,
        poll_seconds=arguments.poll_seconds,
    )
    url = str(payload.get("url") or "")
    if not url.startswith("https://"):
        print(f"smoke FAILED: {app_name} reports no https url", file=sys.stderr)
        return 1
    print(f"smoke: {app_name} is RUNNING")

    probes_path = Path(arguments.probes)
    if not probes_path.is_file():
        print(
            "smoke passed (status only): commit "
            f"{arguments.probes} to add golden-prompt probes"
        )
        return 0
    return _run_probes(probes_path, url)


def _app_name(target: str) -> str:
    stamp = json.loads(Path(TEMPLATE_STAMP).read_text(encoding="utf-8"))
    return f"{stamp['generated_with']['application_name']}-{target}"


def _wait_until_running(
    app_name: str, *, timeout_seconds: int, poll_seconds: int
) -> dict:
    deadline = time.monotonic() + timeout_seconds
    state = "UNKNOWN"
    while True:
        payload = _apps_get(app_name)
        state = str((payload.get("app_status") or {}).get("state") or "UNKNOWN")
        if state == "RUNNING":
            return payload
        if state in FAILED_STATES:
            message = (payload.get("app_status") or {}).get("message") or ""
            print(
                f"smoke FAILED: {app_name} is {state}: {message}",
                file=sys.stderr,
            )
            raise SystemExit(1)
        if time.monotonic() >= deadline:
            print(
                f"smoke FAILED: {app_name} did not reach RUNNING within "
                f"{timeout_seconds}s (last state {state})",
                file=sys.stderr,
            )
            raise SystemExit(1)
        print(f"smoke: {app_name} is {state}; waiting...")
        time.sleep(max(1, poll_seconds))


def _apps_get(app_name: str) -> dict:
    completed = subprocess.run(
        ["databricks", "apps", "get", app_name, "--output", "json"],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if completed.returncode != 0:
        print(
            f"smoke FAILED: `databricks apps get {app_name}` exited "
            f"{completed.returncode}: {completed.stderr.strip()}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return json.loads(completed.stdout)


def _bearer_token() -> str:
    completed = subprocess.run(
        ["databricks", "auth", "token", "--output", "json"],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if completed.returncode != 0:
        print(
            "smoke FAILED: `databricks auth token` exited "
            f"{completed.returncode}: {completed.stderr.strip()}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return str(json.loads(completed.stdout)["access_token"])


def _run_probes(probes_path: Path, url: str) -> int:
    probes = json.loads(probes_path.read_text(encoding="utf-8"))
    if not isinstance(probes, list) or not probes:
        print(f"smoke FAILED: {probes_path} must be a non-empty JSON list")
        return 1
    token = _bearer_token()
    failures = []
    for probe in probes:
        probe_id = str(probe.get("id") or "unnamed-probe")
        verdict = _probe_verdict(probe, url, token)
        if verdict:
            failures.append(f"{probe_id}: {verdict}")
    if failures:
        print("smoke FAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print(f"smoke passed: {len(probes)} golden probe(s) against the live app")
    return 0


def _probe_verdict(probe: dict, url: str, token: str) -> str | None:
    """One probe's failure text, or ``None`` when it passed.

    Only verdicts, probe ids, and latencies are printed — never a request
    or response body; app traffic may carry governed content.
    """

    probe_id = str(probe.get("id") or "unnamed-probe")
    budget = float(probe.get("max_seconds") or 30)
    request = urllib.request.Request(
        url.rstrip("/") + str(probe.get("path") or "/invocations"),
        data=json.dumps(probe.get("request") or {}).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=budget + 30) as response:
            body = response.read().decode("utf-8", errors="replace")
            status = int(response.status)
    except urllib.error.HTTPError as error:
        if error.code in (401, 403):
            return ACCESS_HINT.format(code=error.code)
        return f"HTTP {error.code}"
    except Exception as error:  # noqa: BLE001 - network verdicts only
        return type(error).__name__
    elapsed = time.monotonic() - started
    verdicts = []
    if status != 200:
        verdicts.append(f"HTTP {status}")
    if not body.strip():
        verdicts.append("empty response")
    for must in probe.get("must_contain") or []:
        if str(must).casefold() not in body.casefold():
            verdicts.append(f"missing expected content {must!r}")
    if elapsed > budget:
        verdicts.append(f"{elapsed:.1f}s over the {budget:g}s latency budget")
    if verdicts:
        print(f"smoke probe {probe_id}: FAILED ({elapsed:.1f}s)")
        return "; ".join(verdicts)
    print(f"smoke probe {probe_id}: ok ({elapsed:.1f}s)")
    return None


if __name__ == "__main__":
    raise SystemExit(main())
