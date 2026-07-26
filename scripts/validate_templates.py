"""Render every template and `databricks bundle validate` it for real.

Runs on the credentialed path (post-merge deploy workflow) where PR CI
cannot: resource schemas — including vector search indexes — are validated
against the live workspace API. Configs come from each template's schema
defaults plus the identifier fixture, exactly like the credential-free
render tests.

Stdlib only; needs `databricks` on PATH and workspace auth in the
environment.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
IDENTIFIERS = json.loads((REPO_ROOT / "platform-identifiers.json").read_text())


def discover_templates() -> list[Path]:
    return sorted(
        entry
        for entry in (REPO_ROOT / "templates").iterdir()
        if entry.is_dir() and (entry / "databricks_template_schema.json").is_file()
    )


def config_for(template: Path) -> dict:
    properties = json.loads((template / "databricks_template_schema.json").read_text())[
        "properties"
    ]
    overrides = {
        "workspace_host": IDENTIFIERS["databricks_host"],
        "compute_policy_id": IDENTIFIERS["job_compute_policy_id"],
        "aai_core_volume": IDENTIFIERS["sdk_artifact_volume"],
    }
    return {key: value for key, value in overrides.items() if key in properties}


def main() -> int:
    failures: list[str] = []
    for template in discover_templates():
        with tempfile.TemporaryDirectory() as scratch:
            config_path = Path(scratch) / "config.json"
            config_path.write_text(json.dumps(config_for(template)))
            output = Path(scratch) / "generated"
            try:
                subprocess.run(
                    [
                        "databricks",
                        "bundle",
                        "init",
                        str(template),
                        "--config-file",
                        str(config_path),
                        "--output-dir",
                        str(output),
                    ],
                    check=True,
                    stdin=subprocess.DEVNULL,
                    cwd=scratch,
                )
                subprocess.run(
                    ["databricks", "bundle", "validate", "-t", "dev"],
                    check=True,
                    cwd=output,
                )
            except subprocess.CalledProcessError as error:
                failures.append(f"{template.name}: {error}")
                continue
        print(f"validated {template.name}")

    summary = os.getenv("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as stream:
            stream.write("## Template catalog validation\n\n")
            for template in discover_templates():
                status = (
                    "FAILED"
                    if any(line.startswith(template.name) for line in failures)
                    else "ok"
                )
                stream.write(f"- {template.name}: {status}\n")
            stream.write("\n")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print(f"all {len(discover_templates())} templates validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
