"""Render every template and `databricks bundle validate` it for real.

Runs on the credentialed path (post-merge deploy workflow) where PR CI
cannot: bundle resource schemas are validated against the live
workspace API. Configs come from each template's schema
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

from validate_release import validate_repository

REPO_ROOT = Path(__file__).resolve().parents[1]
IDENTIFIERS = json.loads((REPO_ROOT / "platform-identifiers.json").read_text())

# Extra wizard-answer variants to validate beyond each template's defaults —
# combinations that render resources the default combination omits.
VARIANTS: dict[str, list[dict[str, str]]] = {
    # The chunk-pipeline job graph only renders for Databricks retrieval;
    # validate that variant too so its conditional tasks stay deployable.
    "rag-app": [{"retrieval_provider": "databricks_ai_search"}],
}


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


def validation_runs() -> list[tuple[Path, str, dict[str, str]]]:
    runs = []
    for template in discover_templates():
        runs.append((template, template.name, {}))
        for overrides in VARIANTS.get(template.name, []):
            label = f"{template.name}[{'/'.join(sorted(overrides.values()))}]"
            runs.append((template, label, overrides))
    return runs


def main() -> int:
    # Fail before cloud calls when template provenance, SDK compatibility, or
    # certified dependency declarations have drifted.
    validate_repository()
    failures: list[str] = []
    results: list[tuple[str, str]] = []
    for template, label, overrides in validation_runs():
        with tempfile.TemporaryDirectory() as scratch:
            config_path = Path(scratch) / "config.json"
            config_path.write_text(json.dumps({**config_for(template), **overrides}))
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
                failures.append(f"{label}: {error}")
                results.append((label, "FAILED"))
                continue
        results.append((label, "ok"))
        print(f"validated {label}")

    summary = os.getenv("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as stream:
            stream.write("## Template catalog validation\n\n")
            for label, status in results:
                stream.write(f"- {label}: {status}\n")
            stream.write("\n")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print(f"all {len(results)} template configurations validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
