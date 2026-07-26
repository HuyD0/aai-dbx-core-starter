"""Sync canonical shared scaffold files into every bundle template.

The Databricks template renderer only sees one template root, so files cannot
be shared across template directories natively. templates/_shared/ holds the
canonical copies; this script copies them into each template (a template is
any templates/<dir>/ containing databricks_template_schema.json).

Usage:
    python scripts/sync_template_shared.py          # write copies
    python scripts/sync_template_shared.py --check  # CI drift check, exit 1 on drift
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATES_DIR = REPO_ROOT / "templates"
SHARED_DIR = TEMPLATES_DIR / "_shared"
SCHEMA_FILE = "databricks_template_schema.json"


def discover_templates() -> list[Path]:
    return sorted(
        entry
        for entry in TEMPLATES_DIR.iterdir()
        if entry.is_dir() and (entry / SCHEMA_FILE).is_file()
    )


def planned_copies() -> list[tuple[Path, Path]]:
    """(canonical source, destination) pairs across all discovered templates."""

    manifest = json.loads((SHARED_DIR / "manifest.json").read_text(encoding="utf-8"))
    opt_out: dict[str, list[str]] = manifest.get("opt_out", {})
    pairs: list[tuple[Path, Path]] = []
    for template in discover_templates():
        for relative in manifest["files"]:
            if template.name in opt_out.get(relative, []):
                continue
            pairs.append(
                (SHARED_DIR / "files" / relative, template / "template" / relative)
            )
        for library_file in sorted((SHARED_DIR / "library").glob("*.tmpl")):
            pairs.append((library_file, template / "library" / library_file.name))
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report drift between canonical and template copies; exit 1 on any.",
    )
    args = parser.parse_args()

    drift: list[str] = []
    for source, destination in planned_copies():
        if not source.is_file():
            raise SystemExit(f"canonical file missing: {source}")
        in_sync = (
            destination.is_file() and destination.read_bytes() == source.read_bytes()
        )
        if in_sync:
            continue
        if args.check:
            state = "differs from" if destination.is_file() else "missing vs"
            drift.append(
                f"{destination.relative_to(REPO_ROOT)} {state} canonical "
                f"{source.relative_to(REPO_ROOT)}"
            )
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            print(f"synced {destination.relative_to(REPO_ROOT)}")

    if drift:
        for line in drift:
            print(f"DRIFT: {line}", file=sys.stderr)
        print(
            "Run `python scripts/sync_template_shared.py` after editing the "
            "canonical copy under templates/_shared/.",
            file=sys.stderr,
        )
        return 1
    print(
        f"shared scaffold in sync across {len(discover_templates())} template(s)"
        if args.check
        else "sync complete"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
