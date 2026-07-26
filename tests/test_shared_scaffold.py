"""Cross-template shared-scaffold guarantees: sync drift and pin agreement."""

import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "templates" / "_shared"

_spec = importlib.util.spec_from_file_location(
    "sync_template_shared", ROOT / "scripts" / "sync_template_shared.py"
)
sync_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sync_module)


def test_shared_files_are_in_sync():
    drift = [
        f"{destination.relative_to(ROOT)} vs {source.relative_to(ROOT)}"
        for source, destination in sync_module.planned_copies()
        if not destination.is_file() or destination.read_bytes() != source.read_bytes()
    ]
    assert not drift, (
        "shared scaffold drift (edit the canonical copy under templates/_shared "
        "and run scripts/sync_template_shared.py): " + "; ".join(drift)
    )


def test_common_dependency_pins_agree():
    pins = json.loads((SHARED / "versions.json").read_text())["pins"]
    pin_pattern = re.compile(r"^([A-Za-z0-9_.\[\]-]+)==([A-Za-z0-9_.]+)\s*$")
    mismatches = []
    for template in sync_module.discover_templates():
        lock = template / "template" / "requirements.lock"
        if not lock.is_file():
            continue
        for line in lock.read_text().splitlines():
            match = pin_pattern.match(line.strip())
            if not match:
                continue
            package, version = match.groups()
            if package in pins and version != pins[package]:
                mismatches.append(
                    f"{template.name}: {package}=={version} "
                    f"(canonical {pins[package]})"
                )
    assert not mismatches, "pin drift vs templates/_shared/versions.json: " + "; ".join(
        mismatches
    )


def test_manifest_files_exist_and_nothing_orphaned():
    manifest = json.loads((SHARED / "manifest.json").read_text())
    for relative in manifest["files"]:
        assert (SHARED / "files" / relative).is_file(), relative
    declared = {str(Path(relative)) for relative in manifest["files"]}
    on_disk = {
        str(path.relative_to(SHARED / "files"))
        for path in (SHARED / "files").rglob("*")
        if path.is_file()
    }
    assert declared == on_disk, (
        f"manifest/files mismatch: only-declared={sorted(declared - on_disk)} "
        f"only-on-disk={sorted(on_disk - declared)}"
    )
