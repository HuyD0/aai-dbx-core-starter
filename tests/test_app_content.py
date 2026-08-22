"""The console must render the lifecycle docs/ already teaches — not re-author it.

Prose may be paraphrased. Commands may not: a developer pastes them into a shell, so a
command that has drifted from the documentation is worse than no console at all.
"""

import json
import re
from pathlib import Path

import pytest

from aai_console.config import IDENTIFIER_KEYS, ConsoleConfig
from aai_console.content import load_tracks, placeholder_keys, resolve_placeholders

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "src" / "platform_app"
IDENTIFIERS = json.loads((ROOT / "platform-identifiers.json").read_text())

# Commands the console composes rather than quotes. Each needs a written reason, so
# adding one is a deliberate act rather than a quiet escape from the drift check.
SYNTHESIZED_COMMANDS = {
    "make examples-list": (
        "Makefile target; the docs describe it in prose, not a fenced block."
    ),
    f"export DATABRICKS_HOST={IDENTIFIERS['databricks_host']}": (
        "The docs `source scripts/platform-env.sh` so a clone never has a workspace "
        "host pasted into prose (tests/test_smoke.py enforces that). The console is "
        "hosted and a viewer may have no checkout to source, so it substitutes the "
        "value it was configured with instead."
    ),
    "export DATABRICKS_AUTH_TYPE=azure-cli": (
        "Second half of the same substitution; platform-env.sh exports it too."
    ),
}


def _collapse(text: str) -> str:
    return " ".join(text.split())


def _config() -> ConsoleConfig:
    return ConsoleConfig(
        identifiers={key: IDENTIFIERS[key] for key in IDENTIFIER_KEYS},
        hosted=False,
        app_name=None,
    )


def _command_blocks():
    for track in load_tracks():
        for step in track.steps:
            for block in step.blocks:
                yield track.id, step.id, step.source, block.code


def test_every_command_block_is_verbatim_from_a_cited_document():
    config = _config()
    checked = 0
    for track_id, step_id, source, code in _command_blocks():
        resolved = _collapse(resolve_placeholders(code, config))
        if resolved in SYNTHESIZED_COMMANDS:
            continue
        assert source, f"{track_id}/{step_id} has a command block but cites no source"
        document = ROOT / source
        assert document.is_file(), f"{track_id}/{step_id} cites missing {source}"
        haystack = _collapse(document.read_text(encoding="utf-8"))
        assert (
            resolved in haystack
        ), f"{track_id}/{step_id} command has drifted from {source}:\n  {resolved}"
        checked += 1
    assert checked >= 10, "the drift check covered suspiciously few commands"


def test_every_cited_source_exists():
    for track in load_tracks():
        for step in track.steps:
            if step.source:
                assert (ROOT / step.source).is_file(), f"{step.id} cites {step.source}"


def test_template_choices_match_the_shipped_template_catalog():
    """A choice naming a template that no longer exists generates a broken command."""
    catalog = {
        entry.name
        for entry in (ROOT / "templates").iterdir()
        if entry.is_dir() and (entry / "databricks_template_schema.json").is_file()
    }
    offered = {
        choice.id
        for track in load_tracks()
        for step in track.steps
        for choice in step.choices
    }
    assert offered, "the console offers no templates"
    assert offered <= catalog, f"console offers unknown templates: {offered - catalog}"


def test_placeholders_are_known_identifier_keys():
    for _, _, _, code in _command_blocks():
        assert placeholder_keys(code) <= IDENTIFIER_KEYS


def test_no_environment_identifier_literal_appears_under_the_app():
    """`source_code_path` uploads only src/platform_app, so it cannot read the fixture.

    Every environment value must arrive as bundle-supplied environment configuration. A
    literal here is what makes a clone into another tenant silently wrong rather than
    loudly broken.
    """
    values = [
        value
        for key, value in IDENTIFIERS.items()
        if not key.startswith("$") and isinstance(value, str) and value
    ]
    # Unity Catalog components of the volume path: a literal `dbx_dev` is just as
    # tenant-specific as the full path, and the whole-value scan misses it.
    volume = str(IDENTIFIERS["sdk_artifact_volume"])
    values += [part for part in volume.split("/") if part and part != "Volumes"]
    # Workspace nicknames live only in prose (AGENTS.md section 3), so no fixture
    # value can catch them, but they are exactly what a clone cannot use.
    values += ["dbx-dev", "dbx-uat"]

    offenders = []
    for path in APP_DIR.rglob("*"):
        if not path.is_file() or path.suffix in {".pyc"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for value in values:
            if value in text:
                offenders.append(f"{path.relative_to(ROOT)} contains {value!r}")
    assert not offenders, "identifier literals must not be baked into the app:\n" + (
        "\n".join(offenders)
    )


def test_content_uses_only_backtick_markup():
    """The renderer supports inline `code` and nothing else; anything richer renders
    as literal punctuation to the developer."""
    banned = re.compile(r"(?<!`)(\*\*|__|\[[^\]]+\]\()")
    for track in load_tracks():
        for step in track.steps:
            for field in (track.summary, step.body, *step.checklist):
                assert not banned.search(field or ""), f"{step.id}: unsupported markup"


@pytest.mark.parametrize("key", sorted(IDENTIFIER_KEYS))
def test_identifier_keys_exist_in_the_fixture(key):
    """Keeps the console's closed identifier list honest against the single source."""
    assert IDENTIFIERS.get(key), f"platform-identifiers.json is missing {key}"
