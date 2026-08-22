"""Credential-free contract tests for the documentation index and decision log.

The docs map (`docs/README.md`) is this repository's retrieval layer: grep
plus one enforced index instead of a link graph. These tests keep it honest —
every top-level document stays reachable, every link resolves, and decision
records stay dated and immutable so upstream and downstream clones can both
add entries without ever colliding on a name or an index line.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
DECISIONS = DOCS / "decisions"
# Dated names cannot collide when upstream and a clone both record decisions.
_ENTRY_NAME = re.compile(r"^\d{4}-\d{2}-\d{2}-[a-z0-9][a-z0-9-]*\.md$")
_MD_LINK = re.compile(r"\]\(([^)#\s]+\.md)\)")
_REQUIRED_HEADINGS = ("## Context", "## Decision", "## Consequences")


def test_every_top_level_doc_is_linked_from_the_docs_index():
    index = (DOCS / "README.md").read_text(encoding="utf-8")
    for path in sorted(DOCS.glob("*.md")):
        if path.name == "README.md":
            continue
        assert f"]({path.name})" in index, (
            f"docs/README.md does not link docs/{path.name}; an unlinked "
            "document is invisible to developers and agents"
        )
    assert (
        "](decisions/README.md)" in index
    ), "docs/README.md must link the decision log"


def test_docs_index_links_resolve_and_never_enumerate_decision_entries():
    index = (DOCS / "README.md").read_text(encoding="utf-8")
    targets = _MD_LINK.findall(index)
    assert targets, "docs/README.md links no markdown documents"
    for target in targets:
        if target.startswith(("http://", "https://")):
            continue
        assert (
            DOCS / target
        ).is_file(), f"docs/README.md links {target}, which does not exist"
        is_entry_link = target.startswith("decisions/") and _ENTRY_NAME.match(
            Path(target).name
        )
        assert not is_entry_link, (
            "docs/README.md must not enumerate decision entries; clones add "
            f"their own and an entry list conflicts on every merge: {target}"
        )


def test_decision_log_follows_the_dated_convention():
    assert (
        DECISIONS / "README.md"
    ).is_file(), "docs/decisions/README.md must document the convention"
    entries = sorted(
        path for path in DECISIONS.glob("*.md") if path.name != "README.md"
    )
    assert entries, "docs/decisions/ must carry at least the adoption record"
    for path in entries:
        assert _ENTRY_NAME.match(path.name), (
            f"{path.name}: decision records are named YYYY-MM-DD-slug.md; "
            "date prefixes cannot collide when upstream and a clone both "
            "add entries"
        )
        text = path.read_text(encoding="utf-8")
        for heading in _REQUIRED_HEADINGS:
            assert heading in text, f"{path.name} is missing '{heading}'"
        assert re.search(
            r"^Status:", text, re.MULTILINE
        ), f"{path.name} has no Status line; supersession must stay legible"
    readme = (DECISIONS / "README.md").read_text(encoding="utf-8")
    assert not re.search(r"\]\(\d{4}-\d{2}-\d{2}-", readme), (
        "docs/decisions/README.md must not enumerate entries; "
        "`ls docs/decisions/` is the index"
    )
