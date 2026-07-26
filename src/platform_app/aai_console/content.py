"""Load and resolve the guided content ladder."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from markupsafe import Markup, escape

from .config import IDENTIFIER_KEYS, ConsoleConfig

CONTENT_DIR = Path(__file__).resolve().parent / "content"

_PLACEHOLDER = re.compile(r"\$\{identifier:([a-z_]+)\}")
_INLINE_CODE = re.compile(r"`([^`\n]+)`")


def inline_code(text: str) -> Markup:
    """Render `backticked` spans as <code>, escaping everything else.

    Deliberately the only markup the content may use. Escaping happens first, so content
    can never inject an element; the replacement then operates on already-safe text.
    """
    escaped = str(escape(text))
    return Markup(_INLINE_CODE.sub(r"<code>\1</code>", escaped))


class ContentError(RuntimeError):
    """Raised when the content ladder is malformed."""


@dataclass(frozen=True)
class Block:
    lang: str
    code: str


@dataclass(frozen=True)
class Choice:
    id: str
    label: str
    detail: str


@dataclass(frozen=True)
class Step:
    id: str
    title: str
    body: str
    source: str | None = None
    blocks: tuple[Block, ...] = ()
    checklist: tuple[str, ...] = ()
    choices: tuple[Choice, ...] = ()
    verify: tuple[str, ...] = ()
    generate: str | None = None


@dataclass(frozen=True)
class Track:
    id: str
    title: str
    subtitle: str
    glyph: str
    summary: str
    steps: tuple[Step, ...] = field(default=())


def resolve_placeholders(text: str, config: ConsoleConfig) -> str:
    def replace(match: re.Match[str]) -> str:
        return config.identifier(match.group(1))

    return _PLACEHOLDER.sub(replace, text)


def placeholder_keys(text: str) -> set[str]:
    return set(_PLACEHOLDER.findall(text))


def _parse_step(raw: dict) -> Step:
    blocks = tuple(
        Block(lang=str(b.get("lang", "bash")), code=str(b["code"]).rstrip("\n"))
        for b in raw.get("blocks", [])
    )
    choices = tuple(
        Choice(id=str(c["id"]), label=str(c["label"]), detail=str(c.get("detail", "")))
        for c in raw.get("choices", [])
    )
    return Step(
        id=str(raw["id"]),
        title=str(raw["title"]),
        body=str(raw.get("body", "")).strip(),
        source=raw.get("source"),
        blocks=blocks,
        checklist=tuple(str(item) for item in raw.get("checklist", [])),
        choices=choices,
        verify=tuple(str(item) for item in raw.get("verify", [])),
        generate=raw.get("generate"),
    )


def load_tracks(path: Path | None = None) -> tuple[Track, ...]:
    source = path or (CONTENT_DIR / "onboarding.yml")
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or "tracks" not in document:
        raise ContentError(f"{source} does not define a tracks list")

    tracks: list[Track] = []
    seen_track_ids: set[str] = set()
    for raw in document["tracks"]:
        steps = tuple(_parse_step(step) for step in raw.get("steps", []))
        track = Track(
            id=str(raw["id"]),
            title=str(raw["title"]),
            subtitle=str(raw.get("subtitle", "")),
            glyph=str(raw.get("glyph", "step")),
            summary=str(raw.get("summary", "")).strip(),
            steps=steps,
        )
        if track.id in seen_track_ids:
            raise ContentError(f"duplicate track id {track.id!r}")
        seen_track_ids.add(track.id)

        step_ids = [step.id for step in steps]
        if len(step_ids) != len(set(step_ids)):
            raise ContentError(f"duplicate step id in track {track.id!r}")

        for step in steps:
            for block in step.blocks:
                unknown = placeholder_keys(block.code) - IDENTIFIER_KEYS
                if unknown:
                    raise ContentError(
                        f"{track.id}/{step.id} references unknown identifiers: "
                        f"{sorted(unknown)}"
                    )
        tracks.append(track)

    if not tracks:
        raise ContentError(f"{source} defines no tracks")
    return tuple(tracks)
