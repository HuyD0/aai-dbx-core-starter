"""Track registry.

The console shell (sidebar, main pane, composer) is generic; content arrives through
this registry. Onboarding is the first registered track set; a lifecycle-and-cost
dashboard is intended to be a later registration rather than a rewrite.

Deliberately thin — a protocol and a list. Anything more abstract would be speculative
until the second track set actually exists.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .content import Track, load_tracks


@runtime_checkable
class TrackSource(Protocol):
    """A named group of tracks the console can render."""

    id: str

    def tracks(self) -> tuple[Track, ...]:  # pragma: no cover - protocol definition
        ...


class OnboardingTracks:
    id = "onboarding"

    def tracks(self) -> tuple[Track, ...]:
        return load_tracks()


class TrackRegistry:
    def __init__(self, sources: list[TrackSource]) -> None:
        self._sources = sources

    @classmethod
    def default(cls) -> TrackRegistry:
        return cls([OnboardingTracks()])

    def register(self, source: TrackSource) -> None:
        self._sources.append(source)

    def tracks(self) -> tuple[Track, ...]:
        collected: list[Track] = []
        for source in self._sources:
            collected.extend(source.tracks())
        return tuple(collected)
