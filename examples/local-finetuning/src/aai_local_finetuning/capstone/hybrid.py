"""Hybrid baseline that keeps policy decisions outside the language model."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .policy import DEFAULT_AS_OF, ReadinessPolicyEngine
from .schemas import (
    ApplicationManifest,
    HybridExplanation,
    HybridReview,
    ReadinessCheck,
)

ExplanationRenderer = Callable[[ReadinessCheck], str]


def _deterministic_explanation(check: ReadinessCheck) -> str:
    if check.remediation_text:
        return f"{check.evidence} {check.remediation_text}"
    return check.evidence


def build_hybrid_review(
    manifest: ApplicationManifest | Mapping[str, Any],
    *,
    renderer: ExplanationRenderer | None = None,
    renderer_name: str | None = None,
    engine: ReadinessPolicyEngine | None = None,
) -> HybridReview:
    """Attach rendered explanations to immutable deterministic results.

    A tiny local model can be supplied as ``renderer``.  It receives one frozen
    check and can return prose only; it has no channel for changing the status,
    check result, severity, rule identity, or remediation identifier.  Empty or
    failing renderer calls fall back to policy text so offline use remains safe.
    """

    policy_engine = engine or ReadinessPolicyEngine(as_of=DEFAULT_AS_OF)
    deterministic_review = policy_engine.review(manifest)
    render = renderer or _deterministic_explanation
    name = renderer_name or ("policy_text" if renderer is None else "local_renderer")
    explanations: list[HybridExplanation] = []
    for check in deterministic_review.checks:
        try:
            rendered = render(check).strip()
        except Exception:  # A renderer failure cannot weaken a policy decision.
            rendered = ""
        if not rendered:
            rendered = _deterministic_explanation(check)
        explanations.append(
            HybridExplanation(
                check_name=check.name,
                text=rendered,
                renderer=name,
            )
        )
    return HybridReview(
        schema_version="1.0.0",
        deterministic_review=deterministic_review,
        explanations=tuple(explanations),
    )
